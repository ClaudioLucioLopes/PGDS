import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from mlm import MLM
from PartitionTree import PartitionTree
from RegionExplainer import RegionExplainer


def create_step_by_step_visualization(X, Y, mlm_model, partition_tree, query_idx, filename='experiments/step_by_step_explanation.svg', equal_aspect=True):
    """
    Creates a comprehensive step-by-step visualization showing the mechanics of the explanation method.
    
    Args:
        X: Input features
        Y: Objective values
        mlm_model: Trained MLM model
        partition_tree: PartitionTree instance
        query_idx: Index of the query point to explain
        filename: Output filename for the visualization
        equal_aspect: Whether to enforce equal aspect ratio for 3D plots. Default is False.
    """
    # Get the query point
    x_q = X[query_idx]
    y_q = Y[query_idx]
    
    # Predict current Y
    y_current = mlm_model.predict(x_q.reshape(1, -1))[0]
    
    # Find the region containing the query point and get its dominating point
    query_region = None
    for block in partition_tree.partitions:
        if (np.all(y_q >= block['bounds_min']) and np.all(y_q <= block['bounds_max'])):
            query_region = block
            break
    
    # Use the dominating point from the partition tree
    dominating_point = query_region['dominating_point'] if query_region else np.min(Y, axis=0)
    
    # Get region information for plotting
    explainer = RegionExplainer(mlm_model, X, Y, max_depth=partition_tree.max_depth)
    current_info = explainer.get_region_rules(y_current)
    
    # Calculate saliency using the partition tree's dominating point
    saliency = explainer.calculate_region_saliency(x_q, dominating_point)

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(20, 16))
    
    # ========== PLOT 1: Full 3D view with all partitions ==========
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Draw all partitions
    cmap = plt.get_cmap('tab20c')
    for block in partition_tree.partitions:
        color = cmap((block['id'] - 1) % 20)
        _draw_cuboid_3d(ax1, block['bounds_min'], block['bounds_max'], color)
        center = (block['bounds_min'] + block['bounds_max']) / 2
        ax1.text(center[0], center[1], center[2], str(block['id']), fontsize=9, weight='bold')
    
    # Plot all points
    ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='gray', s=10, alpha=0.3, label='All Solutions')
    
    # Plot dominating points for all regions
    for block in partition_tree.partitions:
        dom_pt = block['dominating_point']
        ax1.scatter(dom_pt[0], dom_pt[1], dom_pt[2], c='red', marker='s', s=80, 
                   edgecolors='black', linewidths=1, alpha=0.7, zorder=5)
    
    # Add a single legend entry for all dominating points
    ax1.scatter([], [], c='red', marker='s', s=80, edgecolors='black', linewidths=1, 
               label='Dominating Points', alpha=0.7)
    
    # Highlight the query point
    ax1.scatter(y_q[0], y_q[1], y_q[2], c='blue', marker='o', s=200, edgecolors='black', linewidths=2, label='Selected Point', zorder=10)
    
    ax1.set_xlabel('Objective 1')
    ax1.set_ylabel('Objective 2')
    ax1.set_zlabel('Objective 3')
    ax1.set_title('Step 1: User Chooses a Point of Interest\n(Point shown in blue, red squares show dominating points for each region)', fontsize=12, weight='bold')
    ax1.legend(loc='upper right')
    ax1.view_init(elev=20, azim=45)
    
    
    # ========== PLOT 2: Same view with direction to dominating point ==========
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    
    # Draw all partitions (same as Plot 1)
    for block in partition_tree.partitions:
        color = cmap((block['id'] - 1) % 20)
        _draw_cuboid_3d(ax2, block['bounds_min'], block['bounds_max'], color)
        center = (block['bounds_min'] + block['bounds_max']) / 2
        ax2.text(center[0], center[1], center[2], str(block['id']), fontsize=9, weight='bold')
    
    # Plot all points (same as Plot 1)
    ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='gray', s=10, alpha=0.3, label='All Solutions')
    
    # Highlight the query point
    ax2.scatter(y_q[0], y_q[1], y_q[2], c='blue', marker='o', s=200, edgecolors='black', linewidths=2, label='Selected Point', zorder=10)
    
    # Plot the dominating point
    ax2.scatter(dominating_point[0], dominating_point[1], dominating_point[2], 
               c='red', marker='s', s=200, edgecolors='black', linewidths=2, label='Dominating Point (Target)', zorder=10)
    
    # Draw arrow from query point to dominating point
    ax2.plot([y_q[0], dominating_point[0]], 
            [y_q[1], dominating_point[1]], 
            [y_q[2], dominating_point[2]], 
            'r--', linewidth=3, alpha=0.8, label='Direction to Target', zorder=9)
    
    ax2.set_xlabel('Objective 1')
    ax2.set_ylabel('Objective 2')
    ax2.set_zlabel('Objective 3')
    ax2.set_title(f'Step 2: User Wants to Reach the Dominating Point\n(Red dashed line shows direction from selected point to target in Region {query_region["id"] if query_region else "?"})', fontsize=12, weight='bold')
    ax2.legend(loc='upper right')
    ax2.view_init(elev=20, azim=45)
    
    if equal_aspect:
        # Enforce equal aspect ratio for both 3D plots
        # Find global bounds for all points to be plotted
        all_plotted_points = np.vstack([Y, [dominating_point]])
        min_bounds = np.min(all_plotted_points, axis=0)
        max_bounds = np.max(all_plotted_points, axis=0)
        max_range = (max_bounds - min_bounds).max() / 2.0
        mid_x = (max_bounds[0] + min_bounds[0]) * 0.5
        mid_y = (max_bounds[1] + min_bounds[1]) * 0.5
        mid_z = (max_bounds[2] + min_bounds[2]) * 0.5
        
        for ax in [ax1, ax2]:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    
    # ========== PLOT 3: Region Constraints (from plot_explanation) ==========
    ax3 = fig.add_subplot(2, 2, 3)
    
    n_obj = len(y_current)
    ax3.plot(range(n_obj), y_current, 'o-', label='Current Solution', color='blue', linewidth=2, markersize=10)
    ax3.plot(range(n_obj), dominating_point, 'x--', label='Target Dominating Point', color='red', linewidth=2, markersize=12)
    
    # Fill between the EXACT tree bounds
    ax3.fill_between(range(n_obj), current_info['bounds_min'], current_info['bounds_max'],
                     color='gray', alpha=0.2, label='Region Constraints')
    
    ax3.set_xticks(range(n_obj))
    ax3.set_xticklabels([f"Obj {i+1}" for i in range(n_obj)])
    ax3.set_title(f"Step 3: Understanding the Region Constraints\n(Contains {len(current_info['neighbors'])} solutions)", fontsize=12, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Objective Value')
    
    # ========== PLOT 4: Variable Importance (from plot_explanation) ==========
    ax4 = fig.add_subplot(2, 2, 4)
    
    x_labels = [f"x{i+1}" for i in range(len(x_q))]
    norm_saliency = saliency / (np.max(np.abs(saliency)) + 1e-9)
    colors = ['red' if s > 0 else 'green' for s in norm_saliency]
    
    bars = ax4.bar(x_labels, norm_saliency, color=colors, edgecolor='black', linewidth=1.5)
    ax4.axhline(0, color='black', lw=2)
    ax4.set_title("Step 4: Which Variables Help or Hinder?\n(Green = Helps reach target, Red = Hinders)", fontsize=12, weight='bold')
    ax4.set_ylabel("Impact on Distance to Target")
    ax4.set_xlabel("Decision Variables")
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, norm_saliency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8, weight='bold')
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(filename, format='svg')
    print(f"\nStep-by-step visualization saved as {filename}")
    plt.close()
    
    return saliency, dominating_point


def create_adjusted_query_visualization(X, Y, mlm_model, partition_tree, query_idx, var_idx, direction, value, dominating_point=None, filename='experiments/adjusted_query.svg', equal_aspect=True):
    """
    Creates a visualization showing the effect of manually adjusting a variable.
    
    Args:
        var_idx: Index of the variable to change (0-based).
        direction: 'inc' (increase) or 'dec' (decrease).
        value: The absolute amount to change the variable by.
        dominating_point: The target point used for calculation (optional, if None it will be calculated).
        equal_aspect: Whether to enforce equal aspect ratio for 3D plots. Default is False.
    """
    # Get the query point
    x_q = X[query_idx]
    y_q = Y[query_idx]
    
    # Predict current Y
    y_current = mlm_model.predict(x_q.reshape(1, -1))[0]
    
    # Find region and dominating point if not provided
    query_region = None
    for block in partition_tree.partitions:
        if (np.all(y_q >= block['bounds_min']) and np.all(y_q <= block['bounds_max'])):
            query_region = block
            break
            
    if dominating_point is None:
        dominating_point = query_region['dominating_point'] if query_region else np.min(Y, axis=0)
    
    # Get region info for Line Plot
    explainer = RegionExplainer(mlm_model, X, Y, max_depth=partition_tree.max_depth)
    current_info = explainer.get_region_rules(y_current)
    
    # --- Adjustment Logic ---
    current_val = x_q[var_idx]
    var_name = f"x{var_idx+1}"
    
    # Calculate new value
    delta = value if direction == 'inc' else -value
    raw_new_val = current_val + delta
    
    # Clamp to bounds
    feat_min = np.min(X[:, var_idx])
    feat_max = np.max(X[:, var_idx])
    final_val = np.clip(raw_new_val, feat_min, feat_max)
    
    # Create adjusted point
    x_new = x_q.copy()
    x_new[var_idx] = final_val
    y_new = mlm_model.predict(x_new.reshape(1, -1))[0]
    
    # Calculate improvement
    dist_old = np.linalg.norm(y_current - dominating_point)
    dist_new = np.linalg.norm(y_new - dominating_point)
    improvement = dist_old - dist_new
    pct_imp = (improvement / dist_old) * 100 if dist_old > 0 else 0
    
    action_str = "Increased" if direction == 'inc' else "Decreased"
    title_str = f"Manual Adjustment: {var_name}\n{action_str} by {value:.4f} ({current_val:.4f} -> {final_val:.4f})"
    
    # --- Visualization ---
    fig = plt.figure(figsize=(20, 10))
    
    # Plot 1: Region Constraints (Line Plot) - Left
    ax1 = fig.add_subplot(1, 2, 1)
    n_obj = len(y_current)
    
    # Plot Lines
    ax1.plot(range(n_obj), y_current, 'o-', label=f'Original (Dist: {dist_old:.4f})', color='blue', linewidth=2, markersize=10)
    ax1.plot(range(n_obj), dominating_point, 'x--', label='Target', color='red', linewidth=2, markersize=12)
    ax1.plot(range(n_obj), y_new, '*-.', label=f'Adjusted (Dist: {dist_new:.4f})', color='green', linewidth=2, markersize=14)
    
    # Fill bounds
    ax1.fill_between(range(n_obj), current_info['bounds_min'], current_info['bounds_max'],
                     color='gray', alpha=0.2, label='Region Bounds')
    
    ax1.set_xticks(range(n_obj))
    ax1.set_xticklabels([f"Obj {i+1}" for i in range(n_obj)])
    ax1.set_title(f"Region View: {title_str}\n(Distance Improvement: {pct_imp:.1f}%)", fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Objective Value')
    
    # Plot 2: Global 3D View - Right
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    cmap = plt.get_cmap('tab20c')
    
    # Draw partitions
    for block in partition_tree.partitions:
        color = cmap((block['id'] - 1) % 20)
        _draw_cuboid_3d(ax2, block['bounds_min'], block['bounds_max'], color)
        center = (block['bounds_min'] + block['bounds_max']) / 2
        ax2.text(center[0], center[1], center[2], str(block['id']), fontsize=9, weight='bold')
        
    ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='gray', s=10, alpha=0.3, label='All Solutions')
    
    # Plot points
    ax2.scatter(y_current[0], y_current[1], y_current[2], c='blue', marker='o', s=150, label='Original', zorder=10)
    ax2.scatter(dominating_point[0], dominating_point[1], dominating_point[2], c='red', marker='s', s=150, label='Target', zorder=10)
    ax2.scatter(y_new[0], y_new[1], y_new[2], c='green', marker='*', s=300, label='Adjusted', zorder=10)
    
    # Arrows
    # 1. Original -> Adjusted (Movement)
    ax2.plot([y_current[0], y_new[0]], [y_current[1], y_new[1]], [y_current[2], y_new[2]], 'k-', lw=2, label='Movement')
    
    # 2. Adjusted -> Target (New Path)
    ax2.plot([y_new[0], dominating_point[0]], [y_new[1], dominating_point[1]], [y_new[2], dominating_point[2]], 'g--', alpha=0.5, label=f'New Dist: {dist_new:.4f}')
    
    # 3. Original -> Target (Old Path) - Requested by User
    ax2.plot([y_current[0], dominating_point[0]], [y_current[1], dominating_point[1]], [y_current[2], dominating_point[2]], 'r--', alpha=0.5, label=f'Old Dist: {dist_old:.4f}')
    
    ax2.set_title("Global View: Movement in Objective Space", fontsize=12, weight='bold')
    ax2.set_xlabel('Objective 1'); ax2.set_ylabel('Objective 2'); ax2.set_zlabel('Objective 3')
    ax2.legend()
    ax2.view_init(elev=20, azim=45)
    
    if equal_aspect:
        # Enforce equal aspect ratio for 3D plot to prevent visual distortion
        # Find global bounds for all points to be plotted
        all_plotted_points = np.vstack([Y, [dominating_point], [y_new]])
        min_bounds = np.min(all_plotted_points, axis=0)
        max_bounds = np.max(all_plotted_points, axis=0)
        max_range = (max_bounds - min_bounds).max() / 2.0
        mid_x = (max_bounds[0] + min_bounds[0]) * 0.5
        mid_y = (max_bounds[1] + min_bounds[1]) * 0.5
        mid_z = (max_bounds[2] + min_bounds[2]) * 0.5
        
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', bbox_inches='tight')
    print(f"\nAdjusted query visualization saved as {filename}")
    print(f"Adjustment: {var_name} {action_str} by {value}")
    print(f"Distance Improvement: {dist_old:.4f} -> {dist_new:.4f} ({pct_imp:.1f}%)")
    plt.close()


def _draw_cuboid_3d(ax, min_b, max_b, color, alpha=0.15):
    """Helper to draw a 3D box."""
    x = [min_b[0], max_b[0]]
    y = [min_b[1], max_b[1]]
    z = [min_b[2], max_b[2]]
    verts = [
        [(x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[1], z[0]), (x[0], y[1], z[0])],  # Bottom
        [(x[0], y[0], z[1]), (x[1], y[0], z[1]), (x[1], y[1], z[1]), (x[0], y[1], z[1])],  # Top
        [(x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[0], z[1]), (x[0], y[0], z[1])],  # Front
        [(x[0], y[1], z[0]), (x[1], y[1], z[0]), (x[1], y[1], z[1]), (x[0], y[1], z[1])],  # Back
        [(x[0], y[0], z[0]), (x[0], y[1], z[0]), (x[0], y[1], z[1]), (x[0], y[0], z[1])],  # Left
        [(x[1], y[0], z[0]), (x[1], y[1], z[0]), (x[1], y[1], z[1]), (x[1], y[0], z[1])]   # Right
    ]
    poly = Poly3DCollection(verts, facecolors=color, alpha=alpha, edgecolor='k', linewidths=0.5, linestyles='--')
    ax.add_collection3d(poly)


if __name__ == "__main__":
    # Load or generate DTLZ7 data
    print("Generating DTLZ7 problem data...")
    ref_dirs = get_reference_directions("energy", 3, 500, seed=42)
    DTLZ7 = get_problem("dtlz7")
    algorithm = NSGA3(pop_size=500, ref_dirs=ref_dirs)
    res = minimize(DTLZ7, algorithm, seed=1, termination=('n_gen', 100), verbose=False)
    X = res.X
    Y = res.F
    
    print(f"Data generated: X shape {X.shape}, Y shape {Y.shape}")
    
    # Train MLM model
    print("Training MLM model...")
    mlm_model = MLM(rp_number=int(X.shape[0]/2))
    mlm_model.fit(X, Y)
    print("MLM model trained.")
    
    # Create partition tree
    print("Creating partition tree...")
    partition_tree = PartitionTree(Y, max_depth=2)
    print(f"Partition tree created with {len(partition_tree.partitions)} regions.")
    
    # Create visualization for a point in region 1
    query_idx = 10  # You can change this to select different points
    # query_idx = 190  # You can change this to select different points
    # query_idx = 95  # You can change this to select different points
    # query_idx = 80  # This is one that need the equal_aspect=True
    
    print(f"X at query_idx {query_idx}: {X[query_idx]}")
    x_q = X[query_idx]
    
    print(f"\nCreating step-by-step visualization for point index {query_idx}...")
    saliency, dominating_point = create_step_by_step_visualization(X, Y, mlm_model, partition_tree, query_idx,filename=f'experiments/img/step_by_step_explanation_{query_idx}.svg')
    
    # --- Automatic Verification based on Saliency ---
    print("\n--- Running Automatic Verification ---")
    
    # 1. Reuse Saliency and Dominating Point from Visualization (Ensures consistency)
    # (Skipping recalculation to avoid stochastic noise differences)
    
    # 2. Find Target X (Best guess from dataset)
    # Use nearest neighbor in Objective Space to be robust against floating point differences
    dists = np.linalg.norm(Y - dominating_point, axis=1)
    target_idx = np.argmin(dists)
    
    if dists[target_idx] < 1e-4:
        target_x = X[target_idx]
        # print(f"Target X found at index {target_idx} (Dist: {dists[target_idx]:.2e})")
    else:
        print(f"Warning: Exact Target X not found (Min Dist: {dists[target_idx]:.2e}). Using nearest neighbor.")
        target_x = X[target_idx]

    print(f"X at target_idx {target_idx}: {X[target_idx]}")

    # 3. Test Blocker (Most Hindering Variable -> Max Positive Saliency)
    blocker_idx = np.argmax(saliency)
    blocker_delta = target_x[blocker_idx] - x_q[blocker_idx]
    blocker_dir = 'inc' if blocker_delta > 0 else 'dec'
    
    print(f"Blocker identified: x{blocker_idx+1} (Saliency: {saliency[blocker_idx]:.4f})")
    print(f"Adjusting x{blocker_idx+1} to Target: {blocker_dir} by {abs(blocker_delta):.4f}")
    
    create_adjusted_query_visualization(X, Y, mlm_model, partition_tree, query_idx, 
                                      var_idx=blocker_idx, 
                                      direction=blocker_dir, 
                                      value=abs(blocker_delta), 
                                      dominating_point=dominating_point,
                                      filename=f'experiments/img/adjusted_query_blocker_{query_idx}.svg')

    # 4. Test Driver (Most Helpful Variable -> Min Negative Saliency)
    driver_idx = np.argmin(saliency)
    driver_delta = target_x[driver_idx] - x_q[driver_idx]
    driver_dir = 'inc' if driver_delta > 0 else 'dec'
    
    print(f"Driver identified: x{driver_idx+1} (Saliency: {saliency[driver_idx]:.4f})")
    print(f"Adjusting x{driver_idx+1} to Target: {driver_dir} by {abs(driver_delta):.4f}")
    
    create_adjusted_query_visualization(X, Y, mlm_model, partition_tree, query_idx, 
                                      var_idx=driver_idx, 
                                      direction=driver_dir, 
                                      value=abs(driver_delta), 
                                      dominating_point=dominating_point,
                                      filename=f'experiments/img/adjusted_query_driver_{query_idx}.svg')
                                      
    print("\nDone!")
