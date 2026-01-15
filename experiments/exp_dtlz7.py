import sys
from pathlib import Path
# Add parent directory to path to import mlm_explainability
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from mlm import MLM
from PartitionTree import PartitionTree
from RegionExplainer import RegionExplainer


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from experiments.create_step_by_step_visualization import create_step_by_step_visualization, create_adjusted_query_visualization


ref_dirs = ref_dirs = get_reference_directions("energy", 3, 500, seed=42)


# Instantiate the DTLZ7 problem with 3 objectives
DTLZ7 = get_problem("dtlz7")


algorithm = NSGA3(pop_size=500,
                  ref_dirs=ref_dirs)

res = minimize(DTLZ7,
               algorithm,
               seed=1,
               termination=('n_gen', 100),
               verbose=True)
X = res.X
Y = res.F


print("Shape of X (input features):")
print(X.shape)
print("\nFirst 5 rows of X:")
print(X[:5])

print("\nShape of Y (objective values):")
print(Y.shape)
print("\nFirst 5 rows of Y:")
print(Y[:5])



##### Step 1: Initialize the MLM model and fit it to the DTLZ7 data
mlm_model = MLM(rp_number=int(X.shape[0]/2))

# Fit the model to the DTLZ7 input features (X) and objective values (Y)
mlm_model.fit(X, Y)

print("NN_MLM model initialized and fitted successfully.")

# Make predictions using the MLM model
predictions = mlm_model.predict(X)

# Calculate metrics for each objective
metrics_str = ""
for i in range(Y.shape[1]):
    mse = mean_squared_error(Y[:, i], predictions[:, i])
    r2 = r2_score(Y[:, i], predictions[:, i])
    mae = mean_absolute_error(Y[:, i], predictions[:, i])
    metrics_str += f"\nObjective {i+1}: R²={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}"

# Create a 3D scatter plot for Y (original) and predictions
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original Y points in blue
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='red', marker='x', alpha=0.6, label='Original Y')

# Plot predicted points in red
ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], c='black', marker='o', alpha=0.6, label='Predicted')

ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')
ax.set_title(f'Y vs. Predicted Objective Values - MLM{metrics_str}')
ax.legend(loc='lower left')

# Adjust the camera view
ax.view_init(elev=20, azim=45)
# ax.view_init(elev=30, azim=30) # Example angles

plt.grid(True)
filename = 'experiments/plot_dtlz7_mlm.svg'
plt.savefig(filename, format='svg')
print(f"Plot saved as {filename}")


##### Step 2: Visualize the Partition of the Pareto Front

partition_tree = PartitionTree(Y, max_depth=2)
partition_tree.visualize(plot_dominating=True,color_regions=False, save=True, filename='experiments/partition_plot_dtlz7_ed.svg')
print(partition_tree.get_dominating_points_per_block())

for block in partition_tree.partitions:
    # print(block['points'])
    print('min:',np.min(block['points'],axis=0))

##### Step 3: Explain a solution

# 1. Initialize Explainer
explainer = RegionExplainer(mlm_model, X, Y)

# 2. Select a solution to explain (e.g., index 10)
query_idx = 10
x_q = X[query_idx]

# 3. User Adjustment Simulation:
# "I want to move to the region dominated by the true Ideal Point (0,0,0)"
user_target = np.zeros(Y.shape[1])

# 4. Run Explanation
explainer.plot_explanation(x_q, target_y=user_target, save=True, filename='experiments/explanation_dtlz7_v1.svg')
explainer.plot_explanation(x_q, save=True, filename='experiments/explanation_dtlz7_v2.svg')


##### Step 4: Create Step-by-Step Visualization

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


# Get the query point
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
current_info = explainer.get_region_rules(y_current)

# Calculate saliency
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
    colors_bar = ['red' if s > 0 else 'green' for s in norm_saliency]
    
    bars = ax4.bar(x_labels, norm_saliency, color=colors_bar, edgecolor='black', linewidth=1.5)
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
    step_by_step_filename = 'experiments/step_by_step_explanation.svg'
    plt.savefig(step_by_step_filename, format='svg', bbox_inches='tight')
    print(f"\nStep-by-step visualization saved as {step_by_step_filename}")
    plt.close()

    print(f"\nCreating adjusted query visualization for point index {query_idx}...")
    
    # Manual Adjustment: Decrease x1 by 0.1
    print("Running Manual Adjustment: Decrease x1 by 0.1...")
    create_adjusted_query_visualization(X, Y, mlm_model, partition_tree, query_idx, 
                                      var_idx=0, 
                                      direction='dec', 
                                      value=0.1, 
                                      filename='experiments/adjusted_query.svg')
    
    print("\nDone!")

print("\n=== All visualizations completed! ===")
