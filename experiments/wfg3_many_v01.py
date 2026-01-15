import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from mlm import MLM
from RegionExplainer import RegionExplainer

def plot_pcp_comparison(Y_all, knee_indices, extreme_indices, knee_target, extreme_target, filename=None):
    """
    Parallel Coordinate Plot (PCP) comparing the 'Knee Region' vs 'Extreme Region'.
    Visualizes the 'Cognitive Drought' solution.
    """
    n_obj = Y_all.shape[1]
    X_axis = range(1, n_obj + 1)
    
    plt.figure(figsize=(15, 8))
    
    # 1. Background (Light Gray)
    subset_idx = np.random.choice(Y_all.shape[0], size=min(200, Y_all.shape[0]), replace=False)
    for y in Y_all[subset_idx]:
        plt.plot(X_axis, y, color='lightgray', alpha=0.3, lw=0.5)
        
    # 2. Knee Region (Blue)
    # Ensure knee_indices is a list of integers
    knee_bundle = Y_all[knee_indices]
    for y in knee_bundle:
        plt.plot(X_axis, y, color='#1f77b4', alpha=0.4, lw=1)
    # Highlight Target
    plt.plot(X_axis, knee_target, color='blue', lw=3, marker='o', label='Knee Target (Balanced)')

    # 3. Extreme Region (Red)
    ext_bundle = Y_all[extreme_indices]
    for y in ext_bundle:
        plt.plot(X_axis, y, color='#d62728', alpha=0.4, lw=1)
    # Highlight Target
    plt.plot(X_axis, extreme_target, color='red', lw=3, marker='*', markersize=12, label='Extreme Target (Best f10)')

    plt.xlabel("Objective ID")
    plt.ylabel("Objective Value (Minimization)")
    plt.title("Parallel Coordinate Plot: Automated Region Discovery (WFG3 - 10 Objs)")
    plt.xticks(X_axis)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Custom Legend
    legend_elements = [
        plt.Line2D([0], [0], color='lightgray', lw=1, label='All Solutions'),
        plt.Line2D([0], [0], color='blue', lw=3, label='Knee Region (Balanced)'),
        plt.Line2D([0], [0], color='red', lw=3, label='Extreme Region (Bias f10)')
    ]
    plt.legend(handles=legend_elements)
    
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" -> Visualization saved to {filename}")
    else:
        plt.show()

def plot_saliency_comparison(saliency_knee, saliency_extreme, feature_names, filename=None):
    """
    Side-by-side comparison of what matters in each region.
    Normalized to [-1, 1] for readability.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    # --- Robust Normalization ---
    # We divide by the maximum absolute value to scale everything between -1 and 1
    # Adding 1e-9 avoids division by zero if saliency is all zeros
    scale_knee = np.max(np.abs(saliency_knee)) + 1e-9
    norm_knee = saliency_knee / scale_knee
    
    scale_ext = np.max(np.abs(saliency_extreme)) + 1e-9
    norm_ext = saliency_extreme / scale_ext
    
    # --- Knee Region Plot ---
    # Green = Driver (Negative score, reduces distance)
    # Red = Blocker (Positive score, increases distance)
    colors_k = ['#2ca02c' if s < 0 else '#d62728' for s in norm_knee]
    ax1.bar(feature_names, norm_knee, color=colors_k, edgecolor='black', alpha=0.8)
    ax1.set_title(f"Knee Region Saliency\n(Max Raw Impact: {scale_knee:.2e})") # Show raw scale in title
    ax1.set_ylabel("Relative Importance (Normalized)")
    ax1.axhline(0, color='black', lw=1)
    ax1.tick_params(axis='x', rotation=90)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- Extreme Region Plot ---
    colors_e = ['#2ca02c' if s < 0 else '#d62728' for s in norm_ext]
    ax2.bar(feature_names, norm_ext, color=colors_e, edgecolor='black', alpha=0.8)
    ax2.set_title(f"Extreme Region Saliency\n(Max Raw Impact: {scale_ext:.2e})")
    ax2.axhline(0, color='black', lw=1)
    ax2.tick_params(axis='x', rotation=90)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add legend manually to explain colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#d62728', lw=4, label='Blocker (Hinders Target)'),
        Line2D([0], [0], color='#2ca02c', lw=4, label='Driver (Helps Target)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" -> Saliency comparison saved to {filename}")
    else:
        plt.show()

def run_experiment():
    print("="*60)
    print("EXPERIMENT 3: Many-Objective WFG3 (Automated Discovery)")
    print("="*60)

    # --- Step 1: Solve WFG3 (10 Objs, 20 Vars) ---
    print("\n[Step 1] Solving WFG3 (10 Obj) using NSGA-III...")
    n_obj = 10
    n_var = 20
    problem = get_problem("wfg3", n_var=n_var, n_obj=n_obj)
    
    # Reference Directions for 10D
    ref_dirs = get_reference_directions("energy", n_obj, 600, seed=42)
    
    algorithm = NSGA3(pop_size=len(ref_dirs), ref_dirs=ref_dirs, seed=42)
    
    res = minimize(problem, algorithm, ('n_gen', 400), seed=42, verbose=True)
    X, Y = res.X, res.F
    print(f" -> Convergence reached. Archive size: {len(X)}")

    # --- Step 2: Initialize Explainer ---
    print("\n[Step 2] Training PGDS Model (Blind Partitioning)...")
    
    rp_k = int(len(X) * 0.5) # Use 50% otherwise
        
    mlm = MLM(rp_number=rp_k)
    mlm.fit(X, Y)
    
    # Use higher depth for 10D to get meaningful clusters
    explainer = RegionExplainer(mlm, X, Y, max_depth=3) 

    # --- Step 3: Automated Region Selection ---
    print("\n[Step 3] Automated 'Blind' Targeting...")
    
    # A. Find Knee Solution (Closest to Ideal Point 0,0...0)
    ideal_dists = np.linalg.norm(Y, axis=1)
    knee_idx = np.argmin(ideal_dists)
    x_knee = X[knee_idx]
    y_knee = Y[knee_idx]
    
    # B. Find Extreme Solution (Best in f10)
    ext_idx = np.argmin(Y[:, 9]) # Index 9 is f10
    x_ext = X[ext_idx]
    y_ext = Y[ext_idx]
    
    print(f" -> Knee Point Found (ID {knee_idx}): Dist to Ideal={ideal_dists[knee_idx]:.4f}")
    print(f" -> Extreme Point Found (ID {ext_idx}): Best f10={y_ext[9]:.4f}")

    # Get Region Contexts
    ctx_knee = explainer.get_region_rules(y_knee)
    ctx_ext = explainer.get_region_rules(y_ext)
    
    # --- Step 4: Saliency Calculation ---
    print("\n[Step 4] Generating Context-Aware Saliency...")
    
    sal_knee = explainer.calculate_region_saliency(x_knee, ctx_knee['dominating_point'], n_masks=2000)
    sal_ext = explainer.calculate_region_saliency(x_ext, ctx_ext['dominating_point'], n_masks=2000)

    # --- Step 5: Interpretation & Validation ---
    print("\n -> Comparison of Top Blockers:")
    
    top_k_idx = np.argmax(sal_knee)
    top_e_idx = np.argmax(sal_ext)
    
    # Note: In WFG3, Position variables are 0-(k-1). Distance are k-(n-1).
    # Default k for WFG3 in pymoo is likely 2*(n_obj-1) or similar, need to check doc.
    # Usually first few are position.
    
    print(f"    Knee Region Blocker: Var {top_k_idx}")
    print(f"    Extreme Region Blocker: Var {top_e_idx}")
    
    if top_k_idx != top_e_idx:
        print(" -> SUCCESS: The method identified DIFFERENT drivers for different regions.")
        print("    This confirms Context-Awareness in 10-Objective space.")
    else:
        print(" -> SAME BLOCKERS: The physics might be dominated by a single variable globally.")
    
    # --- Step 6: Visualization ---
    
    # Find indices for PCP plotting
    # We match the neighbor rows back to the original Y
    def find_indices(sub_Y, full_Y):
        indices = []
        for row in sub_Y:
            # Find where this row exists in full_Y
            matches = np.where((full_Y == row).all(axis=1))[0]
            if len(matches) > 0:
                indices.append(matches[0])
        return indices

    knee_indices = find_indices(ctx_knee['neighbors'], Y)
    ext_indices = find_indices(ctx_ext['neighbors'], Y)
    
    plot_pcp_comparison(Y, knee_indices, ext_indices, 
                        ctx_knee['dominating_point'], ctx_ext['dominating_point'],
                        filename="experiments/img/wfg3_pcp.svg")
    
    feat_names = [f"x{i+1}" for i in range(n_var)]
    plot_saliency_comparison(sal_knee, sal_ext, feat_names, 
                             filename="experiments/img/wfg3_saliency.svg")

if __name__ == "__main__":
    run_experiment()