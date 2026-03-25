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
    if len(knee_indices) > 0:
        knee_bundle = Y_all[knee_indices]
        for y in knee_bundle:
            plt.plot(X_axis, y, color='#1f77b4', alpha=0.4, lw=1)
    # Highlight Knee Target
    plt.plot(X_axis, knee_target, color='blue', lw=3, marker='o', label='Knee Target (Ideal)')

    # 3. Extreme Region (Red)
    if len(extreme_indices) > 0:
        ext_bundle = Y_all[extreme_indices]
        for y in ext_bundle:
            plt.plot(X_axis, y, color='#d62728', alpha=0.4, lw=1)
    # Highlight Extreme Target
    plt.plot(X_axis, extreme_target, color='red', lw=3, marker='*', markersize=12, label='Extreme Target (Best f10)')

    plt.xlabel("Objective ID")
    plt.ylabel("Objective Value (Minimization)")
    plt.title("Parallel Coordinate Plot: Region Discovery (WFG3 - 10 Objs)")
    plt.xticks(X_axis)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=1, label='All Solutions'),
        Line2D([0], [0], color='blue', lw=3, label='Knee Region (Balanced)'),
        Line2D([0], [0], color='red', lw=3, label='Extreme Region (Bias f10)')
    ]
    plt.legend(handles=legend_elements)
    
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" -> PCP Visualization saved to {filename}")
    else:
        plt.show()

def plot_pcp_sol_comparison(Y_all, knee_indices, extreme_indices, knee_target, extreme_target, filename=None):
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
    if len(knee_indices) > 0:
        knee_bundle = Y_all[knee_indices]
        for y in knee_bundle:
            plt.plot(X_axis, y, color='#1f77b4', alpha=0.4, lw=1)
    # Highlight Knee Target
    plt.plot(X_axis, knee_target, color='blue', lw=3, marker='o', label='Knee Target (Ideal)')

    # 3. Extreme Region (Red)
    if len(extreme_indices) > 0:
        ext_bundle = Y_all[extreme_indices]
        for y in ext_bundle:
            plt.plot(X_axis, y, color='#d62728', alpha=0.4, lw=1)
    # Highlight Extreme Target
    plt.plot(X_axis, extreme_target, color='red', lw=3, marker='*', markersize=12, label='Extreme Target (Best f10)')

    plt.xlabel("Objective ID")
    plt.ylabel("Objective Value (Minimization)")
    plt.title("Parallel Coordinate Plot: Region Discovery (WFG3 - 10 Objs)")
    plt.xticks(X_axis)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=1, label='All Solutions'),
        Line2D([0], [0], color='blue', lw=3, label='Knee solution (Balanced)'),
        Line2D([0], [0], color='red', lw=3, label='Extreme solution (Bias f10)')
    ]
    plt.legend(handles=legend_elements)
    
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" -> PCP Visualization saved to {filename}")
    else:
        plt.show()


def plot_saliency_comparison(saliency_knee, saliency_extreme, feature_names, filename=None):
    """
    Side-by-side comparison of what matters in each region.
    Normalized to [-1, 1] for readability.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    # --- Robust Normalization ---
    scale_knee = np.max(np.abs(saliency_knee)) + 1e-9
    norm_knee = saliency_knee / scale_knee
    
    scale_ext = np.max(np.abs(saliency_extreme)) + 1e-9
    norm_ext = saliency_extreme / scale_ext
    
    # --- Knee Region Plot ---
    colors_k = ['#2ca02c' if s < 0 else '#d62728' for s in norm_knee]
    ax1.bar(feature_names, norm_knee, color=colors_k, edgecolor='black', alpha=0.8)
    ax1.set_title(f"Knee Region Saliency\n(Target: Ideal Point)\n(Max Raw Impact: {scale_knee:.2e})")
    ax1.set_ylabel("Relative Importance (Normalized)")
    ax1.axhline(0, color='black', lw=1)
    ax1.tick_params(axis='x', rotation=90)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- Extreme Region Plot ---
    colors_e = ['#2ca02c' if s < 0 else '#d62728' for s in norm_ext]
    ax2.bar(feature_names, norm_ext, color=colors_e, edgecolor='black', alpha=0.8)
    ax2.set_title(f"Extreme Region Saliency\n(Target: Local Best)\n(Max Raw Impact: {scale_ext:.2e})")
    ax2.axhline(0, color='black', lw=1)
    ax2.tick_params(axis='x', rotation=90)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#d62728', lw=4, label='Blocker (Hinders Target)'),
        Line2D([0], [0], color='#2ca02c', lw=4, label='Driver (Helps Target)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" -> Saliency comparison saved to {filename}")
    else:
        plt.show()

def log_top_features(saliency, feature_names, title, top_k=5):
    """
    Helper to print a clean table of the top influential variables.
    """
    # Normalize for display
    norm_sal = saliency / (np.max(np.abs(saliency)) + 1e-9)
    
    # Get indices of top k absolute values
    top_indices = np.argsort(np.abs(norm_sal))[::-1][:top_k]
    
    print(f"\n    [Top {top_k} Influential Variables: {title}]")
    print(f"    {'Variable':<10} | {'Type':<10} | {'Imp. Score':<12} | {'Role'}")
    print("    " + "-"*50)
    
    for idx in top_indices:
        score = norm_sal[idx]
        name = feature_names[idx]
        
        # WFG3 logic: z1-z19 (0-18) are Position/Distance? 
        # Usually k=position, l=distance. We just print ID.
        var_type = "Var" 
        
        role = "BLOCKER" if score > 0 else "DRIVER"
        print(f"    {name:<10} | {var_type:<10} | {score:+.4f}       | {role}")

def run_experiment():
    print("="*60)
    print("EXPERIMENT 2: Many-Objective WFG3 (Automated Discovery)")
    print("="*60)

    # --- Step 1: Solve WFG3 (10 Objs, 20 Vars) ---
    print("\n[Step 1] Solving WFG3 (10 Obj) using NSGA-III...")
    n_obj = 10
    n_var = 20
    problem = get_problem("wfg3", n_var=n_var, n_obj=n_obj)
    
    # Reference Directions for 10D
    ref_dirs = get_reference_directions("energy", n_obj, 600, seed=42)
    
    algorithm = NSGA3(pop_size=len(ref_dirs), ref_dirs=ref_dirs, seed=42)

    
    res = minimize(problem, algorithm, ('n_gen', 400), seed=1, verbose=True)
    X, Y = res.X, res.F
    print(f" -> Convergence reached. Archive size: {len(X)}")

    # --- Step 2: Initialize Explainer ---
    print("\n[Step 2] Training PGDS Model...")
    mlm = MLM(rp_number=int(len(X)*0.5)) 
    mlm.fit(X, Y)
    
    explainer = RegionExplainer(mlm, X, Y, max_depth=4) 

    # --- Step 3: Automated Region Selection ---
    print("\n[Step 3] Automated 'Blind' Targeting...")
    
    # A. Knee Point (Closest to Ideal)
    ideal_dists = np.linalg.norm(Y, axis=1)
    knee_idx = np.argmin(ideal_dists)
    x_knee = X[knee_idx]
    y_knee = Y[knee_idx]
    
    # B. Extreme Point (Best f10)
    ext_idx = np.argmin(Y[:, 9])
    x_ext = X[ext_idx]
    y_ext = Y[ext_idx]
    
    # --- DETAILED LOGGING (Solution Details) ---
    print("-" * 60)
    print("DETAILED POINT LOG:")
    print(f"1. KNEE POINT (Index {knee_idx}):")
    print(f"   Dist to Ideal: {ideal_dists[knee_idx]:.6f}")
    print(f"   Objectives (f1..f10): {np.array2string(y_knee, precision=4, suppress_small=True)}")
    print(f"   Variables (z1..z20):  {np.array2string(x_knee, precision=4, suppress_small=True)}")
    
    print(f"\n2. EXTREME POINT (Index {ext_idx}):")
    print(f"   Best f10 Value: {y_ext[9]:.6f}")
    print(f"   Objectives (f1..f10): {np.array2string(y_ext, precision=4, suppress_small=True)}")
    print(f"   Variables (z1..z20):  {np.array2string(x_ext, precision=4, suppress_small=True)}")
    print("-" * 60)

    # Get Region Contexts
    ctx_knee = explainer.get_region_rules(y_knee)
    ctx_ext = explainer.get_region_rules(y_ext)
    
    # --- Step 4: Saliency Calculation ---
    print("\n[Step 4] Generating Saliency with Adaptive Targets...")
    
    # Knee Target Logic (Force Ideal)
    dist_to_local_best = np.linalg.norm(y_knee - ctx_knee['dominating_point'])
    if dist_to_local_best < 1e-3:
        print(" -> Knee: Auto-switching Target to GLOBAL IDEAL (0..0) for contrast.")
        target_knee = np.zeros(n_obj) 
    else:
        target_knee = ctx_knee['dominating_point']
        
    sal_knee = explainer.calculate_region_saliency(x_knee, target_knee, n_masks=2000)
    sal_ext = explainer.calculate_region_saliency(x_ext, ctx_ext['dominating_point'], n_masks=2000)

    # --- Step 5: Detailed Saliency Log ---
    feat_names = [f"x{i+1}" for i in range(n_var)]
    log_top_features(sal_knee, feat_names, "Knee Region")
    log_top_features(sal_ext, feat_names, "Extreme Region")

    # --- Step 6: Visualization ---
    
    # Helper to find indices for PCP
    def get_indices_from_neighbors(neighbors, full_Y):
        indices = []
        for row in neighbors:
            matches = np.where((full_Y == row).all(axis=1))[0]
            if len(matches) > 0: indices.append(matches[0])
        return indices

    print("\n -> Mapping regions for PCP...")
    knee_indices = get_indices_from_neighbors(ctx_knee['neighbors'], Y)
    ext_indices = get_indices_from_neighbors(ctx_ext['neighbors'], Y)
    
    plot_pcp_comparison(Y, knee_indices, ext_indices, 
                        target_knee, ctx_ext['dominating_point'],
                        filename="experiments/img/wfg3_pcp_v2.svg")
    
    plot_pcp_sol_comparison(Y, knee_indices, ext_indices, 
                        y_knee, y_ext,
                        filename="experiments/img/wfg3_pcp_sol_v2.svg")

    plot_saliency_comparison(sal_knee, sal_ext, feat_names, 
                             filename="experiments/img/wfg3_saliency_v2.svg")

if __name__ == "__main__":
    run_experiment()