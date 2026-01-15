import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from mlm import MLM
from RegionExplainer import RegionExplainer

def plot_combined_analysis(Y_all, x_trap, y_trap, x_relaxed, y_relaxed, y_target, saliency, feat_names, filename=None):
    """
    Dual-plot visualization:
    1. Objective Space movement with X values and Distances.
    2. Saliency Map showing Blockers/Drivers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Plot 1: Objective Space Movement ---
    f1 = Y_all[:, 0] # Cost
    f2 = Y_all[:, 1] # Deflection
    
    # Background
    ax1.scatter(f1, f2, c='lightgray', s=30, alpha=0.4, label='Feasible Solutions')
    
    # Key Points
    ax1.scatter(y_trap[0], y_trap[1], c='red', s=150, edgecolor='black', zorder=10, label='Trap (Start)')
    ax1.scatter(y_target[0], y_target[1], c='green', marker='*', s=250, edgecolor='black', zorder=10, label='Target (Region Dominating)')
    ax1.scatter(y_relaxed[0], y_relaxed[1], c='orange', marker='s', s=120, edgecolor='black', zorder=15, label='Relaxed Blocker')

    # Arrows
    ax1.annotate("", xy=y_target, xytext=y_trap, 
                 arrowprops=dict(arrowstyle="->", color='blue', lw=1.5, linestyle='--'))
    ax1.annotate("", xy=y_relaxed, xytext=y_trap, 
                 arrowprops=dict(arrowstyle="->", color='orange', lw=2))

    # Distances
    dist_trap = np.linalg.norm(y_trap - y_target)
    
    # Annotations
    def fmt_x(x): return f"[{x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}, {x[3]:.2f}]"
    
    ax1.text(y_trap[0], y_trap[1] + 0.002, 
             f"TRAP\nDist={dist_trap:.4f}", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

    ax1.set_xlabel("Objective 1: Cost ($)")
    ax1.set_ylabel("Objective 2: Deflection (in)")
    ax1.set_title("Objective Space: Region-Based Movement")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    # --- Plot 2: Saliency Map ---
    norm_saliency = saliency / (np.max(np.abs(saliency)) + 1e-9)
    colors = ['#d62728' if s > 0 else '#2ca02c' for s in norm_saliency]
    
    bars = ax2.bar(feat_names, norm_saliency, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(0, color='black', lw=1)
    
    ax2.set_ylabel("Influence Score (Normalized)")
    ax2.set_title("Saliency Map: Blockers vs Drivers")
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#d62728', lw=4),
                    Line2D([0], [0], color='#2ca02c', lw=4)]
    ax2.legend(custom_lines, ['Blocker (Hinders Region Target)', 'Driver (Helps Region Target)'])
    
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        print(f" -> Visualization saved to {filename}")
    else:
        plt.show()

def run_experiment():
    print("="*60)
    print("EXPERIMENT 1: Physics-Informed Validation (Region Target)")
    print("="*60)

    # --- Step 1: Solving ---
    print("\n[Step 1] Solving Welded Beam Problem (NSGA-II)...")
    problem = get_problem("welded_beam")
    algorithm = NSGA2(pop_size=200, seed=1)
    res = minimize(problem, algorithm, ('n_gen', 300), seed=1, verbose=False)
    X, Y, G = res.X, res.F, res.G
    
    # --- Step 2: Select Trap ---
    print("\n[Step 2] Selecting 'Constraint Trap' Solution...")
    sorted_idx = np.argsort(Y[:, 0])
    trap_idx = sorted_idx[0]
    x_trap, y_trap, g_trap = X[trap_idx], Y[trap_idx], G[trap_idx]
    
    feat_names = ['h (Weld)', 'l (Length)', 't (Height)', 'b (Width)']
    constr_names = ['Shear (g1)', 'Bend (g2)', 'Buckl (g3)', 'Defl (g4)']
    
    print(f" -> Trap Solution: Cost={y_trap[0]:.4f}, Defl={y_trap[1]:.4f}")

    # --- Step 3: Run PGDS ---
    print("\n[Step 3] Running PGDS (RegionExplainer)...")
    mlm = MLM(rp_number=100); mlm.fit(X, Y)
    
    # Initialize Explainer
    explainer = RegionExplainer(mlm, X, Y, max_depth=2)
    
    # NEW: Get Target from Region Rules (Dominating Point)
    print(" -> Identifying Region Dominating Point...")
    ctx = explainer.get_region_rules(y_trap)
    target_y = ctx['dominating_point']
    
    print(f" -> Region Target (Local Utopia): {target_y}")
    
    saliency = explainer.calculate_region_saliency(x_trap, target_y, n_masks=2000)

    # --- Step 4: Interpret ---
    blocker_idx = np.argmax(saliency)
    blocker_name = feat_names[blocker_idx]
    print(f"\n[Step 4] Primary Blocker Identified: {blocker_name}")

    # --- Step 5: Physics Verification ---
    print("\n[Step 5] Physics-Informed Verification")
    
    # Perturb Blocker
    delta = x_trap[blocker_idx] * 0.10
    x_relaxed = x_trap.copy()
    x_relaxed[blocker_idx] += delta 
    x_relaxed = np.clip(x_relaxed, *problem.bounds())

    # Evaluate
    g_relaxed = problem.evaluate(x_relaxed.reshape(1,-1), return_values_of=["G"])[0]
    y_relaxed = problem.evaluate(x_relaxed.reshape(1,-1), return_values_of=["F"])[0]
    
    # Compare Constraints (Monitor ALL)
    print(f"    {'Constraint':<12} | {'Trap':<10} | {'Relaxed':<10} | {'Effect'}")
    print("-" * 55)
    for i, c_name in enumerate(constr_names):
        diff = g_relaxed[i] - g_trap[i]
        # Active if close to 0 (e.g., > -0.01)
        is_active = g_trap[i] > -0.01
        
        effect = "No Change"
        if abs(diff) > 1e-4:
            if diff < 0: effect = "Relaxed (Safer)"
            else: effect = "Tightened"
            
        marker = "<-- ACTIVE" if is_active else ""
        print(f"    {c_name:<12} | {g_trap[i]:.4f}     | {g_relaxed[i]:.4f}     | {effect} {marker}")

    # --- Step 6: Visualization ---
    plot_combined_analysis(Y, x_trap, y_trap, x_relaxed, y_relaxed, target_y, 
                           saliency, feat_names, 
                           filename="experiments/img/welded_beam_region_target.svg")

if __name__ == "__main__":
    run_experiment()