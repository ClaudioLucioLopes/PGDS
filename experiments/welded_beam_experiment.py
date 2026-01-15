import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from mlm import MLM
from RegionExplainer import RegionExplainer

def run_experiment():
    print("="*60)
    print("EXPERIMENT 1: Physics-Informed Validation (Welded Beam)")
    print("="*60)

    # --- Step 1: Solving Welded Beam Problem ---
    print("\n[Step 1] Solving Welded Beam Problem (NSGA-II)...")
    # The standard Welded Beam has 4 vars: h, l, t, b
    # Objectives: f1 (Cost), f2 (Deflection)
    problem = get_problem("welded_beam")
    
    algorithm = NSGA2(pop_size=200, seed=1)
    
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 300),
                   seed=1,
                   verbose=False)
    
    X = res.X
    Y = res.F
    G = res.G
    
    print(f" -> Convergence reached. Pareto Front size: {len(X)}")

    # --- Step 2: Select 'Constraint Trap' Solution ---
    print("\n[Step 2] Selecting 'Constraint Trap' Solution...")
    # Logic: Find the solution with Minimum Cost. 
    # In Welded Beam, Min Cost drives 'h' and 'l' to be as small as possible 
    # until they hit the Shear Stress constraint (g1).
    
    # Sort by Cost (Objective 1)
    sorted_idx = np.argsort(Y[:, 0])
    
    # Check the top 5 cheapest solutions to find the one tightest on Shear Stress (g0 in pymoo index)
    # Note: pymoo constraints are <= 0. Values close to 0 are active.
    candidates = sorted_idx[:5]
    best_trap_idx = candidates[0]
    
    # Just picking the absolute cheapest valid solution usually works
    trap_idx = sorted_idx[0]
    
    x_trap = X[trap_idx]
    y_trap = Y[trap_idx]
    g_trap = G[trap_idx]
    
    # Feature names for Welded Beam
    feat_names = ['h (Weld)', 'l (Length)', 't (Height)', 'b (Width)']
    
    print(f" -> Selected Solution (Index {trap_idx}):")
    print(f"    Variables: {x_trap}")
    print(f"    Objectives [Cost, Defl]: {y_trap}")
    print(f"    Constraints (g<=0):")
    print(f"      Shear Stress (g1): {g_trap[0]:.4f} {'<-- TRAP!' if g_trap[0] > -1.0 else ''}")
    print(f"      Bend Stress  (g2): {g_trap[1]:.4f}")
    print(f"      Pc (Buckling)(g3): {g_trap[2]:.4f}")
    print(f"      Deflection   (g4): {g_trap[3]:.4f}")

    # --- Step 3: Run PGDS (RegionExplainer) ---
    print("\n[Step 3] Running PGDS (RegionExplainer)...")
    
    # 1. Train MLM on the full manifold
    mlm = MLM(rp_number=100)
    mlm.fit(X, Y)
    
    # 2. Initialize Explainer (Builds PartitionTree internally)
    explainer = RegionExplainer(mlm, X, Y, max_depth=2)
    
    # 3. Define Target
    # We want to IMPROVE DEFLECTION (Lower f2).
    # Since Cost and Deflection are conflicting, this implies moving to a region 
    # with Higher Cost but Lower Deflection.
    # Let's target the "Best Deflection" point found in the population.
    target_idx = np.argmin(Y[:, 1])
    target_y = Y[target_idx]
    
    print(f" -> Current Region Dominating Point: (Auto-calculated)")
    print(f" -> User Target (Best Deflection): {target_y}")
    
    # 4. Calculate Saliency
    print(" -> Calculating Saliency Map...")
    saliency = explainer.calculate_region_saliency(x_trap, target_y, n_masks=2000)

    # --- Step 4: Interpret Results ---
    print("\n[Step 4] Saliency Map Analysis")
    
    # Normalize for display
    norm_saliency = saliency / np.max(np.abs(saliency))
    
    print(f"{'Variable':<15} | {'Raw Score':<12} | {'Norm Score':<10} | {'Interpretation'}")
    print("-" * 60)
    for i, name in enumerate(feat_names):
        interp = "Blocker (Hinders Move)" if saliency[i] > 0 else "Driver (Helps Move)"
        print(f"{name:<15} | {saliency[i]:.4f}       | {norm_saliency[i]:.4f}     | {interp}")
        
    blocker_idx = np.argmax(saliency)
    blocker_name = feat_names[blocker_idx]
    print(f"\n -> Primary Identified Blocker: {blocker_name}")
    print("    Hypothesis: Changing this variable is critical to escaping the local constraint trap.")

    # --- Step 5: Physics Verification ---
    print("\n[Step 5] Physics-Informed Verification")
    print(f"    Testing sensitivity of constraints to {blocker_name}...")
    
    # Perturb the blocker variable by +10% and -10%
    delta = x_trap[blocker_idx] * 0.20
    
    x_plus = x_trap.copy()
    x_plus[blocker_idx] += delta
    
    x_minus = x_trap.copy()
    x_minus[blocker_idx] -= delta
    
    # Fix: Reshape inputs to (1, n_vars) so pymoo sees them as a "population" of size 1
    # We also use the public .evaluate() method which is safer than _evaluate
    out_orig = problem.evaluate(x_trap.reshape(1, -1), return_values_of=["G"])
    out_plus = problem.evaluate(x_plus.reshape(1, -1), return_values_of=["G"])
    out_minus = problem.evaluate(x_minus.reshape(1, -1), return_values_of=["G"])
    
    # Extract Shear Stress (Index 0 in constraints)
    # The result is [[g1, g2, g3, g4]], so we take [0][0]
    g_orig = out_orig[0][0]
    g_plus = out_plus[0][0]
    g_minus = out_minus[0][0]
    
    print(f"\n    {'Perturbation':<20} | {blocker_name:<10} | {'Shear Stress (g1)':<20} | {'Feasible?'}")
    print("-" * 75)
    print(f"    {'Original':<20} | {x_trap[blocker_idx]:.4f}     | {g_orig:.4f}               | {g_orig <= 1e-6}")
    print(f"    {'Increase (+10%)':<20} | {x_plus[blocker_idx]:.4f}     | {g_plus:.4f}               | {g_plus <= 1e-6}")
    print(f"    {'Decrease (-10%)':<20} | {x_minus[blocker_idx]:.4f}     | {g_minus:.4f}               | {g_minus <= 1e-6}")
    
    print("\n -> CONCLUSION:")
    # Constraint logic: g <= 0 is feasible. 
    # If decreasing variable makes g > 0 (positive), it violates constraint.
    if g_minus > g_orig and g_minus > 0:
         print(f"    Physics Confirmed: Decreasing {blocker_name} violates Shear Stress (g1 becomes {g_minus:.2f}).")
         print(f"    Increasing {blocker_name} relaxes the constraint (g1 becomes {g_plus:.2f}).")
         print("    PGDS correctly identified that this variable MUST increase to move along the front.")
    else:
         print(f"    Physics relationship unclear (g_minus={g_minus:.4f}). Check constraint definitions.")

    # --- Visualization ---
    explainer.plot_explanation(x_trap, x_labels=feat_names, target_y=target_y, save=True,filename="experiments/img/welded_beam_explanation.svg")
    print("\nExperiment Complete.")

if __name__ == "__main__":
    run_experiment()