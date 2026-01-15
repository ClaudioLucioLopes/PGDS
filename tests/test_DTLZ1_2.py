# ==========================================
# 3. Unit Tests & Experiments (DTLZ1/DTLZ2)
# ==========================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add parent directory to path to import mlm_explainability
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from pymoo.problems import get_problem
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    import matplotlib.pyplot as plt
    from mlm_explainability import MLMRegressor, MLMDistanceExplainer
    from sklearn.preprocessing import MinMaxScaler
    from scipy.spatial.distance import cdist
    import numpy as np

    def run_experiment(problem_name, n_var, n_obj, explanation_name):
        print(f"\n{'='*60}")
        print(f"Running Experiment: {explanation_name} ({problem_name})")
        print(f"{'='*60}")

        # --- 1. Generate Data via Optimization ---
        problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=18)
        
        algorithm = NSGA3(pop_size=200, ref_dirs=ref_dirs)
        
        print("Optimizing to generate Pareto Archive...")
        res = minimize(problem, algorithm, termination=('n_gen', 100), seed=1, verbose=False)
        
        X_archive = res.X
        Y_archive = res.F
        print(f"Archive Shape: X={X_archive.shape}, Y={Y_archive.shape}")

        # --- 2. Train MLM Surrogate ---
        print("Training MLM Surrogate...")
        # Scale X for better numerical stability in distance calculations
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_archive)
        
        mlm = MLMRegressor(rp_number=0.30) # Use 30% reference points
        mlm.fit(X_scaled, Y_archive)
        
        # Verify Accuracy
        y_pred = mlm.predict(X_scaled)
        mse = np.mean((Y_archive - y_pred)**2)
        print(f"MLM Training MSE: {mse:.6f}")

        # --- 3. Run Distance Explainer ---
        explainer = MLMDistanceExplainer(mlm, X_scaled)
        
        # Select a query point (e.g., the one closest to Ideal Point usually 0,0,0)
        ideal_point = np.zeros(n_obj)
        dists_to_ideal = cdist(Y_archive, ideal_point[np.newaxis, :]).flatten()
        best_idx = np.argmin(dists_to_ideal)
        
        query_x = X_scaled[best_idx]
        query_y = Y_archive[best_idx]
        
        print(f"\nExplaining Solution Index: {best_idx}")
        print(f"Query Objectives: {query_y}")
        print(f"Target: Ideal Point {ideal_point}")
        
        # Run Explanation
        saliency = explainer.explain(query_x, ideal_point, n_masks=1000, p_keep=0.5)
        
        # --- 4. Validation Analysis ---
        # DTLZ Problems Structure:
        # DTLZ1/DTLZ2: Last k variables (k = n_var - n_obj + 1) control CONVERGENCE (Distance to front).
        # First variables control DIVERSITY (Position on front).
        # To get close to Ideal Point (0,0,0), Convergence variables are critical.
        
        k = n_var - n_obj + 1
        diversity_vars = np.arange(n_obj - 1)
        convergence_vars = np.arange(n_obj - 1, n_var)
        
        print("\nSaliency Scores:")
        for i, s in enumerate(saliency):
            type_v = "Convergence" if i in convergence_vars else "Diversity"
            print(f"Var {i:2d} ({type_v}): {s:.4f}")

        avg_conv_saliency = np.mean(saliency[convergence_vars])
        avg_div_saliency = np.mean(saliency[diversity_vars])
        
        print(f"\nAvg Saliency (Convergence Vars): {avg_conv_saliency:.4f}")
        print(f"Avg Saliency (Diversity Vars)  : {avg_div_saliency:.4f}")
        
        if avg_conv_saliency > avg_div_saliency:
            print(">> SUCCESS: Convergence variables deemed more important for proximity to Ideal Point.")
        else:
            print(">> WARNING: Diversity variables scored higher (Check noise or model fit).")

        return saliency

    # Run DTLZ2 Test (Spherical Front)
    # n_obj=3, n_var=12. k=10. Vars 0-1 are Diversity, Vars 2-11 are Convergence.
    saliency_dtlz2 = run_experiment("dtlz2", n_var=12, n_obj=3, explanation_name="DTLZ2 Convergence Check")

    # Run DTLZ1 Test (Linear/Planar Front)
    # n_obj=3, n_var=7. k=5. Vars 0-1 are Diversity, Vars 2-6 are Convergence.
    saliency_dtlz1 = run_experiment("dtlz1", n_var=7, n_obj=3, explanation_name="DTLZ1 Convergence Check")