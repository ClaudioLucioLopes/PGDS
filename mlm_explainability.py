import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ==========================================
# 1. Minimal Learning Machine (Surrogate)
# ==========================================
class MLMRegressor(BaseEstimator, RegressorMixin):
    """
    Minimal Learning Machine (MLM) for Regression.
    Maps input distances to output distances to preserve geometry.
    """
    def __init__(self, rp_number=None, random_state=42):
        self.rp_number = rp_number
        self.random_state = random_state
        self.B = None
        self.rp_X = None
        self.rp_y = None

    def fit(self, X, y):
        """
        Fits the MLM model.
        Args:
            X (np.ndarray): Input decision variables.
            y (np.ndarray): Output objective values.
        """
        N = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # 1. Reference Point Selection
        if self.rp_number is None:
            n_rps = int(0.1 * N) # Default 10%
        elif self.rp_number <= 1.0:
            n_rps = int(self.rp_number * N)
        else:
            n_rps = int(self.rp_number)
        
        # Ensure at least one RP and typically RP < N
        n_rps = max(1, min(n_rps, N))

        idx = rng.choice(N, n_rps, replace=False)
        self.rp_X = X[idx, :]
        self.rp_y = y[idx, :]

        # 2. Distance Matrix Calculation
        D_x = cdist(X, self.rp_X) # Input distances
        D_y = cdist(y, self.rp_y) # Output distances

        # 3. Solve for Regression Matrix B (D_y = D_x * B)
        # Using Moore-Penrose Pseudoinverse for stability
        # B = (Dx^T Dx)^-1 Dx^T Dy
        try:
            self.B = np.linalg.pinv(D_x) @ D_y
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            self.B = np.linalg.lstsq(D_x, D_y, rcond=None)[0]

        return self

    def predict(self, X):
        """
        Predicts outputs y for new inputs X using Multilateration.
        """
        return np.array([self._get_single_output(x) for x in X])

    def _get_single_output(self, x):
        # 1. Calculate input distances for query x
        d_in = cdist(x[np.newaxis, :], self.rp_X)
        
        # 2. Predict output distances
        d_out_pred = d_in @ self.B
        
        # 3. Multilateration (Position Estimation)
        # Find y that minimizes difference between actual distances to RPs and predicted distances
        
        # Objective function for least_squares
        def loss(y_candidate):
            # Calculate actual distance from candidate y to all Reference Outputs
            d_actual = cdist(y_candidate[np.newaxis, :], self.rp_y).flatten()
            return (d_actual**2 - d_out_pred.flatten()**2)

        # Initial guess: Mean of reference outputs
        y0 = self.rp_y.mean(axis=0)
        
        # Optimization
        res = least_squares(loss, y0, method='lm')
        return res.x

# ==========================================
# 2. MLM Distance Explainer (Algorithm 1)
# ==========================================
class MLMDistanceExplainer:
    """
    Explains the proximity of a decision vector to a target objective point
    by perturbing decision variables and observing geometric shifts via MLM.
    """
    def __init__(self, mlm_model, X_train):
        """
        Args:
            mlm_model: Fitted MLMRegressor instance.
            X_train: The training data (Archive) used to calculate mean for imputation.
        """
        self.mlm = mlm_model
        self.X_mean = np.mean(X_train, axis=0)
        self.n_features = X_train.shape[1]

    def explain(self, x_query, target_point, n_masks=1000, p_keep=0.5, seed=None):
        """
        Runs Algorithm 1: MLM-Distance Saliency Calculation.
        
        Args:
            x_query (np.ndarray): The solution to explain (Decision Space).
            target_point (np.ndarray): The target in Objective Space (e.g., Ideal Point).
            n_masks (int): Number of random masks.
            p_keep (float): Probability of keeping a variable (1 - masking rate).
            seed (int): Random seed.
            
        Returns:
            importance (np.ndarray): Saliency scores for each decision variable.
                                     Positive high value = Variable is critical for proximity.
        """
        rng = np.random.RandomState(seed)
        
        # 1. Generate Binary Masks (L x n_vars)
        # 1 = Keep, 0 = Mask (replace with mean)
        masks = rng.choice([0, 1], size=(n_masks, self.n_features), p=[1-p_keep, p_keep])
        
        # 2. Perturb Input
        # masked_input = x_query * mask + mean * (1 - mask)
        x_perturbed = (x_query * masks) + (self.X_mean * (1 - masks))
        
        # 3. Fast Prediction using MLM
        # We can batch predict since MLM.predict supports matrices
        y_pred_perturbed = self.mlm.predict(x_perturbed)
        
        # 4. Calculate Distance Shifts
        # Distance from predicted perturbed outputs to the Target Point
        distances = cdist(y_pred_perturbed, target_point[np.newaxis, :]).flatten()
        
        # 5. Variable Attribution (Correlation)
        # We correlate the presence of the variable (mask=1) with the Distance.
        # Ideally, we want variables where Mask=1 -> Distance is LOW (Close to target).
        # This implies a NEGATIVE correlation.
        # To make "Saliency" intuitive (High Score = Important), we invert the correlation.
        
        importance = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            # Calculate correlation between mask column i and resulting distances
            if np.std(masks[:, i]) == 0:
                corr = 0 # Variance is zero, correlation undefined
            else:
                corr = np.corrcoef(masks[:, i], distances)[0, 1]
            
            # Invert sign: 
            # If keeping variable (1) makes distance smaller (better), corr is negative.
            # We want this to be a POSITIVE importance score.
            importance[i] = -corr
            
        return importance

