import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PartitionTree import PartitionTree

class RegionExplainer:
    def __init__(self, mlm_model, X_train, Y_train, max_depth=3):
        self.mlm = mlm_model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_mean = np.mean(X_train, axis=0)

        # Initialize the Visualization-Ready Partition Tree
        self.partition_tree = PartitionTree(Y_train, max_depth=max_depth)

    def visualize_partitions(self):
        """Wrapper to trigger tree visualization."""
        self.partition_tree.visualize(plot_dominating=True)

    def get_region_rules(self, query_y):
        """Query rules for a specific point."""
        if query_y.ndim > 1: query_y = query_y.flatten()
        leaf_node = self.partition_tree.query_region(query_y)

        neighbors = leaf_node.points
        if len(neighbors) == 0:
            neighbors = np.array([(leaf_node.bounds_min + leaf_node.bounds_max)/2])

        local_ideal_idx = np.argmin(np.linalg.norm(neighbors, axis=1))

        return {
            'bounds_min': leaf_node.bounds_min,
            'bounds_max': leaf_node.bounds_max,
            'dominating_point': neighbors[local_ideal_idx],
            'neighbors': neighbors
        }

    def calculate_region_saliency(self, x_query, desired_region_center, n_masks=10000, p_keep=0.5):
      """
      Explains which variables prevent x_query from reaching the 'desired_region_center'.
      (Logic remains consistent with RISE/MLM perturbation)
      """
      N_vars = len(x_query)
      
      # Determine mask baseline: Use the X corresponding to the desired_region_center (Target)
      # This answers: "What if this variable was already at the target value?"
      
      # Find matching row in Y_train for the desired_region_center
      # Robust lookup using nearest neighbor in Objective Space
      dists = np.linalg.norm(self.Y_train - desired_region_center, axis=1)
      target_idx = np.argmin(dists)
      
      if dists[target_idx] < 1e-4:
          # Use the Target's X as the baseline
          mask_mean = self.X_train[target_idx]
      else:
          # Fallback: Use region median (as implemented previously)
          y_query = self.mlm.predict(x_query.reshape(1, -1))[0]
          region_info = self.get_region_rules(y_query)
          
          neighbor_indices = []
          for neighbor in region_info['neighbors']:
              matches = np.where(np.all(np.isclose(self.Y_train, neighbor), axis=1))[0]
              if len(matches) > 0:
                  neighbor_indices.append(matches[0])
          
          if neighbor_indices:
              mask_mean = np.median(self.X_train[neighbor_indices], axis=0)
          else:
              mask_mean = self.X_mean
              print("Warning: No neighbors found and Target not found. Using global mean.")

      # 1. Generate Masks
      masks = np.random.choice([0, 1], size=(n_masks, N_vars), p=[1-p_keep, p_keep])

      # 2. Perturb X using the REGION SPECIFIC MEAN
      X_perturbed = masks * x_query + (1 - masks) * mask_mean

      # 3. Predict Y using MLM
      Y_pred = self.mlm.predict(X_perturbed)

      # 4. Calculate Distance to Target
      dists = np.linalg.norm(Y_pred - desired_region_center, axis=1)

      # 5. Compute Saliency
      saliency = np.zeros(N_vars)
      for i in range(N_vars):
          on_idx = masks[:, i] == 1
          off_idx = masks[:, i] == 0

          if np.sum(on_idx) > 0 and np.sum(off_idx) > 0:
              # Positive score = Feature increases distance (Hinders convergence)
              score = np.mean(dists[on_idx]) - np.mean(dists[off_idx])
              saliency[i] = score

      return saliency

    def plot_explanation(self, x_query, x_labels=None, target_y=None, save=False, filename='explanation_plot.svg'):
      """
      Visualizes the Region Rules and Saliency Map.
      
      Args:
          x_query: Query point in decision space.
          x_labels: Labels for decision variables. Default is None.
          target_y: Target point in objective space. Default is None.
          save (bool): Whether to save the plot in SVG format. Default is False.
          filename (str): Filename for the SVG file. Default is 'explanation_plot.svg'.
      """
      # 1. Predict current Y
      y_current = self.mlm.predict(x_query.reshape(1, -1))[0]

      # 2. Get Current Region Rules (Using strict partitions)
      current_info = self.get_region_rules(y_current)

      # 3. Define Target
      if target_y is None:
          target_point = current_info['dominating_point']
          title_suffix = "Relative to Local Dominating Point"
      else:
          target_point = target_y
          title_suffix = "Relative to User-Selected Region"

      # 4. Calculate Saliency
      saliency = self.calculate_region_saliency(x_query, target_point)

      # 5. Visualization
      if x_labels is None:
          x_labels = [f"x{i+1}" for i in range(len(x_query))]

      norm_saliency = saliency / (np.max(np.abs(saliency)) + 1e-9)
      colors = ['red' if s > 0 else 'green' for s in norm_saliency]

      fig, ax = plt.subplots(1, 2, figsize=(16, 6))

      # Plot A: The Rules (Objective Space)
      n_obj = len(y_current)
      ax[0].plot(range(n_obj), y_current, 'o-', label='Current Solution', color='black')
      ax[0].plot(range(n_obj), target_point, 'x--', label='Target Dominating Point', color='green')

      # Fill between the EXACT tree bounds
      ax[0].fill_between(range(n_obj), current_info['bounds_min'], current_info['bounds_max'],
                          color='gray', alpha=0.2, label='Strict Region Rules')

      ax[0].set_xticks(range(n_obj))
      ax[0].set_xticklabels([f"Obj {i+1}" for i in range(n_obj)])
      ax[0].set_title(f"Region Constraints (KD-Tree Block)\n(Contains {len(current_info['neighbors'])} solutions)")
      ax[0].legend()
      ax[0].grid(True, alpha=0.3)

      # Plot B: The Saliency (Decision Space)
      ax[1].bar(x_labels, norm_saliency, color=colors)
      ax[1].axhline(0, color='black', lw=1)
      ax[1].set_title(f"Variable Importance: {title_suffix}\n(Green=Helps reach target, Red=Hinders)")
      ax[1].set_ylabel("Impact on Distance to Target")

      plt.tight_layout()
      
      # Save plot in SVG format if requested
      if save:
          plt.savefig(filename, format='svg')
          print(f"Plot saved as {filename}")
      
      plt.show()


      # Textual Explanation
      print("-" * 50)
      print("STRICT REGION RULES (KD-TREE BOUNDS):")
      for i in range(n_obj):
          print(f"Objective {i+1}: {current_info['bounds_min'][i]:.3f} < f_{i+1} < {current_info['bounds_max'][i]:.3f}")
      print("-" * 50)


