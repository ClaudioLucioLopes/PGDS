import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from KDNode import KDNode

class PartitionTree:
    """
    Recursive KD-Tree that builds explicit partitions, calculates local ideal points,
    and supports visualization.
    """
    def __init__(self, points, max_depth=1):
        self.points = points
        self.max_depth = max_depth

        # Calculate global bounds with padding
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        span = max_coords - min_coords
        span = np.where(span == 0, 1e-9, span) # Handle flat data

        self.global_min = min_coords - 0.05 * span
        self.global_max = max_coords + 0.05 * span

        # Build the tree structure
        self.root = self._build_recursive(points, self.global_min, self.global_max, 0)

        # Analyze and store partition data immediately
        self.partitions = []
        self._analyze_partitions(self.root)

    def _build_recursive(self, points, bounds_min, bounds_max, depth):
        n_points, n_dims = points.shape

        if depth >= self.max_depth or n_points <= 1:
            return KDNode(points, bounds_min, bounds_max, depth)

        split_dim = depth % n_dims
        sorted_indices = np.argsort(points[:, split_dim])
        sorted_points = points[sorted_indices]
        median_idx = n_points // 2
        split_val = sorted_points[median_idx, split_dim]

        left_bounds_max = bounds_max.copy()
        left_bounds_max[split_dim] = split_val
        right_bounds_min = bounds_min.copy()
        right_bounds_min[split_dim] = split_val

        node = KDNode(points, bounds_min, bounds_max, depth, axis=split_dim, split_val=split_val)
        node.left = self._build_recursive(sorted_points[:median_idx], bounds_min, left_bounds_max, depth + 1)
        node.right = self._build_recursive(sorted_points[median_idx:], right_bounds_min, bounds_max, depth + 1)

        return node

    def _analyze_partitions(self, node):
        """Recursively traverses the tree to populate self.partitions with block info."""
        if node.is_leaf:
            # 1. Calculate Dominating Point (Local Ideal)
            # Minimized for each axis in this block
            if len(node.points) > 0:
                dominating_point = np.min(node.points, axis=0)

            # 2. Generate Rule String
            n_dims = len(node.bounds_min)
            dims_str = [f"{node.bounds_min[d]:.2f} < f_{d+1} < {node.bounds_max[d]:.2f}" for d in range(n_dims)]
            rule_str = " AND ".join(dims_str)

            # 3. Store Info
            block_id = len(self.partitions) + 1
            self.partitions.append({
                'id': block_id,
                'bounds_min': node.bounds_min,
                'bounds_max': node.bounds_max,
                'points': node.points,
                'dominating_point': dominating_point,
                'rule': rule_str,
                'node': node # Keep reference if needed
            })
        else:
            self._analyze_partitions(node.left)
            self._analyze_partitions(node.right)

    def query_region(self, query_point):
        """Finds the leaf block for a query point."""
        node = self.root
        while not node.is_leaf:
            if query_point[node.axis] < node.split_val:
                node = node.left
            else:
                node = node.right
        return node

    def get_dominating_points_per_block(self):
        """Returns the calculated dominating points for all blocks."""
        return {p['id']: p['dominating_point'] for p in self.partitions}

    def visualize(self, plot_dominating=True, color_regions=True, save=False, filename='partition_plot.svg'):
        """
        Visualizes the partition using the stored partition data.
        
        Args:
            plot_dominating (bool): Whether to plot dominating points. Default is True.
            color_regions (bool): Whether to color the regions. Default is True.
            save (bool): Whether to save the plot in SVG format. Default is False.
            filename (str): Filename for the SVG file. Default is 'partition_plot.svg'.
        """
        n_dims = self.points.shape[1]

        # 1. Setup Plot Context
        do_plot = n_dims <= 3
        fig, ax = None, None
        cmap = plt.get_cmap('tab20c')

        if do_plot:
            if n_dims == 2:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_title(f'KDTree Partition (2D) - Depth {self.max_depth}')
                ax.set_xlabel('Objective 1')
                ax.set_ylabel('Objective 2')
                ax.set_xlim(self.global_min[0], self.global_max[0])
                ax.set_ylim(self.global_min[1], self.global_max[1])
            elif n_dims == 3:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=45, azim=45)
                ax.set_title(f'KDTree Partition (3D) - Depth {self.max_depth}')
                ax.set_xlabel('Obj 1'); ax.set_ylabel('Obj 2'); ax.set_zlabel('Obj 3')
                ax.set_xlim(self.global_min[0], self.global_max[0])
                ax.set_ylim(self.global_min[1], self.global_max[1])
                ax.set_zlim(self.global_min[2], self.global_max[2])
        else:
            print(f"Data is {n_dims}-dimensional. Plotting skipped (Rules generated).")

        # 2. Iterate over stored partitions to draw
        for block in self.partitions:
            if do_plot:
                color = cmap((block['id'] - 1) % 20)
                center = (block['bounds_min'] + block['bounds_max']) / 2

                if n_dims == 2:
                    w = block['bounds_max'][0] - block['bounds_min'][0]
                    h = block['bounds_max'][1] - block['bounds_min'][1]
                    alpha = 0.3 if color_regions else 0.0
                    edge = 'black'
                    rect = Rectangle((block['bounds_min'][0], block['bounds_min'][1]), w, h,
                                     linewidth=1, edgecolor=edge, facecolor=color, alpha=alpha, linestyle='--')
                    ax.add_patch(rect)
                    ax.text(center[0], center[1], str(block['id']), ha='center', fontsize=8, weight='bold')

                elif n_dims == 3:
                    self._draw_cuboid_3d(ax, block['bounds_min'], block['bounds_max'], color if color_regions else (0.7, 0.7, 0.7, 0.0001))
                    ax.text(center[0], center[1], center[2], str(block['id']), fontsize=9, weight='bold')

                # C. Dominating Point
                if plot_dominating and len(block['points']) > 0:
                    dom_pt = block['dominating_point']

                    if n_dims == 2:
                        ax.scatter(dom_pt[0], dom_pt[1], c='red', marker='s', s=40, zorder=10)
                    elif n_dims == 3:
                        ax.scatter(dom_pt[0], dom_pt[1], dom_pt[2], c='red', marker='s', s=50, zorder=10)

        # 3. Finalize Plot
        if do_plot:
            # Plot all points as background
            if n_dims == 2:
                ax.scatter(self.points[:,0], self.points[:,1], c='k', s=10, alpha=0.3, zorder=1, label='Solutions')
            elif n_dims == 3:
                ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], c='k', s=10, alpha=0.3, zorder=1, label='Solutions')
            plt.legend()
            
            # Save plot in SVG format if requested
            if save:    
                plt.savefig(filename, format='svg')
                print(f"Plot saved as {filename}")
            
            plt.show()


        # 4. Output Rules
        print(f"--- Partition Rules (Dimensions: {n_dims}) ---")
        for block in self.partitions[:5]: # Print first 5
             print(f"Block {block['id']}: {block['rule']}")
        if len(self.partitions) > 5: print("...")

    def _draw_cuboid_3d(self, ax, min_b, max_b, color):
        """Helper to draw a 3D box."""
        x = [min_b[0], max_b[0]]
        y = [min_b[1], max_b[1]]
        z = [min_b[2], max_b[2]]
        verts = [
            [(x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[1], z[0]), (x[0], y[1], z[0])], # Bottom
            [(x[0], y[0], z[1]), (x[1], y[0], z[1]), (x[1], y[1], z[1]), (x[0], y[1], z[1])], # Top
            [(x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[0], z[1]), (x[0], y[0], z[1])], # Front
            [(x[0], y[1], z[0]), (x[1], y[1], z[0]), (x[1], y[1], z[1]), (x[0], y[1], z[1])], # Back
            [(x[0], y[0], z[0]), (x[0], y[1], z[0]), (x[0], y[1], z[1]), (x[0], y[0], z[1])], # Left
            [(x[1], y[0], z[0]), (x[1], y[1], z[0]), (x[1], y[1], z[1]), (x[1], y[0], z[1])]  # Right
        ]
        poly = Poly3DCollection(verts, facecolors=color, alpha=0.15, edgecolor='k', linewidths=0.5, linestyles='--')
        ax.add_collection3d(poly)