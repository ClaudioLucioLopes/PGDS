# --- 1. KDNode (Storage Unit) ---
class KDNode:
    """
    Represents a block (Hyperrectangle) in the objective space partition.
    Stores the explicit bounds, points, and tree structure.
    """
    def __init__(self, points, bounds_min, bounds_max, depth, axis=None, split_val=None, left=None, right=None):
        self.points = points
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.depth = depth
        self.axis = axis
        self.split_val = split_val
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None and self.right is None