import numpy as np

class Node:
    """
    Single node in decision tree.
    """
    def __init__(self, feature_idx, thresh, left, right, value):
        self.feature_idx = feature_idx
        self.thresh = thresh
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    
    def _gini(self, y):
        """
        Calculate the Gini impurity of a dataset.
        
        Parameters:
        y : np.ndarray
            Target vector.
        
        Returns:
        float
            Gini impurity of the dataset.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum([p ** 2 for p in ps if p > 0])
    
    def _information_gain(self, y, left_y, right_y):
        """
        Calculate the information gain of a potential split.
        
        Parameters:
        y : np.ndarray
            Target vector.
        left_y : np.ndarray
            Left split target vector.
        right_y : np.ndarray
            Right split target vector.
        
        Returns:
        float
            Information gain of the split.
        """
        p = len(left_y) / len(y)
        return self._gini(y) - p * self._gini(left_y) - (1 - p) * self._gini(right_y)