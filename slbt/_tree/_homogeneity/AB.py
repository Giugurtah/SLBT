import numpy as np

from .base import HomogeneityStrategy
from ..split import _split

# Homogeneity strategy for AB homogeneity.
class HomogeneityAB(HomogeneityStrategy):
    # Get threshold values for AB homogeneity
    def get_treshold_values(self, best_threshold, x_vals, x_s):
        bt = np.asarray(best_threshold)

        if bt.ndim == 2:
            bt = bt[0]   # prendo la soglia unica

        return x_vals[bt > 0]
        
    # Split function for AB homogeneity
    def split(self, X_best, x_s, threshold):
        # Compute thresholds and split indices
        indexL, indexR = _split(X_best, threshold)

        # Returns the indices and thresholds
        return (indexL, indexR)

    # Compute lift for AB homogeneity
    def compute_lift(self, beta, distribution):
        # Compute lift for each stratum
        bt = np.asarray(beta)

        if bt.ndim == 3:
            bt = bt[0]   

        beta_left  = bt[:, 0]  
        beta_right = bt[:, 1]   

        lift1 = beta_left / distribution
        lift2 = beta_right / distribution

        return lift1, lift2

    def make_node_kwargs(self, **kwargs):
        return kwargs
