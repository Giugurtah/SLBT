import numpy as np

from .base import HomogeneityStrategy
from ..split import _splitS

# Homogeneity strategy for B homogeneity.
class HomogeneityB(HomogeneityStrategy):
    # Get threshold values for B homogeneity
    def get_treshold_values(self, best_threshold, x_vals, x_s):
        thresholds = []
        bt = np.asarray(best_threshold)
        strat_vals = np.unique(x_s)


        for t in range(len(strat_vals)):
            thresholds.append(x_vals[bt[t] > 0])

        return thresholds

    # Split function for B homogeneity
    def split(self, X_best, x_s, thresholds):
        # Compute thresholds and split indices
        indexL, indexR = _splitS(X_best, x_s, thresholds)

        # Returns the indices and thresholds
        return (indexL, indexR)

    # Compute lift for B homogeneity
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
