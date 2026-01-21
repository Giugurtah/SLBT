import numpy as np

from .base import HomogeneityStrategy
from ..split import _splitS

# Homogeneity strategy for None homogeneity.
class HomogeneityNone(HomogeneityStrategy):
    # Get threshold values for None homogeneity
    def get_treshold_values(self, best_threshold, x_vals, x_s):
        thresholds = []
        bt = np.asarray(best_threshold)
        strat_vals = np.unique(x_s)


        for t in range(len(strat_vals)):
            thresholds.append(x_vals[bt[t] > 0])

        return thresholds

    # Split function for None homogeneity
    def split(self, X_best, x_s, thresholds):
        # Compute thresholds and split indices
        indexL, indexR = _splitS(X_best, x_s, thresholds)

        # Returns the indices and thresholds
        return (indexL, indexR)

    # Compute lift for None homogeneity
    def compute_lift(self, beta, distribution):
        lift1, lift2 = [], []
        for t in range(len(beta)):
            lift1.append([b[0] for b in beta[t]] / distribution)
            lift2.append([b[1] for b in beta[t]] / distribution)
        return lift1, lift2

    def make_node_kwargs(self, **kwargs):
        return kwargs




