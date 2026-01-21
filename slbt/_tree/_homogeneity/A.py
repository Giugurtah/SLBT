import numpy as np

from .base import HomogeneityStrategy
from ..split import _split

# Homogeneity strategy for A homogeneity.
class HomogeneityA(HomogeneityStrategy):
    # Get threshold values for A homogeneity
    def get_treshold_values(self, best_threshold, x_vals, x_s):
        bt = np.asarray(best_threshold)

        if bt.ndim == 2:
            bt = bt[0]   # prendo la soglia unica

        return x_vals[bt > 0]

    # Split function for A homogeneity
    def split(self, X_best, x_s, threshold):
        # Compute thresholds and split indices
        indexL, indexR = _split(X_best, threshold)

        # Returns the indices and thresholds
        return (indexL, indexR)

    # Compute lift for A homogeneity
    def compute_lift(self, beta, distribution):
        lift1, lift2 = [], []
        for t in range(len(beta)):
            lift1.append([b[0] for b in beta[t]] / distribution)
            lift2.append([b[1] for b in beta[t]] / distribution)
        return lift1, lift2

    def make_node_kwargs(self, **kwargs):
        return kwargs
