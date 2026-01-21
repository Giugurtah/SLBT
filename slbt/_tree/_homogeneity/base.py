class HomogeneityStrategy:

    def get_treshold_values(self, best_threshold, x_vals, x_s):
        """
        Returns:
            thresholds_for_node
        """
        raise NotImplementedError

    def split(self, X_best, x_s, best_threshold):
        """
        Returns:
            (indexL, indexR), threshold_for_node
        """
        raise NotImplementedError

    def compute_lift(self, beta, distribution):
        """
        Returns:
            lift1, lift2
        """
        raise NotImplementedError

    def make_node_kwargs(self, **kwargs):
        """
        Returns:
            dict of kwargs to pass to Node(...)
        """
        raise NotImplementedError


def get_homogeneity_strategy(homogeneity):
    if homogeneity == "none":
        from .none import HomogeneityNone
        return HomogeneityNone()
    elif homogeneity == "A":
        from .A import HomogeneityA
        return HomogeneityA()
    elif homogeneity == "B":
        from .B import HomogeneityB
        return HomogeneityB()
    elif homogeneity == "AB":
        from .AB import HomogeneityAB
        return HomogeneityAB()
    else:
        raise ValueError(f"Unknown homogeneity: {homogeneity}")

