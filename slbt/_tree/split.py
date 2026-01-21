import numpy as np
from slbt._backend.ctypes_interface import gpi, slba

#*------Scoring function---------
HOMOGENITY_MAP = {
    "none": lambda K: (K,K),
    "A": lambda K: (1,K),
    "B": lambda K: (K,1),
    "AB": lambda K: (1,1),
}

def score(Fs_noN, Fs, homogeneity="none"):
    # Prepare inputs
    K, I, J = Fs.shape
    KA, KB = HOMOGENITY_MAP[homogeneity](K)

    FsNoN_flat = Fs_noN.ravel().astype(np.double)
    Fs_flat = Fs.ravel().astype(np.double)

    # Call in C
    return slba(K, KA, KB, I, J, FsNoN_flat, Fs_flat)


#*------Split functions---------
def _split(X_col, best_treshold):
    # normalize to string for safe comparison
    x_vals = X_col.astype(str)

    # set for O(1) lookup
    left_values = set(str(v) for v in best_treshold)

    indexL = x_vals.isin(left_values).to_numpy()
    indexR = ~indexL

    return indexL, indexR

def _splitS(X_col, x_s, best_treshold):
    # normalize to string for safe comparison
    x_vals = X_col.astype(str)
    s_vals = x_s.astype(str)

    strata = np.unique(s_vals)

    # build set of (value, stratum) pairs that go LEFT
    left_pairs = set()
    for t, s in enumerate(strata):
        for v in best_treshold[t]:
            left_pairs.add((str(v), s))

    # vectorized membership test
    pairs = list(zip(x_vals, s_vals))

    indexL = np.array([p in left_pairs for p in pairs])
    indexR = ~indexL

    return indexL, indexR
