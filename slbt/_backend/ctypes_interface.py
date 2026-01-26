# gcc -shared -fPIC -o libslbt.dylib slbt_core.c slbt_als.c
import ctypes 
from pathlib import Path
import numpy as np

c_int = ctypes.c_int
c_double = ctypes.c_double

# Shared library
_lib = ctypes.CDLL(str(Path(__file__).parent.absolute()) + "/libslbt.dylib")

# ============================
# gpi_c signature
# ============================
_lib.gpi_c.argtypes = [
    c_int,   # K
    c_int,   # I
    c_int,   # J
    ctypes.POINTER(c_double),  # Fs(flattened K*I*J)
]
_lib.gpi_c.restype = ctypes.c_double


def gpi(K: int, I: int, J: int, Fs_flat: np.ndarray) -> float:
    return _lib.gpi_c(K, I, J, Fs_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# ============================
# slba_c signature
# ============================
_lib.slba_c.argtypes = [
    c_int,                     # K
    c_int,                     # KA
    c_int,                     # KB
    c_int,                     # I
    c_int,                     # J
    ctypes.POINTER(c_double),      # Fs      (flattened K*I*J)
    ctypes.POINTER(c_double),      # Fs_noN      (flattened K*I*J)
    ctypes.POINTER(c_double),      # out_pi
    ctypes.POINTER(c_double),      # out_S      (K*I)
    ctypes.POINTER(c_double),      # out_alpha  (K*I*2)
    ctypes.POINTER(c_double),      # out_beta   (K*J*2)
]

_lib.slba_c.restype = None

def slba(K: int, KA: int, KB: int, I: int, J: int, FsNoN_flat: np.ndarray, Fs_flat: np.ndarray):
    """
    Python wrapper for slba_none_c
    """

    # --- allocate outputs ---
    out_pi   = np.zeros(1, dtype=np.double)
    out_S     = np.zeros(KA * I, dtype=np.double)
    out_alpha = np.zeros(KA * I * 2, dtype=np.double)
    out_beta  = np.zeros(KB * J * 2, dtype=np.double)

    # --- call C ---
    _lib.slba_c(
        K, KA, KB, I, J,
        
        FsNoN_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Fs_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),

        out_pi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_S.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    # --- reshape outputs ---
    pi   = out_pi[0]
    S     = out_S.reshape(KA, I)
    alpha = out_alpha.reshape(KA, I, 2)
    beta  = out_beta.reshape(KB, J, 2)

    return pi, S, alpha, beta
