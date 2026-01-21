# slbt/_preprocessing/_backend/ctypes_interface.py

"""
ctypes interface to the categorizer C library.

This module provides Python wrappers for the KMeans clustering functions
implemented in C for performance.
"""

import ctypes
from pathlib import Path
import platform
import numpy as np

# Type aliases
c_int = ctypes.c_int
c_double = ctypes.c_double

# ============================================================================
# LIBRARY LOADING
# ============================================================================

def _get_library_path():
    """Determine the correct library file based on OS."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        lib_name = "libcategorizer.dylib"
    elif system == "Linux":
        lib_name = "libcategorizer.so"
    elif system == "Windows":
        lib_name = "libcategorizer.dll"
    else:
        raise OSError(f"Unsupported operating system: {system}")
    
    lib_path = Path(__file__).parent / lib_name
    
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Categorizer library not found at {lib_path}. "
            f"Please run 'make install' in csrc/categorizer/"
        )
    
    return str(lib_path)

# Load the library
_lib = ctypes.CDLL(_get_library_path())

# ============================================================================
# FUNCTION SIGNATURES
# ============================================================================

# categorize_kmeans(int I, int K, double X[], int labels[])
_lib.categorize_kmeans.argtypes = [
    c_int,  # I
    c_int,  # K
    ctypes.POINTER(c_double),  # X
    ctypes.POINTER(c_int),     # labels
]
_lib.categorize_kmeans.restype = None

# categorize_kmeans_elbow(int I, int Kmax, int Kmin, int minN, double X[], int labels[])
_lib.categorize_kmeans_elbow.argtypes = [
    c_int,  # I
    c_int,  # Kmax
    c_int,  # Kmin
    c_int,  # minN
    ctypes.POINTER(c_double),  # X
    ctypes.POINTER(c_int),     # labels
]
_lib.categorize_kmeans_elbow.restype = c_int  # returns optimal K

# categorize_kmeans_silhouette(int I, int Kmax, int Kmin, int minN, double X[], int labels[])
_lib.categorize_kmeans_silhouette.argtypes = [
    c_int,  # I
    c_int,  # Kmax
    c_int,  # Kmin
    c_int,  # minN
    ctypes.POINTER(c_double),  # X
    ctypes.POINTER(c_int),     # labels
]
_lib.categorize_kmeans_silhouette.restype = c_int  # returns optimal K

# get_cluster_centers(int I, int K, double X[], int labels[], double centers[])
_lib.get_cluster_centers.argtypes = [
    c_int,  # I
    c_int,  # K
    ctypes.POINTER(c_double),  # X
    ctypes.POINTER(c_int),     # labels
    ctypes.POINTER(c_double),  # centers
]
_lib.get_cluster_centers.restype = None

# get_cluster_sizes(int I, int K, int labels[], int sizes[])
_lib.get_cluster_sizes.argtypes = [
    c_int,  # I
    c_int,  # K
    ctypes.POINTER(c_int),  # labels
    ctypes.POINTER(c_int),  # sizes
]
_lib.get_cluster_sizes.restype = None

# ============================================================================
# PYTHON WRAPPERS
# ============================================================================

def categorize_fixed_k(X: np.ndarray, k: int):
    """
    Categorize data using KMeans with a fixed number of clusters.
    
    Parameters
    ----------
    X : np.ndarray
        1D array of numeric values to cluster (must be sorted)
    k : int
        Number of clusters
    
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample (integers from 0 to k-1)
    centers : np.ndarray
        Cluster centers
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    I = len(X)
    
    # Allocate output arrays
    labels = np.zeros(I, dtype=np.int32)
    
    # Call C function
    _lib.categorize_kmeans(
        I,
        k,
        X.ctypes.data_as(ctypes.POINTER(c_double)),
        labels.ctypes.data_as(ctypes.POINTER(c_int)),
    )
    
    # Get cluster centers
    centers = get_centers(X, labels, k)
    
    return labels, centers


def categorize_elbow(X: np.ndarray, k_max: int = 10, k_min: int = 2, 
                     min_size: int = 1):
    """
    Categorize data using KMeans with automatic K selection via elbow method.
    
    Parameters
    ----------
    X : np.ndarray
        1D array of numeric values to cluster (must be sorted)
    k_max : int, default=10
        Maximum number of clusters to test
    k_min : int, default=2
        Minimum number of clusters to consider optimal
    min_size : int, default=1
        Minimum size required for each cluster
    
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample
    centers : np.ndarray
        Cluster centers
    k_optimal : int
        Optimal number of clusters found
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    I = len(X)
    
    # Allocate output arrays
    labels = np.zeros(I, dtype=np.int32)
    
    # Call C function
    k_optimal = _lib.categorize_kmeans_elbow(
        I,
        k_max,
        k_min,
        min_size,
        X.ctypes.data_as(ctypes.POINTER(c_double)),
        labels.ctypes.data_as(ctypes.POINTER(c_int)),
    )
    
    # Get cluster centers
    centers = get_centers(X, labels, k_optimal)
    
    return labels, centers, k_optimal


def categorize_silhouette(X: np.ndarray, k_max: int = 10, k_min: int = 2,
                          min_size: int = 1):
    """
    Categorize data using KMeans with automatic K selection via silhouette method.
    
    Parameters
    ----------
    X : np.ndarray
        1D array of numeric values to cluster (must be sorted)
    k_max : int, default=10
        Maximum number of clusters to test
    k_min : int, default=2
        Minimum number of clusters to consider optimal
    min_size : int, default=1
        Minimum size required for each cluster
    
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample
    centers : np.ndarray
        Cluster centers
    k_optimal : int
        Optimal number of clusters found
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    I = len(X)
    
    # Allocate output arrays
    labels = np.zeros(I, dtype=np.int32)
    
    # Call C function
    k_optimal = _lib.categorize_kmeans_silhouette(
        I,
        k_max,
        k_min,
        min_size,
        X.ctypes.data_as(ctypes.POINTER(c_double)),
        labels.ctypes.data_as(ctypes.POINTER(c_int)),
    )
    
    # Get cluster centers
    centers = get_centers(X, labels, k_optimal)
    
    return labels, centers, k_optimal


def get_centers(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Calculate cluster centers for a given clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Original data
    labels : np.ndarray
        Cluster labels
    k : int
        Number of clusters
    
    Returns
    -------
    centers : np.ndarray
        Array of cluster centers (k elements)
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    I = len(X)
    
    centers = np.zeros(k, dtype=np.float64)
    
    _lib.get_cluster_centers(
        I,
        k,
        X.ctypes.data_as(ctypes.POINTER(c_double)),
        labels.ctypes.data_as(ctypes.POINTER(c_int)),
        centers.ctypes.data_as(ctypes.POINTER(c_double)),
    )
    
    return centers


def get_sizes(labels: np.ndarray, k: int) -> np.ndarray:
    """
    Calculate cluster sizes for a given clustering.
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    k : int
        Number of clusters
    
    Returns
    -------
    sizes : np.ndarray
        Array of cluster sizes (k elements)
    """
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    I = len(labels)
    
    sizes = np.zeros(k, dtype=np.int32)
    
    _lib.get_cluster_sizes(
        I,
        k,
        labels.ctypes.data_as(ctypes.POINTER(c_int)),
        sizes.ctypes.data_as(ctypes.POINTER(c_int)),
    )
    
    return sizes