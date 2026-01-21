# slbt/_preprocessing/_backend/__init__.py

"""
Backend interface for the Categorizer module.
"""

from .ctypes_interface import (
    categorize_fixed_k,
    categorize_elbow,
    categorize_silhouette,
    get_centers,
    get_sizes,
)

__all__ = [
    'categorize_fixed_k',
    'categorize_elbow',
    'categorize_silhouette',
    'get_centers',
    'get_sizes',
]
