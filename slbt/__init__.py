# slbt/__init__.py

"""
SLBT - Simultaneous Latent Budget Tree

A supervised machine learning algorithm that builds decision trees
using Latent Budget Analysis at each node.
"""

from .slbt import SLBT
from ._preprocessing.categorizer import Categorizer

__all__ = ['SLBT', 'Categorizer']
__version__ = '0.1.0'