# slbt/_preprocessing/__init__.py

"""
Preprocessing module for SLBT.

This module provides tools for data preprocessing, including
automatic categorization of continuous variables.
"""

from .categorizer import Categorizer

__all__ = ['Categorizer']