# slbt/_preprocessing/categorizer.py

"""
Categorizer class for automatic discretization of continuous variables.

This module provides a scikit-learn style interface for converting
continuous numeric variables into categorical ones using KMeans clustering.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict
import warnings

from ._backend import (
    categorize_fixed_k,
    categorize_elbow,
    categorize_silhouette,
)


class Categorizer:
    """
    Automatic categorization of continuous variables using KMeans clustering.
    
    This class provides a scikit-learn style interface for discretizing
    continuous variables. It supports three methods for determining the
    optimal number of bins:
    
    - 'fixed': Use a fixed number of bins (specify k)
    - 'elbow': Automatically determine k using the elbow method
    - 'silhouette': Automatically determine k using silhouette score
    
    Parameters
    ----------
    method : {'fixed', 'elbow', 'silhouette'}, default='elbow'
        Method for determining the number of bins
    k : int, optional
        Number of bins (required if method='fixed')
    k_max : int, default=10
        Maximum number of bins to test (for 'elbow' and 'silhouette')
    k_min : int, default=2
        Minimum number of bins to consider (for 'elbow' and 'silhouette')
    min_size : int, default=1
        Minimum size required for each bin
    labels : list of str, optional
        Custom labels for the bins (e.g., ['low', 'medium', 'high'])
        If None, uses integer labels (0, 1, 2, ...)
    
    Attributes
    ----------
    bins_ : dict
        Dictionary mapping column names to bin edges
    labels_ : dict
        Dictionary mapping column names to bin labels
    k_ : dict
        Dictionary mapping column names to number of bins found
    centers_ : dict
        Dictionary mapping column names to cluster centers
    
    Examples
    --------
    >>> import pandas as pd
    >>> from slbt import Categorizer
    >>> 
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'age': [18, 22, 25, 45, 50, 55, 70, 75, 80],
    ...     'income': [20000, 25000, 30000, 60000, 65000, 70000, 100000, 105000, 110000]
    ... })
    >>> 
    >>> # Automatic categorization with elbow method
    >>> cat = Categorizer(method='elbow', k_max=5)
    >>> df_cat = cat.fit_transform(df)
    >>> 
    >>> # With custom labels
    >>> cat = Categorizer(method='fixed', k=3, labels=['young', 'adult', 'senior'])
    >>> df_cat = cat.fit_transform(df[['age']])
    """
    
    def __init__(
        self,
        method: str = 'elbow',
        k: Optional[int] = None,
        k_max: int = 10,
        k_min: int = 2,
        min_size: int = 1,
        labels: Optional[List[str]] = None,
    ):
        # Validate method
        if method not in ['fixed', 'elbow', 'silhouette']:
            raise ValueError(
                f"method must be 'fixed', 'elbow', or 'silhouette', got '{method}'"
            )
        
        # Validate k for fixed method
        if method == 'fixed' and k is None:
            raise ValueError("k must be specified when method='fixed'")
        
        if method == 'fixed' and k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        
        # Store parameters
        self.method = method
        self.k = k
        self.k_max = k_max
        self.k_min = k_min
        self.min_size = min_size
        self.labels = labels
        
        # Attributes set during fit
        self.bins_ = {}
        self.labels_ = {}
        self.k_ = {}
        self.centers_ = {}
        self.columns_ = None
    
    def fit(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> 'Categorizer':
        """
        Fit the categorizer on the data.
        
        This learns the bin edges for each column in X.
        
        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            Continuous data to categorize
        
        Returns
        -------
        self : Categorizer
            Fitted categorizer
        """
        # Convert input to DataFrame
        X_df = self._validate_input(X)
        
        # Store column names
        self.columns_ = X_df.columns.tolist()
        
        # Fit each column
        for col in self.columns_:
            self._fit_column(col, X_df[col].values)
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Transform continuous data to categorical.
        
        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            Continuous data to categorize
        
        Returns
        -------
        X_cat : DataFrame
            Categorized data
        """
        # Check if fitted
        if self.columns_ is None:
            raise RuntimeError("Categorizer must be fitted before transform")
        
        # Convert input to DataFrame
        X_df = self._validate_input(X)
        
        # Check columns match
        if set(X_df.columns) != set(self.columns_):
            raise ValueError(
                f"Columns in X don't match fitted columns. "
                f"Expected {self.columns_}, got {X_df.columns.tolist()}"
            )
        
        # Transform each column
        X_cat = pd.DataFrame(index=X_df.index)
        
        for col in self.columns_:
            X_cat[col] = self._transform_column(col, X_df[col].values)
        
        return X_cat
    
    def fit_transform(
        self, 
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            Continuous data to categorize
        
        Returns
        -------
        X_cat : DataFrame
            Categorized data
        """
        return self.fit(X).transform(X)
    
    def get_bin_info(self, column: Optional[str] = None) -> Dict:
        """
        Get information about the bins for a column.
        
        Parameters
        ----------
        column : str, optional
            Column name. If None, returns info for all columns.
        
        Returns
        -------
        info : dict
            Dictionary with bin information
        """
        if self.columns_ is None:
            raise RuntimeError("Categorizer must be fitted first")
        
        if column is not None:
            if column not in self.columns_:
                raise ValueError(f"Column '{column}' not found")
            
            return {
                'k': self.k_[column],
                'centers': self.centers_[column],
                'bins': self.bins_[column],
                'labels': self.labels_[column],
            }
        else:
            # Return info for all columns
            return {
                col: {
                    'k': self.k_[col],
                    'centers': self.centers_[col],
                    'bins': self.bins_[col],
                    'labels': self.labels_[col],
                }
                for col in self.columns_
            }
    
    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    
    def _validate_input(
        self, 
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """Convert input to pandas DataFrame and validate."""
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X_df = pd.DataFrame({'X': X})
            else:
                X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        elif isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            raise TypeError(
                f"X must be DataFrame, Series, or ndarray, got {type(X)}"
            )
        
        # Check for non-numeric columns
        non_numeric = X_df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise ValueError(
                f"All columns must be numeric. Found non-numeric: {non_numeric.tolist()}"
            )
        
        return X_df
    
    def _fit_column(self, column: str, X: np.ndarray) -> None:
        """Fit categorizer for a single column."""
        # Remove NaN values
        X_clean = X[~np.isnan(X)]
        
        if len(X_clean) == 0:
            warnings.warn(f"Column '{column}' contains only NaN values")
            self.k_[column] = 0
            self.centers_[column] = np.array([])
            self.bins_[column] = np.array([])
            self.labels_[column] = []
            return
        
        # Sort for clustering
        X_sorted = np.sort(X_clean)
        
        # Call appropriate backend function
        if self.method == 'fixed':
            cluster_labels, centers = categorize_fixed_k(X_sorted, self.k)
            k_opt = self.k
        elif self.method == 'elbow':
            cluster_labels, centers, k_opt = categorize_elbow(
                X_sorted, self.k_max, self.k_min, self.min_size
            )
        else:  # silhouette
            cluster_labels, centers, k_opt = categorize_silhouette(
                X_sorted, self.k_max, self.k_min, self.min_size
            )
        
        # Store results
        self.k_[column] = k_opt
        self.centers_[column] = centers
        
        # Calculate bin edges (boundaries between clusters)
        bin_edges = self._calculate_bin_edges(X_sorted, cluster_labels, centers)
        self.bins_[column] = bin_edges
        
        # Create labels
        self.labels_[column] = self._create_labels(k_opt)
    
    def _transform_column(self, column: str, X: np.ndarray) -> pd.Series:
        """Transform a single column."""
        if self.k_[column] == 0:
            # Column was all NaN during fit
            return pd.Series([np.nan] * len(X))
        
        bins = self.bins_[column]
        labels = self.labels_[column]
        
        # Handle edge case: single bin
        if len(bins) <= 2:
            # All values go to the same category
            return pd.Series([labels[0]] * len(X))
        
        # pd.cut expects n bins to have n-1 labels
        # But we have k clusters, so k labels
        # bins array has k+1 edges (including min and max)
        n_bins = len(bins) - 1  # Number of intervals
        
        # Ensure we have the right number of labels
        cut_labels = labels[:n_bins] if len(labels) >= n_bins else labels
        
        try:
            result = pd.cut(
                X, 
                bins=bins, 
                labels=cut_labels,
                include_lowest=True,
                duplicates='drop'  # Drop duplicate bin edges
            )
        except ValueError as e:
            # If still failing, fall back to using cluster centers
            warnings.warn(
                f"Could not use pd.cut for column '{column}': {e}. "
                f"Falling back to nearest center assignment."
            )
            result = self._assign_to_nearest_center(X, column)
        
        return result

    def _assign_to_nearest_center(self, X: np.ndarray, column: str) -> pd.Series:
        """Assign values to nearest cluster center (fallback method)."""
        centers = self.centers_[column]
        labels = self.labels_[column]
        
        # For each value, find nearest center
        assignments = []
        for x in X:
            if np.isnan(x):
                assignments.append(np.nan)
            else:
                distances = np.abs(centers - x)
                nearest_idx = np.argmin(distances)
                assignments.append(labels[nearest_idx])
    
        return pd.Series(assignments)

    def _calculate_bin_edges(self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Calculate bin edges from cluster assignments."""
        k = len(centers)
        
        if k == 1:
            # Single cluster: use min and max
            return np.array([X.min(), X.max()])
        
        # Fix: Ensure labels are in valid range [0, k-1]
        unique_labels = np.unique(labels)
        if len(unique_labels) > k:
            warnings.warn(
                f"Got {len(unique_labels)} unique labels but expected {k} clusters. "
                f"Reassigning labels based on centers."
            )
            # Reassign labels based on nearest center
            labels = np.array([np.argmin(np.abs(centers - x)) for x in X])
            unique_labels = np.unique(labels)
        
        # Sort centers and create mapping
        sorted_indices = np.argsort(centers)
        sorted_centers = centers[sorted_indices]
        
        # Create mapping from old labels to sorted labels
        # Handle case where not all labels 0..k-1 are present
        label_map = {}
        for new_idx, old_idx in enumerate(sorted_indices):
            # Find which original label corresponds to this center
            if old_idx < len(unique_labels):
                old_label = unique_labels[old_idx] if old_idx < len(unique_labels) else old_idx
            else:
                old_label = old_idx
            label_map[old_label] = new_idx
        
        # Additional safety: map any missing labels
        for ul in unique_labels:
            if ul not in label_map:
                # Find nearest center
                if ul < k:
                    label_map[ul] = np.argmin(np.abs(sorted_centers - centers[ul]))
                else:
                    label_map[ul] = 0  # Default to first cluster
        
        # Map labels to sorted order
        try:
            sorted_labels = np.array([label_map.get(int(l), 0) for l in labels])
        except Exception as e:
            warnings.warn(f"Error mapping labels: {e}. Using simple assignment.")
            # Fallback: just use the labels as-is
            sorted_labels = labels % k  # Ensure in range [0, k-1]
        
        # Bin edges: [min, boundary1, boundary2, ..., max]
        edges = [X.min()]
        
        for i in range(k - 1):
            # Find boundary between cluster i and i+1
            cluster_i_points = X[sorted_labels == i]
            cluster_i_plus_1_points = X[sorted_labels == i + 1]
            
            if len(cluster_i_points) > 0 and len(cluster_i_plus_1_points) > 0:
                boundary = (cluster_i_points.max() + cluster_i_plus_1_points.min()) / 2
                edges.append(boundary)
            elif i + 1 < len(sorted_centers):
                # Fallback: use midpoint between centers
                boundary = (sorted_centers[i] + sorted_centers[i + 1]) / 2
                edges.append(boundary)
        
        edges.append(X.max())
        
        # Ensure edges are unique and sorted
        edges = np.unique(edges)
        
        return edges
    
    def _create_labels(self, k: int) -> List:
        """Create labels for bins."""
        if self.labels is not None:
            if len(self.labels) != k:
                warnings.warn(
                    f"Number of custom labels ({len(self.labels)}) doesn't match "
                    f"number of bins ({k}). Using integer labels instead."
                )
                return list(range(k))
            return self.labels
        else:
            return list(range(k))
    
    def __repr__(self) -> str:
        if self.columns_ is None:
            return f"Categorizer(method='{self.method}', not fitted)"
        else:
            k_str = ', '.join([f"{col}={self.k_[col]}" for col in self.columns_])
            return f"Categorizer(method='{self.method}', fitted on {len(self.columns_)} columns, k={{{k_str}}})"