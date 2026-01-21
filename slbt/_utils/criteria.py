import pandas as pd
import numpy as np

from slbt._backend.ctypes_interface import gpi
from slbt._utils.utils import _stratified_contingency

#*This function evaluates the gpi for all avaialable features and returns them sorted based on their gpi
def _gpi(X, y, x_s):
    gpi_vals = []
    gpi_index = []

    for x in X:
        # Build frequency table
        Fs = _stratified_contingency(X[x], y, x_s, norm=False)
        
        K, I, J = Fs.shape #Dimensions of F

        # Preparing data for C call
        Fs_flat = Fs.ravel().astype(np.double)

        # Call in C
        gpi_val = gpi(K, I, J, Fs_flat)
        print(x, " GPI: ", gpi_val) #TODO: DA CANCELLARE

        gpi_vals.append(gpi_val)
        gpi_index.append(x)

    #Ordering gpi values
    gpi_vals, gpi_index = zip(
        *sorted(zip(gpi_vals, gpi_index), reverse=True)
    )
    return gpi_vals, gpi_index


#*Function to calculate the gini impurity of a label array
def _impurity(y):
    dist = np.unique(y, return_counts=True)[1]/len(y)
    
    impurity = 1
    for x in dist:
        impurity -= x*x

    return impurity

#*This function given X and y returns the number of samples, number of features, number of labels, impurity and distribution of y
def _get_sizes(X, y):
    n_samples, n_feats = X.shape
    n_labels = len(np.unique(y))
    impurity = _impurity(y)
    distribution =  np.unique(y, return_counts=True)[1]/len(y)
    return n_samples, n_feats, n_labels, impurity, distribution
