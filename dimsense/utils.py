"""
Utility Functions for DimSense
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


def normalize_data(X):
    """
    Normalize the input data.

    Parameters:
    - X (array-like): Input data.

    Returns:
    - X_normalized (array-like): Normalized data.
    """
    X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    return X_normalized

def standardize_data(X):
    """
    Standardize the input data.

    Parameters:
    - X (array-like): Input data.

    Returns:
    - X_standardized (array-like): Standardized data.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std
    return X_standardized

def scatter_plot(X, y=None, title="Scatter Plot"):
    """
    Create a scatter plot for visualization.

    Parameters:
    - X (array-like): Input data.
    - y (array-like): Target labels (optional).
    - title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    if y is None:
        plt.scatter(X[:, 0], X[:, 1], marker='o')
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def detect_outliers(X):
    """
    Detect outliers using Local Outlier Factor.

    Parameters:
    - X (array-like): Input data.

    Returns:
    - outliers (array-like): Boolean array indicating outliers.
    """
    lof = LocalOutlierFactor()
    outliers = lof.fit_predict(X)
    return outliers == -1

def validate_data(X, y=None):
    """
    Validate data integrity and dimensions.

    Parameters:
    - X (array-like): Input data.
    - y (array-like): Target labels (optional).
    """
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise ValueError("Input data should be a NumPy array or pandas DataFrame.")
    
    if y is not None and len(X) != len(y):
        raise ValueError("Number of samples in X and y must match.")
    
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError("Column names in X should be unique.")
        
        if X.duplicated().any():
            raise ValueError("Duplicate rows found in X.")
        
        if X.isnull().any().any():
            raise ValueError("Missing values found in X.")
        
    if isinstance(y, pd.Series):
        if not y.name:
            raise ValueError("Target labels (y) should have a valid name.")
        
        if y.duplicated().any():
            raise ValueError("Duplicate values found in y.")
        
        if y.isnull().any():
            raise ValueError("Missing values found in y.")