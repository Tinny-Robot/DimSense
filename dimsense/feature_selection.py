"""
Feature Selection Methods DimSense
"""

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

class FeatureSelector:
    """
    FeatureSelector provides various feature selection methods.
    """

    def __init__(self, method='select_k_best', num_features=10):
        """
        Initialize the FeatureSelector.

        Parameters:
        - method (str): Feature selection method to use ('select_k_best', 'mutual_info').
        - num_features (int): Number of features to select.
        """
        self.method = method
        self.num_features = num_features

    def fit_transform(self, X, y):
        """
        Fit the feature selection method and transform the data.

        Parameters:
        - X (array-like): Input data.
        - y (array-like): Target labels.

        Returns:
        - X_selected (array-like): Selected features.
        """
        if self.method == 'select_k_best':
            selector = SelectKBest(score_func=f_classif, k=self.num_features)
        elif self.method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=self.num_features)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")

        X_selected = selector.fit_transform(X, y)
        return X_selected

    def set_num_features(self, num_features):
        """
        Set the number of features to select.

        Parameters:
        - num_features (int): Number of features to select.
        """
        self.num_features = num_features

    def set_method(self, method):
        """
        Set the feature selection method.

        Parameters:
        - method (str): Feature selection method to use.
        """
        self.method = method
