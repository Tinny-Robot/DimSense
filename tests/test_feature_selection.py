import unittest
from dimsense.feature_selection import FeatureSelector
from sklearn.datasets import load_iris

class TestFeatureSelector(unittest.TestCase):
    def test_select_k_best(self):
        # Load iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Initialize FeatureSelector with select_k_best
        selector = FeatureSelector(method='select_k_best', num_features=2)
        
        # Test fit_transform
        X_selected = selector.fit_transform(X, y)
        
        self.assertEqual(X_selected.shape[1], 2)
        
    def test_rfe(self):
        # Load iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Initialize FeatureSelector with rfe
        selector = FeatureSelector(method='rfe', num_features=2)
        
        # Test fit_transform
        X_selected = selector.fit_transform(X, y)
        
        self.assertEqual(X_selected.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
