# Feature Selection Methods

The "DimSense" library provides various feature selection methods to help you choose the most relevant features from your data. These methods can improve the efficiency and performance of your machine learning models by reducing the dimensionality of your feature space.

## FeatureSelector

The `FeatureSelector` class offers several feature selection methods.

### Parameters

- `method` (str, default='select_k_best'): Feature selection method to use ('select_k_best', 'mutual_info', 'chi2', 'rfe', 'variance', 'random_forest', 'lasso').
- `num_features` (int, default=10): Number of features to select.

### Methods

- `fit_transform(X, y)`: Fit the selected feature selection method and transform the data.
- `set_method(method)`: Set the feature selection method.
- `set_num_features(num_features)`: Set the number of features to select.

#### Example

```python
from dimsense.feature_selection import FeatureSelector
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize the FeatureSelector
selector = FeatureSelector(method='select_k_best', num_features=2)

# Fit and transform the data
X_selected = selector.fit_transform(X, y)

print("Selected features shape:", X_selected.shape)
```

## Usage Tips

- Choose the appropriate feature selection method based on your data type and problem.
- Experiment with different numbers of selected features to find the optimal configuration.

## Next Steps

The "DimSense" library offers a variety of feature extraction methods as well. For more details on each method and their usage, refer to the [Feature Extraction Methods](feature_extraction.md) documentation.

Keep in mind that the right feature selection approach depends on your specific use case and data characteristics. Use these methods to enhance your machine learning workflows and improve model performance.