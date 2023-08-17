# Usage Guide

Welcome to the Usage Guide for the DimSense library! In this guide, you'll learn how to effectively use the feature selection and extraction methods provided by DimSense to preprocess and transform your data.

## Getting Started

To get started, make sure you have installed DimSense using the following command:

```bash
pip install dimsense
```

Once you've installed DimSense, you can import the necessary modules and classes to start using the library in your Python code.

## Feature Selection

DimSense provides the `FeatureSelector` class for feature selection. Here's a basic example of how to use it:

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

## Feature Extraction

DimSense offers various feature extraction methods. Let's see how to use the `PCAExtractor` and `TSNEExtractor` classes:

```python
from dimsense.feature_extraction import PCAExtractor, TSNEExtractor
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 5)

# Initialize the PCAExtractor
pca_extractor = PCAExtractor(num_components=2)
X_pca = pca_extractor.fit_transform(X)

# Initialize the TSNEExtractor
tsne_extractor = TSNEExtractor(num_components=2)
X_tsne = tsne_extractor.fit_transform(X)

print("PCA extracted data shape:", X_pca.shape)
print("t-SNE extracted data shape:", X_tsne.shape)
```

## Utility Functions

DimSense provides utility functions for data preprocessing and validation. Here's how to use the `normalize_data` and `validate_data` functions:

```python
from dimsense.utils import normalize_data, validate_data
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 5)

# Normalize the data
X_normalized = normalize_data(X)

# Validate the data
validate_data(X)
```

## Next Steps

This guide covers the basics of using DimSense for feature selection and extraction. For more details on each method and advanced usage, refer to the [Feature Extraction Methods](feature_extraction.md) documentation.

Remember to explore the library's documentation for a comprehensive understanding of its features and capabilities. If you have any questions or feedback, feel free to contribute or reach out to the community!

## Happy feature engineering with DimSense!