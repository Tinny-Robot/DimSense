# Welcome to DimSense

DimSense is a Python library designed to simplify feature selection and extraction tasks for machine learning. With DimSense, you can easily transform and preprocess your data using a variety of feature selection and extraction methods.

## Installation

You can install DimSense using pip:

```
pip install dimsense
```

## Quick Start

Let's dive into a quick example of how to use DimSense for feature extraction:

```python
import numpy as np
from dimsense.feature_extraction import PCAExtractor

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 5)

# Initialize the PCAExtractor
extractor = PCAExtractor(num_components=2)

# Transform the data
X_extracted = extractor.fit_transform(X)

print("Original data shape:", X.shape)
print("Extracted data shape:", X_extracted.shape)
```

## Features

DimSense offers a range of feature selection and extraction methods, including:

- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Term Frequency-Inverse Document Frequency (TF-IDF) extraction
- Word count-based feature extraction
- Topic modeling using Latent Dirichlet Allocation (LDA)
- Independent Component Analysis (ICA)
- Autoencoders for feature extraction

For more details on each method, refer to the [Feature Extraction Methods](feature_extraction.md) documentation.

## Documentation

Explore the detailed documentation for more information on how to use DimSense effectively. Check out the [Usage Guide](usage_guide.md) for step-by-step instructions and examples.

## Contributing

DimSense is an open-source project, and contributions are welcome! If you have ideas for improvements or new feature extraction methods, check out our [Contributing Guidelines](/CONTRIBUTING.md) for more information.

## Version

DimSense v0.1.2
