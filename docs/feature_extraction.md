# Feature Extraction Methods

The "DimSense" library provides various feature extraction methods to transform your data into a lower-dimensional representation. These methods can be useful for reducing the complexity of your data while retaining essential information.

## PCAExtractor

The `PCAExtractor` class performs Principal Component Analysis (PCA) for feature extraction.

### Parameters

- `num_components` (int, default=2): Number of components to extract.

### Methods

- `fit_transform(X)`: Fit the PCA model and transform the data.
- `set_num_components(num_components)`: Set the number of components to extract.

## TSNEExtractor

The `TSNEExtractor` class performs t-Distributed Stochastic Neighbor Embedding (t-SNE) for feature extraction.

### Parameters

- `num_components` (int, default=2): Number of components to extract.
- `perplexity` (float, default=30): Perplexity parameter for t-SNE.

### Methods

- `fit_transform(X)`: Fit the t-SNE model and transform the data.
- `set_num_components(num_components)`: Set the number of components to extract.
- `set_perplexity(perplexity)`: Set the perplexity parameter for t-SNE.

## TFIDFExtractor

The `TFIDFExtractor` class performs feature extraction using Term Frequency-Inverse Document Frequency (TF-IDF).

### Parameters

- `max_features` (int, default=1000): Maximum number of features.

### Methods

- `fit_transform(X)`: Fit the TF-IDF vectorizer and transform the data.
- `set_max_features(max_features)`: Set the maximum number of features.

## CountVectorizerExtractor

The `CountVectorizerExtractor` class performs feature extraction using word counts.

### Parameters

- `max_features` (int, default=1000): Maximum number of features.

### Methods

- `fit_transform(X)`: Fit the CountVectorizer and transform the data.
- `set_max_features(max_features)`: Set the maximum number of features.

## LatentDirichletAllocationExtractor

The `LatentDirichletAllocationExtractor` class performs topic modeling feature extraction using Latent Dirichlet Allocation (LDA).

### Parameters

- `num_topics` (int, default=10): Number of topics for LDA.

### Methods

- `fit_transform(X)`: Fit the LDA model and transform the data.
- `set_num_topics(num_topics)`: Set the number of topics for LDA.

## FastICAExtractor

The `FastICAExtractor` class performs feature extraction using FastICA.

### Parameters

- `num_components` (int, default=2): Number of components to extract.

### Methods

- `fit_transform(X)`: Fit the FastICA model and transform the data.
- `set_num_components(num_components)`: Set the number of components to extract.

## AutoencoderExtractor

The `AutoencoderExtractor` class performs feature extraction using autoencoders.

### Parameters

- `encoding_dim` (int, default=2): Dimension of the encoded representation.

### Methods

- `fit_transform(X)`: Fit the autoencoder model and transform the data.
- `set_encoding_dim(encoding_dim)`: Set the dimension of the encoded representation.
