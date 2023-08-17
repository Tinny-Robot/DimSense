"""
Feature Extraction Methods for DimSense
"""

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class AutoencoderExtractor:
    """
    AutoencoderExtractor provides feature extraction using autoencoders.
    """
    import numpy as np
    import tensorflow as tf

    def __init__(self, encoding_dim=2):
        """
        Initialize the AutoencoderExtractor.

        Parameters:
        - encoding_dim (int): Dimension of the encoded representation.
        """
        self.encoding_dim = encoding_dim
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_layer = tf.keras.layers.Input(shape=(X.shape[1],))
        encoded = tf.keras.layers.Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = tf.keras.layers.Dense(X.shape[1], activation='sigmoid')(encoded)
        autoencoder = tf.keras.models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder

    def fit_transform(self, X):
        """
        Fit the autoencoder model and transform the data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - X_extracted (array-like): Extracted features.
        """
        self.autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=0)
        encoder = tf.keras.models.Model(inputs=self.autoencoder.input, outputs=self.autoencoder.layers[1].output)
        X_extracted = encoder.predict(X)
        return X_extracted

    def set_encoding_dim(self, encoding_dim):
        """
        Set the dimension of the encoded representation.

        Parameters:
        - encoding_dim (int): Dimension of the encoded representation.
        """
        self.encoding_dim = encoding_dim
        self.autoencoder = self.build_autoencoder()


class PCAExtractor:
    """
    PCAExtractor provides PCA-based feature extraction.
    """

    def __init__(self, num_components=2):
        """
        Initialize the PCAExtractor.

        Parameters:
        - num_components (int): Number of components to extract.
        """
        self.num_components = num_components

    def fit_transform(self, X):
        """
        Fit the PCA model and transform the data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - X_extracted (array-like): Extracted features.
        """
        pca = PCA(n_components=self.num_components)
        X_extracted = pca.fit_transform(X)
        return X_extracted

    def set_num_components(self, num_components):
        """
        Set the number of components to extract.

        Parameters:
        - num_components (int): Number of components to extract.
        """
        self.num_components = num_components

class TSNEExtractor:
    """
    TSNEExtractor provides t-SNE-based feature extraction.
    """

    def __init__(self, num_components=2, perplexity=30):
        """
        Initialize the TSNEExtractor.

        Parameters:
        - num_components (int): Number of components to extract.
        - perplexity (float): Perplexity parameter for t-SNE.
        """
        self.num_components = num_components
        self.perplexity = perplexity

    def fit_transform(self, X):
        """
        Fit the t-SNE model and transform the data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - X_extracted (array-like): Extracted features.
        """
        tsne = TSNE(n_components=self.num_components, perplexity=self.perplexity)
        X_extracted = tsne.fit_transform(X)
        return X_extracted

    def set_num_components(self, num_components):
        """
        Set the number of components to extract.

        Parameters:
        - num_components (int): Number of components to extract.
        """
        self.num_components = num_components

    def set_perplexity(self, perplexity):
        """
        Set the perplexity parameter for t-SNE.

        Parameters:
        - perplexity (float): Perplexity parameter for t-SNE.
        """
        self.perplexity = perplexity

class TFIDFExtractor:
    """
    TFIDFExtractor provides feature extraction using TF-IDF.
    """

    def __init__(self, max_features=1000):
        """
        Initialize the TFIDFExtractor.

        Parameters:
        - max_features (int): Maximum number of features.
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, X):
        """
        Fit the TF-IDF vectorizer and transform the data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - X_extracted (array-like): Extracted features.
        """
        X_extracted = self.vectorizer.fit_transform(X).toarray()
        return X_extracted

    def set_max_features(self, max_features):
        """
        Set the maximum number of features.

        Parameters:
        - max_features (int): Maximum number of features.
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)

class CountVectorizerExtractor:
    """
    CountVectorizerExtractor provides feature extraction using word counts.
    """

    def __init__(self, max_features=1000):
        """
        Initialize the CountVectorizerExtractor.

        Parameters:
        - max_features (int): Maximum number of features.
        """
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=max_features)

    def fit_transform(self, X):
        """
        Fit the CountVectorizer and transform the data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - X_extracted (array-like): Extracted features.
        """
        X_extracted = self.vectorizer.fit_transform(X).toarray()
        return X_extracted

    def set_max_features(self, max_features):
        """
        Set the maximum number of features.

        Parameters:
        - max_features (int): Maximum number of features.
        """
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=max_features)

class LatentDirichletAllocationExtractor:
    """
    LatentDirichletAllocationExtractor provides topic modeling feature extraction using LDA.
    """

    def __init__(self, num_topics=10):
        """
        Initialize the LatentDirichletAllocationExtractor.

        Parameters:
        - num_topics (int): Number of topics for LDA.
        """
        self.num_topics = num_topics
        self.lda = LatentDirichletAllocation(n_components=num_topics)

    def fit_transform(self, X):
        """
        Fit the LDA model and transform the data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - X_extracted (array-like): Extracted features representing topics.
        """
        X_extracted = self.lda.fit_transform(X)
        return X_extracted

    def set_num_topics(self, num_topics):
        """
        Set the number of topics for LDA.

        Parameters:
        - num_topics (int): Number of topics for LDA.
        """
        self.num_topics = num_topics
        self.lda = LatentDirichletAllocation(n_components=num_topics)

class FastICAExtractor:
    """
    FastICAExtractor provides feature extraction using FastICA.
    """

    def __init__(self, num_components=2):
        """
        Initialize the FastICAExtractor.

        Parameters:
        - num_components (int): Number of components to extract.
        """
        self.num_components = num_components
        self.ica = FastICA(n_components=num_components)

    def fit_transform(self, X):
        """
        Fit the FastICA model and transform the data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - X_extracted (array-like): Extracted features.
        """
        X_extracted = self.ica.fit_transform(X)
        return X_extracted

    def set_num_components(self, num_components):
        """
        Set the number of components to extract.

        Parameters:
        - num_components (int): Number of components to extract.
        """
        self.num_components = num_components
        self.ica = FastICA(n_components=num_components)

