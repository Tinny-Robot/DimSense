import numpy as np
import unittest
from dimsense.feature_extraction import PCAExtractor, TSNEExtractor, TFIDFExtractor, CountVectorizerExtractor, LatentDirichletAllocationExtractor, FastICAExtractor, AutoencoderExtractor
from sklearn.datasets import load_iris


# Classes
class TestPCAExtractor(unittest.TestCase):
    def test_fit_transform(self):
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        
        # Initialize the PCAExtractor
        extractor = PCAExtractor(num_components=2)
        
        # Test fit_transform
        X_pca = extractor.fit_transform(X)
        
        self.assertEqual(X_pca.shape[1], 2)

class TestTSNEExtractor(unittest.TestCase):
    def test_fit_transform(self):
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        
        # Initialize TSNEExtractor
        extractor = TSNEExtractor(num_components=2)
        
        # Test fit_transform
        X_tsne = extractor.fit_transform(X)
        
        self.assertEqual(X_tsne.shape[1], 2)

class TestTFIDFExtractor(unittest.TestCase):
    def test_fit_transform(self):
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        
        # Initialize TFIDFExtractor
        extractor = TFIDFExtractor(max_features=10)
        
        # Test fit_transform
        X_tfidf = extractor.fit_transform(X)
        
        self.assertEqual(X_tfidf.shape[1], 10)

class TestCountVectorizerExtractor(unittest.TestCase):
    def test_fit_transform(self):
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        
        # Initialize CountVectorizerExtractor
        extractor = CountVectorizerExtractor(max_features=10)
        
        # Test fit_transform
        X_count = extractor.fit_transform(X)
        
        self.assertEqual(X_count.shape[1], 10)

class TestLatentDirichletAllocationExtractor(unittest.TestCase):
    def test_fit_transform(self):
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        
        # Initialize LatentDirichletAllocationExtractor
        extractor = LatentDirichletAllocationExtractor(num_topics=3)
        
        # Test fit_transform
        X_lda = extractor.fit_transform(X)
        
        self.assertEqual(X_lda.shape[1], 3)

class TestFastICAExtractor(unittest.TestCase):
    def test_fit_transform(self):
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        
        # Initialize FastICAExtractor
        extractor = FastICAExtractor(num_components=2)
        
        # Test fit_transform
        X_ica = extractor.fit_transform(X)
        
        self.assertEqual(X_ica.shape[1], 2)

class TestAutoencoderExtractor(unittest.TestCase):
    def test_fit_transform(self):
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        
        # Initialize AutoencoderExtractor
        extractor = AutoencoderExtractor(encoding_dim=2)
        
        # Test fit_transform
        X_ae = extractor.fit_transform(X)
        
        self.assertEqual(X_ae.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
