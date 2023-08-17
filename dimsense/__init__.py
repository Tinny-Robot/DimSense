"""
DimSense: A Feature Selection and Extraction Library
"""

from .feature_selection import FeatureSelector
from .feature_extraction import PCAExtractor, TSNEExtractor, TFIDFExtractor, CountVectorizerExtractor, LatentDirichletAllocationExtractor, FastICAExtractor
from .utils import normalize_data, standardize_data, scatter_plot, detect_outliers, validate_data

__version__ = "0.1.0"
