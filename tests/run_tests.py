import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (DimSense) to the Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import unittest
from test_feature_selection import TestFeatureSelector
from test_feature_extraction import TestPCAExtractor, TestTSNEExtractor, TestTFIDFExtractor, TestCountVectorizerExtractor, TestLatentDirichletAllocationExtractor, TestFastICAExtractor, TestAutoencoderExtractor

# Create a test suite
test_suite = unittest.TestSuite()

# Add test classes from feature extraction and feature selection
test_suite.addTest(unittest.makeSuite(TestFeatureSelector))
test_suite.addTest(unittest.makeSuite(TestPCAExtractor))
test_suite.addTest(unittest.makeSuite(TestTSNEExtractor))
test_suite.addTest(unittest.makeSuite(TestTFIDFExtractor))
test_suite.addTest(unittest.makeSuite(TestCountVectorizerExtractor))
test_suite.addTest(unittest.makeSuite(TestLatentDirichletAllocationExtractor))
test_suite.addTest(unittest.makeSuite(TestFastICAExtractor))
test_suite.addTest(unittest.makeSuite(TestAutoencoderExtractor))

# Run the test suite
test_runner = unittest.TextTestRunner()
test_result = test_runner.run(test_suite)
