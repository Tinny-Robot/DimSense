# DimSense: Feature Selection and Extraction Library

[![Downloads](https://static.pepy.tech/badge/dimsense)](https://pepy.tech/project/dimsense) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DimSense is a Python library designed to streamline the process of feature selection and extraction in machine learning projects. Whether you're working with large datasets or aiming to enhance model performance, DimSense offers a collection of methods to help you identify crucial features and reduce dimensionality effectively.

## Installation

You can install DimSense using pip:

```bash
pip install dimsense
```

## Usage

DimSense provides a range of feature selection and extraction methods that can be seamlessly integrated into your machine learning pipelines. Here's a basic example demonstrating how to use DimSense's feature selection:

```python
from dimsense import FeatureSelector

# Load your dataset
X, y = load_dataset()

# Initialize the FeatureSelector
selector = FeatureSelector(method='select_k_best', num_features=10)

# Fit and transform the data
X_transformed = selector.fit_transform(X, y)
```

For more detailed examples, function explanations, and advanced usage scenarios, refer to our [documentation](https://github.com/Tinny-Robot/DimSense/blob/main/docs/index.md).

## Contributing

We welcome contributions from the community! If you'd like to contribute to DimSense, please refer to our [Contributing Guidelines](https://github.com/Tinny-Robot/DimSense/blob/main/docs/CONTRIBUTING.md).


## Testing

We take testing seriously to ensure the reliability of DimSense. You can run the test suite using the following steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Tinny-Robot/DimSense
   ```

2. Navigate to the project directory:

   ```bash
   cd DimSense
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the tests:

   ```bash
   python run_tests.py
   ```

If all tests pass, you'll see output indicating the success. If any tests fail, carefully review the error messages and traceback to identify the issue. Feel free to reach out to us if you encounter any problems!

### Continuous Integration

We also have set up continuous integration (CI) to automatically run tests whenever changes are pushed to the repository. You can view the test results and coverage reports directly in the pull request checks or on our CI provider's website.

### Test Coverage

We aim for good test coverage to ensure the robustness of our code. If you're interested in measuring the test coverage, you can do so by running:

```bash
coverage run run_tests.py
coverage report -m
```

Happy testing with DimSense!

## Changelog

For a complete list of changes and versions, please refer to the [Changelog](https://github.com/Tinny-Robot/DimSense/blob/main/CHANGELOG.md).


## License

DimSense is released under the [MIT License](https://github.com/Tinny-Robot/DimSense/blob/main/LICENSE).

## Contact

If you have any questions or feedback, feel free to reach out to us at [handanfoun@gmail.com](mailto:handanfoun@gmail.com).

Happy feature engineering with DimSense!
