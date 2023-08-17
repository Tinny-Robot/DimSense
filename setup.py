from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dimsense",
    version="0.1.0",
    author="Nathaniel Handan",
    author_email="handanfoun@gmail.com",
    description="A feature selection and extraction library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tinny-Robot/DimSense",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
        'pandas',
        'tensorflow'
        # Add other dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
