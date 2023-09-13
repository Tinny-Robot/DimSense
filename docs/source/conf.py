import os
import sys


# -- Path setup --------------------------------------------------------------

# Add the project's root directory to the sys.path if needed.
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'DimSense'
author = 'Nathaniel Handan'

# The full version, including alpha/beta/rc tags
release = '1.0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

htmlhelp_basename = 'DimSensedoc'

# -- Extension configuration -------------------------------------------------

# Add any additional Sphinx extension configuration here.

# -- Napoleon settings -------------------------------------------------------

# Enable type annotations support in docstrings
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = True
