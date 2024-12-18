# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, target_dir)
print(target_dir)


project = 'AutoDiff'
copyright = '2022, team01'
author = 'team01'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary', 'sphinx.ext.napoleon',
  'sphinx_copybutton',
  'sphinx.ext.viewcode',
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []
add_module_names = False
autodoc_typehints = "description"

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__str__,__repr__,__neg__,__add__,__sub__,__mul__,__truediv__,__pow__,__radd__,__rsub__,__rmul__,__rtruediv__,__rpow__,__lt__,__gt__,__le__,__ge__,__eq__,__ne__',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
