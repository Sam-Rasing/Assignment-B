# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Assignment_a_doc'
copyright = '2025, Sam Rasing'
author = 'Sam Rasing'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []

# -- Path setup -------------------------
import os, sys
sys.path.insert(0, os.path.abspath('..'))        
sys.path.insert(0, os.path.abspath('../../Code')) 

# print python search paths sphinx uses
print('Python sys.path used by sphinx:')
for p in sys.path:
	print(' -', p)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
