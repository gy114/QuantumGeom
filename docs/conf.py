# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QuantumGeom'
copyright = '2024, Gen Yin'
author = 'Gen Yin'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Handles automodule, autoclass, automethod directives
    'sphinx.ext.napoleon',  # If using Google or NumPy style docstrings
    'sphinx.ext.viewcode',  # Adds links to highlighted source code
    'breathe',              # If you're integrating Doxygen XML output
]

breathe_default_project = "QuantumGeom"

# Keep method signatures on their API pages and out of the sidebar.
toc_object_entries = False

templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {'collapse_navigation': False}
html_logo = 'static/logo.png'
html_static_path = ['static']
