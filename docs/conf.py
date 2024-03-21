# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
from pathlib import Path

project = 'CapyMOA'
copyright = '2024, Heitor Murilo Gomes, Anton Lee, Nuwan Gunasekara, Marco Heyden'
author = 'Heitor Murilo Gomes, Anton Lee, Nuwan Gunasekara, Marco Heyden'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx.ext.doctest",
]

autoclass_content = 'both'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extlinks = {
    'wiki': ('https://en.wikipedia.org/wiki/%s', ''),
    'moa-api': ('https://javadoc.io/doc/nz.ac.waikato.cms.moa/moa/latest/%s', ''),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']

# Setup symbolic links for notebooks

notebooks = Path("../notebooks")
notebook_doc_source = Path("notebooks")
if not notebook_doc_source.exists():
    os.symlink(notebooks, notebook_doc_source)
