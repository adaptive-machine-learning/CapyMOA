# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
from pathlib import Path
from capymoa.__about__ import __version__
from docs.util.github_link import make_linkcode_resolve

project = 'CapyMOA'
copyright = '2024 CapyMOA Developers'
author = 'Heitor Murilo Gomes, Anton Lee, Nuwan Gunasekara, Marco Heyden, Yibin Sun, Guilherme Weigert Cassales'
release = __version__
html_title = f"{project}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.linkcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx.ext.doctest",
    "myst_parser",
]

nitpick_ignore_regex = [
    ('py:class', r'sklearn\..*'),
    ('py:class', r'numpy\..*'),
    ('py:class', r'pathlib\..*'),
    ('py:class', r'abc\..*'),
    ('py:class', r'moa\..*'),
    ('py:class', r'com\..*'),
    ('py:class', r'java\..*'),
    ('py:class', r'org\..*'),
    ('py:class', r'torch\..*'),

]
bibtex_bibfiles = ['references.bib']

autoclass_content = 'class'
autodoc_class_signature = 'separated'
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extlinks = {
    'wiki': ('https://en.wikipedia.org/wiki/%s', ''),
    'moa-api': ('https://javadoc.io/doc/nz.ac.waikato.cms.moa/moa/latest/%s', ''),
    'doi': ('https://doi.org/%s', ''),
    'sklearn': ('https://scikit-learn.org/stable/modules/generated/sklearn.%s.html', 'sklearn.%s'),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

# Setup symbolic links for notebooks

python_maximum_signature_line_length = 88

notebooks = Path("../notebooks")
notebook_doc_source = Path("notebooks")
if not notebook_doc_source.exists():
    os.symlink(notebooks, notebook_doc_source)

# -- Options for InterSphinx -------------------------------------------------
# See: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
# tldr: This allows us to link to other projects' documentation

intersphinx_mapping = {
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

""" Options for linkcode extension ------------------------------------------
The linkcode extension is used to provide links to the source code of functions
and classes in the documentation.
"""

linkcode_resolve = make_linkcode_resolve(
    "capymoa",
    (
        "https://github.com/adaptive-machine-learning/"
        "CapyMOA/blob/{revision}/src/"
        "{package}/{path}#L{lineno}"
    ),
)

""" Options for the Theme ---------------------------------------------------
"""
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/adaptive-machine-learning/CapyMOA",  # required
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/capymoa/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/spd2gQJGAb",
            "icon": "fa-brands fa-discord",
            "type": "fontawesome",
        }
   ]
}
