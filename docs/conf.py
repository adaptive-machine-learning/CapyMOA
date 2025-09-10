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

discord_link = "https://discord.gg/spd2gQJGAb"
contact_email = "heitor.gomes@vuw.ac.nz"
capymoa_github = "https://github.com/adaptive-machine-learning/CapyMOA"

project = "CapyMOA"
copyright = "2024 CapyMOA Developers"
author = "Heitor Murilo Gomes, Anton Lee, Nuwan Gunasekara, Marco Heyden, Yibin Sun, Guilherme Weigert Cassales"
release = __version__
html_title = f"{project}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinxcontrib.programoutput",
    "myst_parser",
]

nitpick_ignore_regex = [
    ("py:class", r".*\._[\w_]*"),  # Ignore private classes from nitpick errors
    ("py:obj", r".*\._[\w_]*"),  # Ignore private objects from nitpick errors
    ("py:class", r"abc\..*"),
    ("py:class", r"com\..*"),
    ("py:class", r"java\..*"),
    ("py:class", r"moa\..*"),
    ("py:class", r"numpy\..*"),
    ("py:class", r"org\..*"),
    ("py:class", r"pandas\..*"),
    ("py:class", r"pathlib\..*"),
    ("py:class", r"sklearn\..*"),
    ("py:class", r"torch\..*"),
    ("py:class", r"tqdm\..*"),
]

toc_object_entries_show_parents = "hide"
autosummary_ignore_module_all = False
autosummary_generate = True
autosummary_context = {
    # List of modules that we do not include inherited members in. This is
    # usually because they import from torch.nn.Module or similar large
    # classes.
    "inherited_members_module_denylist": ["capymoa.ann"]
}

autodoc_member_order = "groupwise"
autodoc_class_signature = "separated"

# Suppress the leading module names of the typehints in the documentation.
# This is useful to abbreviate the typehints in the documentation.
autodoc_typehints_format = "short"

# The default argument values of functions will be not evaluated on generating
# document. It preserves them as is in the source code.
autodoc_preserve_defaults = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extlinks = {
    "wiki": ("https://en.wikipedia.org/wiki/%s", ""),
    "moa-api": ("https://javadoc.io/doc/nz.ac.waikato.cms.moa/moa/latest/%s", ""),
    "doi": ("https://doi.org/%s", ""),
    "sklearn": (
        "https://scikit-learn.org/stable/modules/generated/sklearn.%s.html",
        "sklearn.%s",
    ),
    "github": ("https://github.com/%s", "GitHub %s"),
}

# Add refs to the documentation
rst_epilog = f"""
.. _Discord: {discord_link}
.. _Email: mailto:{contact_email}
.. _CapyMOA GitHub: {capymoa_github}
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/citation.css"]

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
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
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
    "show_toc_level": 3,
    "icon_links": [
        {
            "name": "GitHub",
            "url": capymoa_github,
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
            "url": discord_link,
            "icon": "fa-brands fa-discord",
            "type": "fontawesome",
        },
        {
            "name": "Email",
            "url": f"mailto:{contact_email}",
            "icon": "fa-solid fa-envelope",
            "type": "fontawesome",
        },
    ],
}
