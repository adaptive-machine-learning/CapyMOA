# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re
import sys
from pathlib import Path

from capymoa.__about__ import __version__
from docs.release_scripts import site_base_url
from docs.util.github_link import make_linkcode_resolve
from docs.util.sphinx_llm import (
    fix_markdown_image,
    fix_notebook_output_markdown,
    fix_unsupported_markdown_nodes,
)

# Any subprocesses created during document building should use the same python environment
os.environ["PYTHONEXECUTABLE"] = sys.executable

discord_link = "https://discord.gg/spd2gQJGAb"
contact_email = "heitor.gomes@vuw.ac.nz"
capymoa_github = "https://github.com/adaptive-machine-learning/CapyMOA"

# Read from pyproject.toml's [project.urls] Documentation, so the domain has one
# authoritative source instead of being hardcoded separately here and in
# docs/release_scripts.py's build-switcher subcommand.
SITE_BASE_URL = site_base_url()

project = "CapyMOA"
copyright = "2026 CapyMOA Developers"
author = "Heitor Murilo Gomes, Anton Lee, Nuwan Gunasekara, Marco Heyden, Yibin Sun, Guilherme Weigert Cassales"
release = __version__
html_title = f"{project}"

# Must match one of the "version" fields in switcher.json for the version-switcher
# dropdown to highlight the version being viewed. CI always builds from a `vX.Y.Z` tag,
# which is exactly how docs/release_scripts.py's build-switcher subcommand names each
# entry.
version_match = f"v{__version__}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Supersedes myst_parser: it registers the same MyST markdown parser for
    # plain `.md` docs and additionally parses/executes the notebooks (see
    # `nb_custom_formats` below).
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.programoutput",
    "sphinx_llm.txt",
    "sphinx_reredirects",
    "matplotlib.sphinxext.plot_directive",  # https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html
]

nitpick_ignore_regex = [
    (r"py:(class|obj)", r"(.*\.)?_[\w_]*"),  # Ignore private objects
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
    ("py:class", r"torchvision\..*"),
    ("py:class", r"Tensor"),
    ("py:class", r"nn\.Module"),
    # `autodoc_typehints_format = "short"` renders
    # torch.optim.optimizer.Optimizer as a bare name, like Tensor above.
    ("py:class", r"Optimizer"),
]

# These warnings are usually false positives.
suppress_warnings = [
    "myst.xref_missing",
    # The Jupytext `py:percent` source only carries `kernelspec` metadata (no
    # `language_info`, which is populated by an actual kernel run), so
    # MyST-NB highlights code cells with the plain "python" Pygments lexer.
    # That lexer doesn't understand Jupyter's `!shell`/`%magic` syntax (e.g.
    # `!uv pip install ...`); MyST-NB already degrades gracefully ("relaxed
    # mode") when that happens, so the warning is just noise.
    "misc.highlighting_failure",
]

toc_object_entries_show_parents = "hide"
autosummary_ignore_module_all = False
autosummary_generate = True

autodoc_member_order = "groupwise"
autodoc_class_signature = "separated"

# Suppress the leading module names of the typehints in the documentation.
# This is useful to abbreviate the typehints in the documentation.
autodoc_typehints_format = "short"

# The default argument values of functions will be not evaluated on generating
# document. It preserves them as is in the source code.
autodoc_preserve_defaults = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Not documentation pages: this file and the helper modules it imports,
    # but they'd otherwise match the `.py` notebook suffix registered by
    # `nb_custom_formats` below and get built as (broken) notebook pages.
    "conf.py",
    "util",
    # MyST-NB writes each notebook's executed `.ipynb` copy here (used for the
    # rendered page's "download notebook" link), inside the source directory
    # rather than under `_build`. Left unexcluded, Sphinx also picks these
    # `.ipynb` copies up as extra source documents in their own right.
    "jupyter_execute",
]

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
.. _CapyMOA Releases: {capymoa_github}/releases
.. _CapyMOA Docs: {SITE_BASE_URL}/
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/citation.css", "css/llm-page-actions.css", "css/dataframe.css"]
html_js_files = ["js/llm-page-actions.js"]
html_show_sourcelink = False

# "auto" (the default) names the generated markdown files "<page>.html.md",
# but sphinx_markdown_builder writes cross-page links as "<page>.md", which
# breaks navigation between generated markdown files. "replace" makes the
# file names match those links.
llms_txt_suffix_mode = "replace"

# Setup symbolic links for notebooks

python_maximum_signature_line_length = 88

notebooks = Path("../notebooks")
notebook_doc_source = Path("notebooks")
if not notebook_doc_source.exists():
    os.symlink(notebooks, notebook_doc_source)

# Redirects for the notebooks that moved into domain subfolders when the
# `notebooks/` layout was reorganized (PR #420). `getting_started` and
# `evaluation` covered both classification and regression before that
# split into per-domain notebooks; the old URL redirects to the classifier
# variant.
redirects = {
    "notebooks/00_getting_started": "/notebooks/classifier/getting_started.html",
    "notebooks/01_evaluation": "/notebooks/classifier/evaluation.html",
    "notebooks/02_sklearn": "/notebooks/common/sklearn_models.html",
    "notebooks/03_pytorch": "/notebooks/common/pytorch_integration.html",
    "notebooks/04_drift_streams": "/notebooks/drift/drift_streams.html",
    "notebooks/05_new_learner": "/notebooks/classifier/new_learner.html",
    "notebooks/06_advanced_API": "/notebooks/common/advanced_API.html",
    "notebooks/07_pipelines": "/notebooks/common/pipelines.html",
    "notebooks/08_prediction_interval": "/notebooks/uncertainty/prediction_interval.html",
    "notebooks/09_automl": "/notebooks/automl/automl.html",
    "notebooks/10_ocl": "/notebooks/ocl/ocl.html",
    "notebooks/SSL_example": "/notebooks/ssl/ssl_example.html",
    "notebooks/anomaly_detection": "/notebooks/anomaly/anomaly_detection.html",
    "notebooks/drift_detection": "/notebooks/drift/drift_detection.html",
    "notebooks/optimizing_detectors": "/notebooks/drift/optimizing_detectors.html",
    "notebooks/parallel_ensembles": "/notebooks/classifier/parallel_ensembles.html",
    "notebooks/save_and_load_model": "/notebooks/common/save_and_load_model.html",
    "notebooks/clustering": "/notebooks/clusterer/clustering.html",
    "notebooks/feature_importance": "/notebooks/feature/feature_importance.html",
    "notebooks/ocl_event_system": "/notebooks/ocl/ocl_event_system.html",
}

# -- Options for Matplotlib Sphinx Plot Directive ----------------------------
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ["png"]

# -- Options for MyST-NB ------------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

# Sphinx never executes notebooks itself: that's a distinct step
# (`invoke docs.nb`, using nbmake's `--overwrite` flag) so notebook
# failures and Sphinx build failures show up separately. `docs.nb` bakes
# real outputs directly into each notebook's generated `.ipynb` file; when
# that file exists (checked out of the box for the plain `.py` source, or
# generated by `docs.nb`), MyST-NB picks it over the `.py` source and
# renders whatever outputs are stored in it, without executing anything.
# Without `docs.nb` having run, there's no `.ipynb` file, so MyST-NB
# falls back to the `.py` source and renders it without outputs -- fast,
# and what you want when just iterating on non-notebook doc pages.
nb_execution_mode = "off"

# -- Options for InterSphinx -------------------------------------------------
# See: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
# tldr: This allows us to link to other projects' documentation

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
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

""" Options for sphinx-copybutton --------------------------------------------
Adds a "copy" button to code blocks, stripping prompts so the copied text is
directly runnable.
"""
# Matches `>>> `/`... ` (doctests) and `$ ` (shell examples), each optionally
# followed by output lines that get excluded via `copybutton_only_copy_prompt_lines`.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

""" Options for the Theme ---------------------------------------------------
"""
html_theme_options = {
    "show_toc_level": 3,
    "logo": {
        "text": "CapyMOA",
        "image_light": "_static/logo-96x96.png",
        "image_dark": "_static/logo-96x96.png",
    },
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
    "secondary_sidebar_items": [
        "page-toc",
        "edit-this-page",
        "sourcelink",
        "components/llm-page-actions.html",
    ],
    "switcher": {
        "json_url": f"{SITE_BASE_URL}/switcher.json",
        "version_match": version_match,
    },
    # pydata_sphinx_theme defaults navbar_end to ["theme-switcher", "navbar-icon-links"];
    # it must be set explicitly here to add the version switcher without losing those.
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # Local/PR builds fetch switcher.json live from capymoa.org, which won't have an
    # entry for an in-progress/unreleased version -- checking would fail every non-release
    # build under -W (warnings-as-errors).
    "check_switcher": False,
}

autodoc_skip_member_patterns = [
    # Inheriting from torch.nn.Module creates issues so we skip them.
    r"torch\.nn\.modules\..*",
    # TypedDict inherits these dict methods, which are not useful API members.
    r"builtins\.dict\..*",
]


def autodoc_skip_member(app, obj_type, name, obj, skip, options) -> bool | None:
    if skip:
        return None
    if not hasattr(obj, "__qualname__"):
        return None
    module = getattr(obj, "__module__", None) or "builtins"
    fqn = f"{module}.{obj.__qualname__}"

    for pattern in autodoc_skip_member_patterns:
        if re.match(pattern, fqn):
            return True

    return None


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)

    # Patches for sphinx_llm extension.
    fix_notebook_output_markdown(app)
    fix_markdown_image(app)
    fix_unsupported_markdown_nodes(app)
