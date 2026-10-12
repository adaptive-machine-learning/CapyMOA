"""Open in Colab support for the notebook pages.

Colab opens `.ipynb` files only, and ours are build artifacts (the sources are
Jupytext `.py` files). So the docs build ships a Colab copy of each executed
notebook, with a setup cell on top, and the notebook pages link to it.
"""

from pathlib import Path

import nbformat

COLAB_DIR = "colab"

_SETUP_SOURCE = """\
# Colab only: install CapyMOA. Other environments skip this cell.
import sys

if "google.colab" in sys.modules:
    %pip install -q capymoa=={version}"""


def is_notebook_page(pagename: str) -> bool:
    """Is `pagename` a notebook page (not an index)?"""
    parts = pagename.split("/")
    return parts[0] == "notebooks" and len(parts) == 3 and parts[-1] != "index"


def colab_url(site_url: str, version: str, pagename: str) -> str:
    """Colab link for the notebook page `pagename` of docs `version`."""
    nb_url = f"{site_url}/{version}/{COLAB_DIR}/{pagename}.ipynb"
    return f"https://colab.research.google.com/url/{nb_url}"


def write_colab_notebooks(
    src_dir: Path, out_dir: Path, capymoa_version: str
) -> list[Path]:
    """Write a Colab copy of each `src_dir/*/*.ipynb` to `out_dir`.

    Each copy gets a setup cell on top. Returns the paths written.
    """
    written = []
    for src in sorted(src_dir.glob("*/*.ipynb")):
        nb = nbformat.read(src, as_version=4)
        setup = nbformat.v4.new_code_cell(_SETUP_SOURCE.format(version=capymoa_version))
        nb.cells.insert(0, setup)
        nbformat.validator.normalize(nb)
        dest = out_dir / src.parent.name / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, dest)
        written.append(dest)
    return written
