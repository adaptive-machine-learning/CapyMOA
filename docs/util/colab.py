"""Open in Colab support for the notebook pages.

Colab opens `.ipynb` files from GitHub only, and ours are build artifacts (the
sources are Jupytext `.py` files). `invoke docs.colab` writes a small copy of
each notebook (no outputs, plus a setup cell). The Release workflow publishes
the copies to the `colab-notebooks` branch, and the notebook pages link there.
"""

from pathlib import Path

import nbformat

COLAB_BRANCH = "colab-notebooks"

_SETUP_SOURCE = """\
# Colab only: install CapyMOA. Other environments skip this cell.
import sys

if "google.colab" in sys.modules:
    %pip install -q capymoa"""


def is_notebook_page(pagename: str) -> bool:
    """Is `pagename` a notebook page (not an index)?"""
    parts = pagename.split("/")
    return parts[0] == "notebooks" and len(parts) == 3 and parts[-1] != "index"


def colab_url(github_url: str, pagename: str) -> str:
    """Colab link for the notebook page `pagename`.

    `github_url` is the repo URL, e.g. `https://github.com/owner/repo`.
    """
    repo = github_url.removeprefix("https://github.com/").rstrip("/")
    path = pagename.removeprefix("notebooks/")
    return (
        "https://colab.research.google.com/github/"
        f"{repo}/blob/{COLAB_BRANCH}/{path}.ipynb"
    )


def write_colab_notebooks(src_dir: Path, out_dir: Path) -> list[Path]:
    """Write a Colab copy of each `src_dir/*/*.ipynb` to `out_dir/*/`.

    Copies have no outputs (keeps them small) and get a setup cell on top.
    Returns the paths written.
    """
    written = []
    for src in sorted(src_dir.glob("*/*.ipynb")):
        nb = nbformat.read(src, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        nb.cells.insert(0, nbformat.v4.new_code_cell(_SETUP_SOURCE))
        nbformat.validator.normalize(nb)
        dest = out_dir / src.parent.name / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, dest)
        written.append(dest)
    return written
