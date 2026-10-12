"""Open in Colab support for the notebook pages.

Colab opens `.ipynb` files from GitHub only, and ours are build artifacts (the
sources are Jupytext `.py` files). `invoke docs.colab` writes a small copy of
each notebook (no outputs, plus a setup cell that pins `version`). The Release
workflow publishes the copies to the `colab-notebooks` branch, tagged per release, and the notebook
pages link to that tag.
"""

from pathlib import Path

import nbformat

from docs.release_scripts import colab_tag

# Notebook metadata Colab needs. The rest (widget state, ...) is dropped.
KEEP_METADATA = ("kernelspec", "language_info")

_SETUP_SOURCE = """\
# Install CapyMOA (https://capymoa.org/setup/)
%pip install -q capymoa=={version}"""


def is_notebook_page(pagename: str) -> bool:
    """Is `pagename` a notebook page (not an index)?"""
    parts = pagename.split("/")
    return parts[0] == "notebooks" and len(parts) == 3 and parts[-1] != "index"


def colab_url(github_url: str, version: str, pagename: str) -> str:
    """Colab link for the notebook page `pagename` of release `version`.

    `github_url` is the repo URL, e.g. `https://github.com/owner/repo`.
    """
    repo = github_url.removeprefix("https://github.com/").rstrip("/")
    path = pagename.removeprefix("notebooks/")
    return (
        "https://colab.research.google.com/github/"
        f"{repo}/blob/{colab_tag(f'v{version}')}/{path}.ipynb"
    )


def write_colab_notebooks(src_dir: Path, out_dir: Path, version: str) -> list[Path]:
    """Write a Colab copy of each `src_dir/*/*.ipynb` to `out_dir/*/`.

    Copies have no outputs or metadata (keeps them small) and lack the cells tagged
    `remove-cell` (hidden in the docs). A setup cell on top installs CapyMOA
    `version`. Colab opens only the notebook, so each `.json` file next to it is
    written by a `%%writefile` cell after the setup cell.
    Returns the paths written.
    """
    written = []
    for src in sorted(src_dir.glob("*/*.ipynb")):
        nb = nbformat.read(src, as_version=4)
        nb.metadata = {k: v for k, v in nb.metadata.items() if k in KEEP_METADATA}
        # Drop the cells the docs site hides.
        nb.cells = [
            c for c in nb.cells if "remove-cell" not in c.metadata.get("tags", [])
        ]
        for cell in nb.cells:
            cell.metadata = {}
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        setup = [nbformat.v4.new_code_cell(_SETUP_SOURCE.format(version=version))]
        for data in sorted(src.parent.glob("*.json")):
            setup.append(
                nbformat.v4.new_code_cell(
                    f"%%writefile {data.name}\n{data.read_text()}"
                )
            )
        nb.cells[:0] = setup
        nbformat.validate(nb)
        dest = out_dir / src.parent.name / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, dest)
        written.append(dest)
    return written
