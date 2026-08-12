"""Shared fixtures for simulating/asserting PyTorch's absence.

See docs/setup/developer.rst for when to use these vs. ``pytest.importorskip``
or ``skipif(find_spec(...))``.
"""

import subprocess
import sys
import textwrap

import pytest

#: Prelude that makes ``import torch`` fail, as it would without the extra.
_BLOCK_TORCH = """
import sys

class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith(("torch.", "torchvision")):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _BlockTorch())
"""


@pytest.fixture
def run_without_torch():
    """Return a function that runs code in a subprocess where importing torch fails."""

    def _run(body: str) -> subprocess.CompletedProcess:
        script = _BLOCK_TORCH + textwrap.dedent(body)
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )

    return _run
