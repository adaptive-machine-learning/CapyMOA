"""Tests for deprecated/renamed module redirects in ``capymoa.__init__``."""

import importlib
import sys

import pytest

from capymoa import _RENAMED_MODULES, ModuleRenamedError


@pytest.fixture(autouse=True)
def _clean_sys_modules():
    """Ensure renamed modules are re-imported (and re-raise) for each test."""
    yield
    for old_name in _RENAMED_MODULES:
        sys.modules.pop(old_name, None)


def _needs_torch(old_name: str, new_name: str) -> bool:
    return new_name == "capymoa.core.torch.ann"


@pytest.mark.parametrize(
    "old_name,new_name",
    [
        pytest.param(old_name, new_name, marks=pytest.mark.torch)
        if _needs_torch(old_name, new_name)
        else (old_name, new_name)
        for old_name, new_name in sorted(_RENAMED_MODULES.items())
    ],
)
def test_renamed_module_raises(old_name, new_name):
    if _needs_torch(old_name, new_name):
        pytest.markskip("torch")

    with pytest.raises(ModuleRenamedError, match=new_name):
        importlib.import_module(old_name)

    # the new module should still be importable and usable
    importlib.import_module(new_name)
