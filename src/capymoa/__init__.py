"""Machine learning library tailored for data streams."""

import importlib
import importlib.abc
import importlib.machinery
import sys

from ._prepare_jpype import _start_jpype, about
from .__about__ import __version__

# It is important that this is called before importing any other module
_start_jpype()

# Imported here (after _start_jpype) to ensure jpype has been started
from . import core  # noqa: E402

__all__ = [
    "about",
    "__version__",
    "core",
]

# Modules that have been renamed/merged. Old dotted name -> new dotted name.
# Importing (or `from capymoa import ...`-ing) an old name raises a
# ModuleRenamedError pointing at the new module.
_RENAMED_MODULES = {
    "capymoa.instance": "capymoa.core",
    "capymoa.type_alias": "capymoa.core",
    "capymoa._cli": "capymoa.core.moa._cli",
    "capymoa.splitcriteria": "capymoa.core.moa.splitcriteria",
    "capymoa.misc": "capymoa.core.io",
    "capymoa.ann": "capymoa.core.torch.ann",
    "capymoa.clusterers": "capymoa.cluster",
    "capymoa.feature_selection": "capymoa.feature",
    "capymoa.prediction_interval": "capymoa.uncertainty",
    "capymoa.interval": "capymoa.uncertainty",
}


class ModuleRenamedError(ImportError):
    """Raised when importing a module that has been renamed/merged."""


class _RenamedModuleLoader(importlib.abc.Loader):
    def __init__(self, old_name: str, new_name: str) -> None:
        self._old_name = old_name
        self._new_name = new_name

    def create_module(self, spec):
        return None  # use default module creation

    def exec_module(self, module) -> None:
        raise ModuleRenamedError(
            f"{self._old_name} is deprecated, use {self._new_name} instead"
        )


class _RenamedModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        new_name = _RENAMED_MODULES.get(fullname)
        if new_name is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname, _RenamedModuleLoader(fullname, new_name)
        )


sys.meta_path.insert(0, _RenamedModuleFinder())
