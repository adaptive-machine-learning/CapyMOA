"""Helpers for exposing optional-dependency-backed names lazily.

CapyMOA keeps PyTorch out of the default installation (see the ``torch`` extra in
``pyproject.toml``). Some public names -- ``capymoa.classifier.Finetune``,
``capymoa.anomaly.Autoencoder``, the ``Batch*`` base classes, everything in
``capymoa.ocl`` and ``capymoa.ann`` -- genuinely need PyTorch and cannot be
imported without it.

Re-exporting those eagerly from a package ``__init__`` would drag PyTorch into
``import capymoa``, which defeats the whole point. Instead each package declares
them here and they are imported on first attribute access, using :pep:`562`
module-level ``__getattr__``.

The user-visible behaviour is unchanged when PyTorch *is* installed:
``from capymoa.classifier import Finetune`` works exactly as before. Without
PyTorch the same import raises
:class:`~capymoa.exception.OptionalDependencyError` naming the feature and the
install command, instead of a bare ``ModuleNotFoundError``.
"""

from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Tuple

__all__ = ["lazy_torch_attrs"]


def lazy_torch_attrs(
    package: str,
    mapping: Mapping[str, str],
    feature: str,
    static: List[str] = (),
) -> Tuple[Callable[[str], Any], Callable[[], List[str]]]:
    """Build ``__getattr__``/``__dir__`` that import torch-backed names lazily.

    Use it at the bottom of a package ``__init__.py``::

        _LAZY = {"Finetune": "._finetune"}
        __getattr__, __dir__ = lazy_torch_attrs(__name__, _LAZY, "Finetune", __all__)

    :param package: ``__name__`` of the calling package.
    :param mapping: Public name -> module to import it from, relative to
        ``package`` (e.g. ``{"Finetune": "._finetune"}``).
    :param feature: What the user was reaching for, used in the error message.
    :param static: Names already imported eagerly, so ``dir()`` stays complete.
    :return: ``(__getattr__, __dir__)`` to assign in the calling module.
    """
    lazy: Dict[str, str] = dict(mapping)
    known = sorted(set(static) | set(lazy))

    def __getattr__(name: str) -> Any:
        module_name = lazy.get(name)
        if module_name is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        # Raise a helpful error before the ModuleNotFoundError from the import.
        from capymoa.exception import _requires_torch

        _requires_torch(feature)
        module = import_module(module_name, package)
        value = getattr(module, name)
        # Cache on the package so repeated access skips this machinery.
        setattr(import_module(package), name, value)
        return value

    def __dir__() -> List[str]:
        # Union with the module's real namespace rather than replacing it.
        # Returning only the curated list hides eagerly-imported names from
        # tools that enumerate modules via dir() -- Sphinx autosummary then
        # documents classes under their private module path and every
        # cross-reference to them breaks.
        module = import_module(package)
        return sorted(set(vars(module)) | set(known))

    return __getattr__, __dir__
