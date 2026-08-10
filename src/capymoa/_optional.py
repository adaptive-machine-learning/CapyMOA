"""Helpers for exposing optional-dependency-backed names lazily.

CapyMOA keeps PyTorch out of the default installation (see the ``torch`` extra in
``pyproject.toml``). Some public names -- ``capymoa.classifier.Finetune``,
``capymoa.anomaly.Autoencoder``, the ``Batch*`` base classes, everything in
``capymoa.ocl`` and ``capymoa.core.torch.ann`` -- genuinely need PyTorch and cannot be
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

from functools import lru_cache
from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, MutableSequence, Tuple

__all__ = ["lazy_torch_attrs", "torch_available"]


@lru_cache(maxsize=1)
def torch_available() -> bool:
    """Whether PyTorch can be imported, resolved once per process."""
    from importlib.util import find_spec

    try:
        return find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def lazy_torch_attrs(
    package: str,
    mapping: Mapping[str, str],
    feature: str,
    all_names: MutableSequence[str],
) -> Tuple[Callable[[str], Any], Callable[[], List[str]]]:
    """Build ``__getattr__``/``__dir__`` that import torch-backed names lazily.

    Use it at the bottom of a package ``__init__.py``::

        _LAZY = {"Finetune": "._finetune"}
        __getattr__, __dir__ = lazy_torch_attrs(__name__, _LAZY, "Finetune", __all__)

    When PyTorch is **not** installed the lazy names are removed from
    ``all_names`` in place. ``__all__`` drives ``from package import *``, so
    leaving them listed would make a wildcard import resolve every torch-backed
    name and fail with :class:`~capymoa.exception.OptionalDependencyError`, even
    for a user who only wanted the core names. Dropping them means
    ``import *`` yields exactly what is usable, while an explicit
    ``from capymoa.classifier import Finetune`` still raises the actionable
    error. With PyTorch installed, ``__all__`` is left untouched.

    :param package: ``__name__`` of the calling package.
    :param mapping: Public name -> module to import it from, relative to
        ``package`` (e.g. ``{"Finetune": "._finetune"}``).
    :param feature: What the user was reaching for, used in the error message.
    :param all_names: The package's ``__all__``. Filtered in place when PyTorch
        is missing, as described above.
    :return: ``(__getattr__, __dir__)`` to assign in the calling module.
    """
    lazy: Dict[str, str] = dict(mapping)
    known = sorted(set(all_names) | set(lazy))

    if not torch_available():
        for name in lazy:
            while name in all_names:
                all_names.remove(name)

    def __getattr__(name: str) -> Any:
        module_name = lazy.get(name)
        if module_name is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        # Raise a helpful error before the ModuleNotFoundError from the import.
        from capymoa.exception import OptionalDependencyError

        if not torch_available():
            raise OptionalDependencyError("PyTorch", feature)
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
