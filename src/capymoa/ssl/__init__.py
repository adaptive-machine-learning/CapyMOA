from capymoa._optional import lazy_torch_attrs
from ._sleade import SLEADE

__all__ = ["OSNN", "SLEADE"]


#: Names that need PyTorch. Imported on first access so ``import capymoa`` stays
#: torch-free -- see :mod:`capymoa._optional`.
_LAZY = {
    "OSNN": "._osnn",
}

__getattr__, __dir__ = lazy_torch_attrs(__name__, _LAZY, "OSNN", __all__)
