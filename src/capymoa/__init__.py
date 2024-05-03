from ._prepare_jpype import _start_jpype, about
from .__about__ import __version__

# It is important that this is called before importing any other module
_start_jpype()

__all__ = ["about", "__version__"]