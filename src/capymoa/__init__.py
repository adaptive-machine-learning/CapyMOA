"""Machine learning library tailored for data streams."""

from ._prepare_jpype import _start_jpype, about
from .__about__ import __version__

# It is important that this is called before importing any other module
_start_jpype()

from . import stream  # noqa Module imported here to ensure that jpype has been started


__all__ = [
    "about",
    "__version__",
    "stream",
]
