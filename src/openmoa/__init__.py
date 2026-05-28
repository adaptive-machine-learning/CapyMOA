# SPDX-License-Identifier: BSD-3-Clause
"""Machine learning library tailored for data streams."""

from ._prepare_jpype import about
from .__about__ import __version__


def __getattr__(name: str):
    # Lazy-load the stream submodule so that `import openmoa` does not
    # require Java.  Java is only started when a MOA-dependent class is
    # actually instantiated.
    if name == "stream":
        from openmoa import stream as _stream
        return _stream
    raise AttributeError(f"module 'openmoa' has no attribute {name!r}")


__all__ = [
    "about",
    "__version__",
    "stream",
]
