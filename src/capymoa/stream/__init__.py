from capymoa._optional import lazy_torch_attrs
from ._stream import (
    Stream,
    Schema,
    ARFFStream,
    NumpyStream,
    MOAStream,
)
from ._csv_stream import CSVStream
from ._stream_from_file import stream_from_file
from . import drift, generator, preprocessing

__all__ = [
    "Stream",
    "Schema",
    "ARFFStream",
    "TorchStream",
    "CSVStream",
    "drift",
    "generator",
    "preprocessing",
    "NumpyStream",
    "MOAStream",
    "stream_from_file",
]


#: Names that need PyTorch. Imported on first access so ``import capymoa`` stays
#: torch-free -- see :mod:`capymoa._optional`.
_LAZY = {
    "TorchStream": ".torch",
}

__getattr__, __dir__ = lazy_torch_attrs(__name__, _LAZY, "TorchStream", __all__)
