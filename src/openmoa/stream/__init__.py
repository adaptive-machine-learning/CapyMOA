# SPDX-License-Identifier: BSD-3-Clause
from ._stream import (
    Stream,
    Schema,
    ARFFStream,
    stream_from_file,
    CSVStream,
    NumpyStream,
    MOAStream,
    ConcatStream,
    LibsvmStream,
    BagOfWordsStream,
)
from .stream_wrapper import OpenFeatureStream, EvolvingFeatureStream
from .stream_wrapper import CapriciousStream, TrapezoidalStream, EvolvableStream, ShuffledStream

def __getattr__(name: str):
    if name == "TorchClassifyStream":
        from .torch import TorchClassifyStream
        globals()["TorchClassifyStream"] = TorchClassifyStream
        return TorchClassifyStream
    if name in ("drift", "generator", "preprocessing"):
        import importlib
        mod = importlib.import_module(f".{name}", package=__name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'openmoa.stream' has no attribute {name!r}")


__all__ = [
    "Stream",
    "Schema",
    "stream_from_file",
    "ARFFStream",
    "TorchClassifyStream",
    "CSVStream",
    "OpenFeatureStream",
    "EvolvingFeatureStream",
    "drift",
    "generator",
    "preprocessing",
    "NumpyStream",
    "MOAStream",
    "ConcatStream",
    "CapriciousStream",
    "TrapezoidalStream",
    "EvolvableStream",
    "LibsvmStream",
    "BagOfWordsStream",
    "ShuffledStream",
]
