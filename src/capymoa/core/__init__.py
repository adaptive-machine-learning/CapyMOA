"""Shared utilities and core types used across CapyMOA."""

# `_instance` must be imported before `io`/`moa`/`torch`: those submodules
# (transitively) import names from `capymoa.core`, which requires this
# module's own `_instance` re-exports to already be bound. isort would
# normally alphabetize `from . import ...` above `from ._instance import
# ...`, which reintroduces that circular import, hence `isort: skip`.
from ._instance import (  # isort: skip
    FeatureVector,
    Instance,
    Label,
    LabeledInstance,
    LabelIndex,
    LabelProbabilities,
    RegressionInstance,
    TargetValue,
    _AnyInstance,
)
from ._argmax import random_argmax  # isort: skip
from . import io, moa, torch

__all__ = [
    "FeatureVector",
    "Instance",
    "Label",
    "LabelIndex",
    "LabelProbabilities",
    "LabeledInstance",
    "RegressionInstance",
    "TargetValue",
    "_AnyInstance",
    "io",
    "moa",
    "random_argmax",
    "torch",
]
