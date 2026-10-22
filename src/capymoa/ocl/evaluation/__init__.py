"""Deprecated alias of :mod:`capymoa.ocl.evaluate`. Use that module instead."""

import warnings

from capymoa.ocl.evaluate import (
    Anytime,
    OCLResults,
    OnlineResults,
    PerTask,
    TaskWindows,
    evaluate_ocl,
    events,
)

warnings.warn(
    "capymoa.ocl.evaluation is deprecated, use capymoa.ocl.evaluate instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Anytime",
    "OCLResults",
    "OnlineResults",
    "PerTask",
    "TaskWindows",
    "evaluate_ocl",
    "events",
]
