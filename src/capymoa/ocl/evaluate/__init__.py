"""Evaluate online continual learning (OCL) and its result types."""

from .._evaluate import evaluate_ocl
from . import events
from ._results import Anytime, OCLResults, OnlineResults, PerTask, TaskWindows

__all__ = [
    "Anytime",
    "OCLResults",
    "OnlineResults",
    "PerTask",
    "TaskWindows",
    "evaluate_ocl",
    "events",
]
