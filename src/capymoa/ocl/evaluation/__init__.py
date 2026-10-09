"""Events and result types for evaluating online continual learning."""

from . import events
from ._results import Anytime, OCLResults, OnlineResults, PerTask, TaskWindows

__all__ = [
    "Anytime",
    "OCLResults",
    "OnlineResults",
    "PerTask",
    "TaskWindows",
    "events",
]
