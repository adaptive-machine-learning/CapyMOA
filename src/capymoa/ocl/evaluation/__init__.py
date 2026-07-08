"""Evaluate online continual learning in classification tasks."""

from ._loop import ocl_train_eval_loop
from ._metrics import OCLMetrics
from . import events

__all__ = ["OCLMetrics", "ocl_train_eval_loop", "events"]
