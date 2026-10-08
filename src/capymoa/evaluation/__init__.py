"""Shared parts of evaluation.

Each research domain has its own results type and ``evaluate_*`` function, for
example :func:`capymoa.classifier.evaluate_classifier`. This module holds what
they share: :class:`RunInfo` (the base of every result) and
:func:`prequential_evaluation`, which picks the ``evaluate_*`` function from the
type of the learner.
"""

from . import results
from ._prequential import prequential_evaluation
from .results import RunInfo

__all__ = [
    "RunInfo",
    "prequential_evaluation",
    "results",
]
