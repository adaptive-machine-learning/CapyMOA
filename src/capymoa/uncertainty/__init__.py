"""Uncertainty.

Uncertainty methods say how sure a learner is about a prediction, rather than
giving a single value on its own. In data stream learning these estimates are
updated as each new instance arrives, so they keep working when the underlying
distribution drifts.

This module currently provides prediction intervals for regression. A
prediction interval gives a range of plausible values around a point
prediction.
"""

from ._adaptive_prediction_interval import AdaPI
from ._evaluate import evaluate_prediction_interval, evaluate_prediction_intervals
from ._mean_and_standard_deviation_estimation import MVE
from ._results import PredictionIntervalResults, PredictionIntervalWindows

__all__ = [
    "MVE",
    "AdaPI",
    "PredictionIntervalResults",
    "PredictionIntervalWindows",
    "evaluate_prediction_interval",
    "evaluate_prediction_intervals",
]
