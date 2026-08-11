"""Prediction Interval.

Prediction interval methods estimate a range of plausible values around a
regression prediction rather than a single point. In data stream learning,
these intervals must be maintained incrementally as new instances arrive and
the underlying distribution drifts.
"""

from ._mean_and_standard_deviation_estimation import MVE
from ._adaptive_prediction_interval import AdaPI

__all__ = [
    "MVE",
    "AdaPI",
]
