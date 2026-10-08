from typing import NotRequired

import numpy as np

from capymoa.regressor import RegressorResults, RegressorWindows


class PredictionIntervalWindows(RegressorWindows):
    """The windowed metrics of a prediction interval learner."""

    #: Percent of targets inside the interval in each window.
    coverage: np.ndarray
    #: Mean width of the interval, in units of the target.
    average_length: np.ndarray
    #: Mean width of the interval divided by the range of the target, in percent.
    nmpiw: np.ndarray


class PredictionIntervalResults(RegressorResults):
    """Results of evaluating a prediction interval learner.

    See :func:`evaluate_prediction_interval`. The regression metrics score the
    point prediction, which is the middle of the interval.
    """

    #: Percent of targets inside the interval.
    coverage: float
    #: Mean width of the interval, in units of the target.
    average_length: float
    #: Mean width of the interval divided by the range of the target, in percent.
    nmpiw: float
    #: The metrics of each window (see :class:`PredictionIntervalWindows`).
    #: Absent if ``window_size`` is absent.
    windowed: NotRequired[PredictionIntervalWindows]
