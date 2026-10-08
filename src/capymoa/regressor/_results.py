from typing import NotRequired

import numpy as np

from capymoa.evaluation.results import RunInfo, Windows


class RegressorWindows(Windows):
    """The windowed metrics of a regressor, one entry per window."""

    #: Mean absolute error, in units of the target.
    mae: np.ndarray
    #: Root mean squared error, in units of the target.
    rmse: np.ndarray
    #: Relative absolute error.
    rmae: np.ndarray
    #: Coefficient of determination.
    r2: np.ndarray
    #: R2 adjusted for the number of attributes.
    adjusted_r2: np.ndarray


class RegressorResults(RunInfo):
    """Results of evaluating a regressor. See :func:`evaluate_regressor`.

    Metrics are over the whole stream.
    """

    #: Mean absolute error, in units of the target.
    mae: float
    #: Root mean squared error, in units of the target.
    rmse: float
    #: Relative absolute error: the MAE divided by the MAE of predicting the
    #: previous target. Below 1 is better than that baseline.
    rmae: float
    #: Coefficient of determination. 1 is perfect, 0 is no better than the mean.
    r2: float
    #: R2 adjusted for the number of attributes.
    adjusted_r2: float
    #: The metrics of each window (see :class:`RegressorWindows`). Absent if
    #: ``window_size`` is absent.
    windowed: NotRequired[RegressorWindows]
