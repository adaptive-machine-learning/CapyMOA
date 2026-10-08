from typing import NotRequired

import numpy as np

from capymoa.evaluation.results import RunInfo, Windows


class AnomalyWindows(Windows):
    """The windowed metrics of an anomaly detector, one entry per window."""

    #: Area under the ROC curve of the anomaly scores (0 to 1).
    auc: np.ndarray
    #: Area under the ROC curve over a sliding window of the stream (0 to 1).
    s_auc: np.ndarray


class AnomalyResults(RunInfo):
    """Results of evaluating an anomaly detector. See :func:`evaluate_anomaly`.

    Metrics are over the whole stream.
    """

    #: Area under the ROC curve of the anomaly scores (0 to 1).
    auc: float
    #: Area under the ROC curve over a sliding window of the stream, averaged
    #: (0 to 1).
    s_auc: float
    #: The metrics of each window (see :class:`AnomalyWindows`). Absent if
    #: ``window_size`` is absent.
    windowed: NotRequired[AnomalyWindows]
