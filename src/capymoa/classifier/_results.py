from typing import NotRequired, TypedDict

import numpy as np

from capymoa.evaluation.results import RunInfo, Windows


class ClassifierWindows(Windows):
    """The windowed metrics of a classifier, one entry per window."""

    #: Percent of correct predictions in each window.
    accuracy: np.ndarray
    #: Cohen's kappa in percent.
    kappa: np.ndarray
    #: Kappa temporal in percent.
    kappa_t: np.ndarray
    #: Kappa M in percent.
    kappa_m: np.ndarray
    #: Class-weighted F1 score in percent.
    f1_score: np.ndarray
    #: Class-weighted precision in percent.
    precision: np.ndarray
    #: Class-weighted recall in percent.
    recall: np.ndarray
    #: Area under the ROC curve (0 to 1).
    roc_auc: np.ndarray


class PerClass(TypedDict):
    """Metrics of each class, one entry per class (in columns)."""

    #: The label of the class.
    label: np.ndarray
    #: Precision in percent.
    precision: np.ndarray
    #: Recall in percent.
    recall: np.ndarray
    #: F1 score in percent.
    f1_score: np.ndarray


class ClassifierResults(RunInfo):
    """Results of evaluating a classifier. See :func:`evaluate_classifier`.

    Metrics are over the whole stream.
    """

    #: Percent of correct predictions.
    accuracy: float
    #: Cohen's kappa in percent.
    kappa: float
    #: Kappa temporal in percent. Compares with a classifier that predicts the
    #: previous label.
    kappa_t: float
    #: Kappa M in percent. Compares with a classifier that predicts the
    #: majority label.
    kappa_m: float
    #: Class-weighted F1 score in percent.
    f1_score: float
    #: Class-weighted precision in percent.
    precision: float
    #: Class-weighted recall in percent.
    recall: float
    #: Area under the ROC curve (0 to 1). Absent if it is not defined.
    roc_auc: NotRequired[float]
    #: The metrics of each class in columns (see :class:`PerClass`).
    per_class: PerClass
    #: The metrics of each window (see :class:`ClassifierWindows`). Absent if
    #: ``window_size`` is absent.
    windowed: NotRequired[ClassifierWindows]
