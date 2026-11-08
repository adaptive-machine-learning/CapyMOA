"""Evaluators and result types for anomaly detection.

:func:`~capymoa.anomaly.evaluate_anomaly` returns a :class:`AnomalyResults`. It uses the
evaluators below, which wrap MOA's evaluators. You can also use them in your own test-
then-train loop.
"""

from typing import NotRequired

import numpy as np
import pandas as pd
from com.yahoo.labs.samoa.instances import Attribute, DenseInstance, Instances
from java.util import ArrayList
from moa.core import InstanceExample
from moa.evaluation import (
    BasicAUCImbalancedPerformanceEvaluator,
    WindowAUCImbalancedPerformanceEvaluator,
)
from typing_extensions import TypedDict

from capymoa._utils import _translate_metric_name
from capymoa.evaluation.results import RunInfo, Windows
from capymoa.stream import Schema

#: MOA metric names that clash with the classifier names, or have no
#: translation. Keys are the MOA names. Values are the names used here.
_MOA_NAMES = {
    "Accuracy": "accuracy",
    "Kappa": "kappa",
    "KappaM": "kappa_m",
    "Recall": "recall",
    "G-Mean": "g_mean",
    "Pos/Neg ratio": "pos_neg_ratio",
    "Periodical holdout AUC": "periodical_auc",
}


def _metric_names(moa_names) -> list[str]:
    """The names used here for the metrics MOA reports, in order."""
    names = []
    for moa_name in moa_names:
        moa_name = "".join(moa_name)
        names.append(
            _MOA_NAMES.get(moa_name) or _translate_metric_name(moa_name, to="capymoa")
        )
    return names


class AnomalyWindows(Windows):
    """The windowed metrics of an anomaly detector, one entry per window."""

    #: Area under the ROC curve of the anomaly scores (0 to 1).
    auc: np.ndarray
    #: Area under the ROC curve over a sliding window of the stream (0 to 1).
    s_auc: np.ndarray
    #: Accuracy as MOA computes it.
    accuracy: np.ndarray
    #: Cohen's kappa as MOA computes it.
    kappa: np.ndarray
    #: Kappa M as MOA computes it.
    kappa_m: np.ndarray
    #: Recall as MOA computes it.
    recall: np.ndarray
    #: G-mean as MOA computes it.
    g_mean: np.ndarray
    #: Ratio of positive to negative instances in the window.
    pos_neg_ratio: np.ndarray
    #: Area under the ROC curve of a holdout over the stream, in each window.
    periodical_auc: np.ndarray


class AnomalyCumulative(TypedDict):
    """The metrics over the whole stream of an anomaly detector."""

    #: Area under the ROC curve of the anomaly scores (0 to 1).
    auc: float
    #: Area under the ROC curve over a sliding window of the stream, averaged
    #: (0 to 1).
    s_auc: float
    #: Accuracy as MOA computes it.
    accuracy: float
    #: Cohen's kappa as MOA computes it.
    kappa: float
    #: Kappa M as MOA computes it.
    kappa_m: float
    #: Recall as MOA computes it.
    recall: float
    #: G-mean as MOA computes it.
    g_mean: float
    #: Ratio of positive to negative instances.
    pos_neg_ratio: float


class AnomalyResults(RunInfo):
    """Results of evaluating an anomaly detector. See :func:`~capymoa.anomaly.evaluate_anomaly`."""

    #: The metrics over the whole stream (see
    #: :class:`~capymoa.anomaly.evaluate.AnomalyCumulative`).
    cumulative: AnomalyCumulative

    #: The metrics of each window (see
    #: :class:`~capymoa.anomaly.evaluate.AnomalyWindows`). Absent if ``window_size`` is
    #: absent.
    windowed: NotRequired[AnomalyWindows]


class AnomalyDetectionEvaluator:
    """
    Wrapper for the Anomaly (AUC) Performance Evaluator from MOA. By default, it uses the
    BasicAUCImbalancedPerformanceEvaluator
    """

    def __init__(
        self,
        schema: Schema = None,
        window_size=None,
    ):
        self.instances_seen = 0
        self.result_windows = []
        self.window_size = window_size

        self.moa_basic_evaluator = BasicAUCImbalancedPerformanceEvaluator()

        self.moa_basic_evaluator.calculateAUC.set()

        _attributeValues = ArrayList()
        self.pred_template = [0, 0]

        self.schema = schema
        self._header = None
        if self.schema is not None:
            if self.schema.get_label_indexes() is not None:
                for value in self.schema.get_label_indexes():
                    _attributeValues.append(value)
                _classAttribute = Attribute("Class", _attributeValues)
                attSub = ArrayList()
                attSub.append(_classAttribute)
                self._header = Instances("", attSub, 1)
                self._header.setClassIndex(0)
            else:
                raise ValueError(
                    "Schema was not initialised properly, please define a proper Schema."
                )
        else:
            raise ValueError("Schema is None, please define a proper Schema.")

        # Create the denseInstance just once and keep reusing it by changing the classValue (more efficient).
        self._instance = DenseInstance(1)
        self._instance.setDataset(self._header)

    def __str__(self):
        return str(self.metrics_dict())

    def get_instances_seen(self):
        return self.instances_seen

    def update(self, y_target_index: int, score: float):
        """Update the evaluator with the ground-truth and the prediction.

        :param y_target_index: The ground-truth class index. This is NOT
            the actual class value, but the index of the class value in the
            schema.
        :param score: The predicted scores. Should be in the range [0, 1].
        """
        if not isinstance(y_target_index, (np.integer, int)):
            raise TypeError(
                f"y_target_index must be an integer, not {type(y_target_index)}"
            )

        # Notice, in MOA the class value is an index, not the actual value
        # (e.g. not "one" but 0 assuming labels=["one", "two"])
        self._instance.setClassValue(y_target_index)
        example = InstanceExample(self._instance)

        self.moa_basic_evaluator.addResult(example, [1 - score, score])

        self.instances_seen += 1

        # If the window_size is set, then check if it should record the intermediary results.
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            performance_values = self.metrics()
            self.result_windows.append(performance_values)

    def metrics_header(self):
        performance_measurements = self.moa_basic_evaluator.getPerformanceMeasurements()
        return _metric_names(m.getName() for m in performance_measurements)

    def metrics(self):
        return [
            measurement.getValue()
            for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()
        ]

    def metrics_dict(self):
        return {
            header: value
            for header, value in zip(self.metrics_header(), self.metrics())
        }

    def metrics_per_window(self):
        return pd.DataFrame(self.result_windows, columns=self.metrics_header())

    def auc(self):
        index = self.metrics_header().index("auc")
        return self.metrics()[index]

    def s_auc(self):
        index = self.metrics_header().index("s_auc")
        return self.metrics()[index]


class AnomalyDetectionWindowedEvaluator:
    """
    Wrapper for the AUC Performance Evaluator from MOA. By default, it uses the
    WindowAUCImbalancedPerformanceEvaluator
    """

    def __init__(
        self,
        schema: Schema = None,
        window_size=None,
    ):
        self.instances_seen = 0
        self.result_windows = []
        self.window_size = window_size

        self.moa_evaluator = WindowAUCImbalancedPerformanceEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)

        _attributeValues = ArrayList()
        self.pred_template = [0, 0]

        self.schema = schema
        self._header = None
        if self.schema is not None:
            if self.schema.get_label_indexes() is not None:
                for value in self.schema.get_label_indexes():
                    _attributeValues.append(value)
                _classAttribute = Attribute("Class", _attributeValues)
                attSub = ArrayList()
                attSub.append(_classAttribute)
                self._header = Instances("", attSub, 1)
                self._header.setClassIndex(0)
            else:
                raise ValueError(
                    "Schema was not initialised properly, please define a proper Schema."
                )
        else:
            raise ValueError("Schema is None, please define a proper Schema.")

        # Create the denseInstance just once and keep reusing it by changing the classValue (more efficient).
        self._instance = DenseInstance(1)
        self._instance.setDataset(self._header)

    def __str__(self):
        return str(self.metrics_dict())

    def get_instances_seen(self):
        return self.instances_seen

    def update(self, y_target_index: int, score: float):
        """Update the evaluator with the ground-truth and the prediction.

        :param y_target_index: The ground-truth class index. This is NOT
            the actual class value, but the index of the class value in the
            schema.
        :param score: The predicted scores. Should be in the range [0, 1].
        """
        if not isinstance(y_target_index, (np.integer, int)):
            raise TypeError(
                f"y_target_index must be an integer, not {type(y_target_index)}"
            )

        # Notice, in MOA the class value is an index, not the actual value
        # (e.g. not "one" but 0 assuming labels=["one", "two"])
        self._instance.setClassValue(y_target_index)
        example = InstanceExample(self._instance)

        self.moa_evaluator.addResult(example, [1 - score, score])

        self.instances_seen += 1

        # If the window_size is set, then check if it should record the intermediary results.
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            performance_values = self.metrics()
            self.result_windows.append(performance_values)

    def metrics_header(self):
        performance_measurements = self.moa_evaluator.getPerformanceMeasurements()
        return _metric_names(m.getName() for m in performance_measurements)

    def metrics(self):
        return [
            measurement.getValue()
            for measurement in self.moa_evaluator.getPerformanceMeasurements()
        ]

    def metrics_dict(self):
        return {
            header: value
            for header, value in zip(self.metrics_header(), self.metrics())
        }

    def metrics_per_window(self):
        return pd.DataFrame(self.result_windows, columns=self.metrics_header())

    def auc(self):
        index = self.metrics_header().index("auc")
        return self.metrics()[index]

    def s_auc(self):
        index = self.metrics_header().index("s_auc")
        return self.metrics()[index]
