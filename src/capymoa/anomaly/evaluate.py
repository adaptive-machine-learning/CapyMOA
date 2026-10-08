"""Evaluators for anomaly detection, the engine behind :func:`evaluate_anomaly`.

The evaluators wrap MOA's evaluators. Use them directly in a custom loop.
"""

import numpy as np
import pandas as pd
from com.yahoo.labs.samoa.instances import Attribute, DenseInstance, Instances
from java.util import ArrayList
from moa.core import InstanceExample
from moa.evaluation import (
    BasicAUCImbalancedPerformanceEvaluator,
    WindowAUCImbalancedPerformanceEvaluator,
)

from capymoa._utils import _translate_metric_name
from capymoa.stream import Schema


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
        performance_names = [
            _translate_metric_name("".join(measurement.getName()), to="capymoa")
            for measurement in performance_measurements
        ]
        return performance_names

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
        performance_names = [
            _translate_metric_name("".join(measurement.getName()), to="capymoa")
            for measurement in performance_measurements
        ]
        return performance_names

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
