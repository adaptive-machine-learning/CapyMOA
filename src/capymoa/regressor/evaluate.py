"""Evaluators and result types for regression.

The result types are the output of
:func:`~capymoa.regressor.evaluate_regressor`. The evaluators wrap MOA's
evaluators. Use them directly in a custom loop.
"""

from typing import NotRequired

import numpy as np
import pandas as pd
from com.yahoo.labs.samoa.instances import Attribute, DenseInstance, Instances
from java.util import ArrayList
from moa.core import InstanceExample
from moa.evaluation import (
    BasicRegressionPerformanceEvaluator,
    WindowRegressionPerformanceEvaluator,
)

from capymoa._utils import _translate_metric_name
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
    """Results of evaluating a regressor. See :func:`~capymoa.regressor.evaluate_regressor`.

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
    #: The metrics of each window (see
    #: :class:`~capymoa.regressor.evaluate.RegressorWindows`). Absent if ``window_size``
    #: is absent.
    windowed: NotRequired[RegressorWindows]


class RegressionEvaluator:
    """
    Wrapper for the Regression Performance Evaluator from MOA.
    By default, it uses the MOA BasicRegressionPerformanceEvaluator as moa_evaluator.
    """

    def __init__(self, schema=None, window_size=None, moa_evaluator=None):
        self.instances_seen = 0
        self.result_windows = []
        self.window_size = window_size

        self.moa_basic_evaluator = moa_evaluator
        if self.moa_basic_evaluator is None:
            self.moa_basic_evaluator = BasicRegressionPerformanceEvaluator()

        _attributeValues = ArrayList()

        self.schema = schema
        self._header = None
        if self.schema is not None:
            if self.schema.is_regression():
                attSub = ArrayList()
                for _ in range(self.schema.get_num_attributes()):
                    attSub.append(Attribute("Attribute"))
                _targetAttribute = Attribute("Target")

                attSub.append(_targetAttribute)
                self._header = Instances("", attSub, 1)
                self._header.setClassIndex(self.schema.get_num_attributes())
            else:
                raise ValueError("Schema was not set for a regression task")
        else:
            raise ValueError("Schema is None, please define a proper Schema.")

        # Regression has only one output
        self.pred_template = [0]

        # Create the denseInstance just once and keep reusing it by changing the classValue (more efficient).
        self._instance = DenseInstance(self.schema.get_num_attributes() + 1)
        self._instance.setDataset(self._header)

    def __str__(self):
        return str(self.metrics_dict())

    def get_instances_seen(self):
        return self.instances_seen

    def update(self, y, y_pred: float | None):
        if y is None:
            raise ValueError(f"Invalid ground-truth y = {y}")

        self._instance.setClassValue(y)
        example = InstanceExample(self._instance)

        # The learner did not produce a prediction for this instance, thus y_pred is None
        if y_pred is None:
            # We used to produce a warning here, but since `None` predictions are common
            # at the beginning of training, most warnings that were not useful.
            y_pred = 0.0

        # Different from classification, there is no need to copy the prediction array, just override the value.
        self.pred_template[0] = y_pred
        self.moa_basic_evaluator.addResult(example, self.pred_template)

        self.instances_seen += 1

        # If the window_size is set, then check if it should record the intermediary results.
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            performance_values = [
                measurement.getValue()
                for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()
            ]
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
        return pd.DataFrame(self.result_windows, columns=self.metrics_header()).copy()

    def ground_truth_y(self):
        return self.gt_y

    def mae(self):
        index = self.metrics_header().index("mae")
        return self.metrics()[index]

    def rmse(self):
        index = self.metrics_header().index("rmse")
        return self.metrics()[index]

    def rmae(self):
        index = self.metrics_header().index("rmae")
        return self.metrics()[index]

    def r2(self):
        index = self.metrics_header().index("r2")
        return self.metrics()[index]

    def adjusted_r2(self):
        index = self.metrics_header().index("adjusted_r2")
        return self.metrics()[index]


class RegressionWindowedEvaluator(RegressionEvaluator):
    """
    Uses the RegressionEvaluator to perform a windowed evaluation.

    IMPORTANT: The results for the last window are always through ```metrics()```, if the window_size does not
    perfectly divide the stream, the metrics corresponding to the last remaining instances in the last window can
    be obtained by invoking ```metrics()```
    """

    def __init__(self, schema=None, window_size=1000):
        self.moa_evaluator = WindowRegressionPerformanceEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)

        super().__init__(
            schema=schema, window_size=window_size, moa_evaluator=self.moa_evaluator
        )

    def mae(self):
        return self.metrics_per_window()["mae"].tolist()

    def rmse(self):
        return self.metrics_per_window()["rmse"].tolist()

    def rmae(self):
        return self.metrics_per_window()["rmae"].tolist()

    def r2(self):
        return self.metrics_per_window()["r2"].tolist()

    def adjusted_r2(self):
        return self.metrics_per_window()["adjusted_r2"].tolist()
