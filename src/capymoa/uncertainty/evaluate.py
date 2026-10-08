"""Evaluators for prediction intervals, the engine behind
:func:`evaluate_prediction_interval`.

The evaluators wrap MOA's evaluators. Use them directly in a custom loop.
"""

import warnings

import pandas as pd
from com.yahoo.labs.samoa.instances import Attribute, DenseInstance, Instances
from java.util import ArrayList
from moa.core import InstanceExample
from moa.evaluation import (
    BasicPredictionIntervalEvaluator,
    WindowPredictionIntervalEvaluator,
    WindowRegressionPerformanceEvaluator,
)

from capymoa._utils import _translate_metric_name
from capymoa.regressor.evaluate import RegressionEvaluator


class PredictionIntervalEvaluator(RegressionEvaluator):
    """Scores a prediction interval and its point prediction.

    A prediction is ``[lower, point, upper]``. ``coverage``, ``average_length``
    and ``nmpiw`` score the interval. The regression metrics (``mae``,
    ``rmse`` ...) score the point.
    """

    def __init__(
        self,
        schema=None,
        window_size=None,
        moa_evaluator=None,
        moa_point_evaluator=None,
    ):
        """
        :param schema: The schema of the stream.
        :param window_size: Keep the metrics every ``window_size`` instances in
            :meth:`metrics_per_window`. ``None`` to keep none.
        :param moa_evaluator: MOA interval evaluator. Defaults to a
            ``BasicPredictionIntervalEvaluator``.
        :param moa_point_evaluator: MOA regression evaluator for the point.
            Defaults to a ``BasicRegressionPerformanceEvaluator``.
        """
        self.instances_seen = 0
        self.result_windows = []
        self.window_size = window_size

        self.moa_basic_evaluator = moa_evaluator
        if self.moa_basic_evaluator is None:
            self.moa_basic_evaluator = BasicPredictionIntervalEvaluator()

        # self.moa_basic_evaluator.prepareForUse()

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
                # print(self._header)
            else:
                raise ValueError("Schema was not set for a regression task")
        else:
            raise ValueError("Schema is None, please define a proper Schema.")

        # Prediction Interval has three outputs: lower bound, prediction, upper bound
        self.pred_template = [0, 0, 0]

        # Create the denseInstance just once and keep reusing it by changing the classValue (more efficient).
        self._instance = DenseInstance(self.schema.get_num_attributes() + 1)
        self._instance.setDataset(self._header)

        # Scores the point prediction. No ``window_size``, so it keeps no
        # windows of its own.
        self._point = RegressionEvaluator(schema, moa_evaluator=moa_point_evaluator)

    def update(self, y, y_pred):
        if y is None:
            raise ValueError(f"Invalid ground-truth y = {y}")

        self._instance.setClassValue(y)
        example = InstanceExample(self._instance)

        # if y_pred is None, it indicates the learner did not produce a prediction for this instace
        if y_pred is None:
            # if the y_pred is None, give a warning and then assign y_pred with an all zero prediction array
            warnings.warn(
                "The learner did not produce a prediction interval for this instance"
            )
            y_pred = [0, 0, 0]

        if len(y_pred) != len(self.pred_template):
            warnings.warn(
                "The learner did not produce a valid prediction interval for this instance"
            )

        for i in range(len(y_pred)):
            self.pred_template[i] = y_pred[i]

        self.moa_basic_evaluator.addResult(example, self.pred_template)
        self._point.update(y, self.pred_template[1])
        self.instances_seen += 1

        # If the window_size is set, then check if it should record the intermediary results.
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            self.result_windows.append(self.metrics())

    def _interval_metrics(self) -> dict:
        return {
            _translate_metric_name("".join(m.getName()), to="capymoa"): m.getValue()
            for m in self.moa_basic_evaluator.getPerformanceMeasurements()
        }

    def metrics_dict(self):
        # MOA's regression metrics are wrong, so the point ones replace them.
        return {**self._interval_metrics(), **self._point.metrics_dict()}

    def metrics_header(self):
        return list(self.metrics_dict())

    def metrics(self):
        return list(self.metrics_dict().values())

    def metrics_per_window(self):
        return pd.DataFrame(self.result_windows, columns=self.metrics_header())

    def coverage(self):
        index = self.metrics_header().index("coverage")
        return self.metrics()[index]

    def average_length(self):
        index = self.metrics_header().index("average_length")
        return self.metrics()[index]

    def nmpiw(self):
        index = self.metrics_header().index("nmpiw")
        return self.metrics()[index]


class PredictionIntervalWindowedEvaluator(PredictionIntervalEvaluator):
    def __init__(self, schema=None, window_size=1000):
        self.moa_evaluator = WindowPredictionIntervalEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)
        moa_point_evaluator = WindowRegressionPerformanceEvaluator()
        moa_point_evaluator.widthOption.setValue(window_size)

        super().__init__(
            schema=schema,
            window_size=window_size,
            moa_evaluator=self.moa_evaluator,
            moa_point_evaluator=moa_point_evaluator,
        )

    def coverage(self):
        return self.metrics_per_window()["coverage"].tolist()

    def average_length(self):
        return self.metrics_per_window()["average_length"].tolist()

    def nmpiw(self):
        return self.metrics_per_window()["nmpiw"].tolist()
