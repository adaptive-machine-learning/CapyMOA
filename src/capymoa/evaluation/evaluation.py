from typing import Optional

import pandas as pd
import numpy as np
import time
import warnings
import json
import csv
import os

from capymoa.stream import Schema, Stream
from capymoa.base import (
    ClassifierSSL,
    MOAPredictionIntervalLearner
)

from capymoa.evaluation.results import PrequentialResults
from capymoa._utils import _translate_metric_name

from com.yahoo.labs.samoa.instances import Instances, Attribute, DenseInstance
from moa.core import InstanceExample
from moa.evaluation import (
    BasicClassificationPerformanceEvaluator,
    WindowClassificationPerformanceEvaluator,
    BasicRegressionPerformanceEvaluator,
    WindowRegressionPerformanceEvaluator,
    BasicPredictionIntervalEvaluator,
    WindowPredictionIntervalEvaluator,
    BasicAUCImbalancedPerformanceEvaluator,
)

from java.util import ArrayList
from moa.evaluation import EfficientEvaluationLoops
from moa.streams import InstanceStream


def _is_fast_mode_compilable(stream: Stream, learner, optimise=True) -> bool:

    # refuse prediction interval learner
    if not hasattr(learner, "moa_learner") or isinstance(learner.moa_learner, MOAPredictionIntervalLearner):
        return False

    """Check if the stream is compatible with the efficient loops in MOA."""
    is_moa_stream = stream.moa_stream is not None and isinstance(
        stream.moa_stream, InstanceStream
    )
    is_moa_learner = hasattr(learner, "moa_learner") and learner.moa_learner is not None

    return is_moa_stream and is_moa_learner and optimise


class ClassificationEvaluator:
    """
    Wrapper for the Classification Performance Evaluator from MOA. By default, it uses the
    BasicClassificationPerformanceEvaluator
    """

    def __init__(
            self,
            schema: Schema = None,
            window_size=None,
            allow_abstaining=True,
            moa_evaluator=None,
    ):
        self.instances_seen = 0
        self.result_windows = []
        self.window_size = window_size

        self.allow_abstaining = allow_abstaining

        self.moa_basic_evaluator = moa_evaluator
        if self.moa_basic_evaluator is None:
            self.moa_basic_evaluator = BasicClassificationPerformanceEvaluator()

        self.moa_basic_evaluator.recallPerClassOption.set()
        self.moa_basic_evaluator.precisionPerClassOption.set()
        self.moa_basic_evaluator.precisionRecallOutputOption.set()
        self.moa_basic_evaluator.f1PerClassOption.set()
        self.moa_basic_evaluator.prepareForUse()

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

        self.pred_template = [0] * len(self.schema.get_label_indexes())

        # Create the denseInstance just once and keep reusing it by changing the classValue (more efficient).
        self._instance = DenseInstance(1)
        self._instance.setDataset(self._header)

    def __str__(self):
        return str(self.metrics_dict())

    def get_instances_seen(self):
        return self.instances_seen

    def update(self, y_target_index: int, y_pred_index: Optional[int]):
        """Update the evaluator with the ground-truth and the prediction.

        :param y_target_index: The ground-truth class index. This is NOT
            the actual class value, but the index of the class value in the
            schema.
        :param y_pred_index: The predicted class index. If the classifier
            abstains from making a prediction, this value can be None.
        :raises ValueError: If the values are not valid indexes in the schema.
        """
        if not isinstance(y_target_index, (np.integer, int)):
            raise ValueError(
                f"y_target_index must be an integer, not {type(y_target_index)}"
            )
        if not (y_pred_index is None or isinstance(y_pred_index, (np.integer, int))):
            raise ValueError(
                f"y_pred_index must be an integer, not {type(y_pred_index)}"
            )

        # If the prediction is invalid, it could mean the classifier is abstaining from making a prediction;
        # thus, it is allowed to continue (unless parameterized differently).
        if y_pred_index is not None and not self.schema.is_y_index_in_range(
                y_pred_index
        ):
            if self.allow_abstaining:
                y_pred_index = None
            else:
                raise ValueError(f"Invalid prediction y_pred_index = {y_pred_index}")

        # Notice, in MOA the class value is an index, not the actual value
        # (e.g. not "one" but 0 assuming labels=["one", "two"])
        self._instance.setClassValue(y_target_index)
        example = InstanceExample(self._instance)

        # Shallow copy of the pred_template
        # MOA evaluator accepts the result of getVotesForInstance which is similar to a predict_proba
        #    (may or may not be normalised, but for our purposes it doesn't matter)
        prediction_array = self.pred_template[:]

        # if y_pred is None, it indicates the learner did not produce a prediction for this instance,
        # count as an error
        if y_pred_index is None:
            # TODO: Modify this once the option to abstain from predictions is implemented. Currently, by default it
            #  sets the prediction to the first class (index zero), which is consistent with MOA.
            y_pred_index = 0
            # Set y_pred_index to any valid prediction that is not y (force an incorrect prediction)
            # This does not affect recall or any other metrics, because the selected value is always
            # incorrect.

            # Create an intermediary array with indices excluding the y
            # indexesWithoutY = [
            #     i for i in range(len(self.schema.get_label_indexes())) if i != y_target_index
            # ]
            # random_y_pred = random.choice(indexesWithoutY)
            # y_pred_index = self.schema.get_label_indexes()[random_y_pred]

        prediction_array[int(y_pred_index)] += 1
        self.moa_basic_evaluator.addResult(example, prediction_array)

        self.instances_seen += 1

        # If the window_size is set, then check if it should record the intermediary results.
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            performance_values = self.metrics()
            self.result_windows.append(performance_values)

    def metrics_header(self):
        performance_measurements = self.moa_basic_evaluator.getPerformanceMeasurements()
        performance_names = [
            _translate_metric_name("".join(measurement.getName()), to='capymoa')
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

    def __getitem__(self, key):
        if hasattr(self, key):
            attr = getattr(self, key)
            return attr()
        return self.__getattr__(key)()

    # This allows access to metrics that are generated dynamically like recall_0, f1_score_3, ...
    def __getattr__(self, metric):
        if metric in self.metrics_header():
            index = self.metrics_header().index(metric)

            def metric_value():
                return float(self.metrics()[index])

            return metric_value
        return None

    def accuracy(self):
        index = self.metrics_header().index("accuracy")
        return float(self.metrics()[index])

    def kappa(self):
        index = self.metrics_header().index("kappa")
        return float(self.metrics()[index])

    def kappa_t(self):
        index = self.metrics_header().index("kappa_t")
        return float(self.metrics()[index])

    def kappa_m(self):
        index = self.metrics_header().index("kappa_m")
        return float(self.metrics()[index])

    def f1_score(self):
        index = self.metrics_header().index("f1_score")
        return float(self.metrics()[index])

    def precision(self):
        index = self.metrics_header().index("precision")
        return float(self.metrics()[index])

    def recall(self):
        index = self.metrics_header().index("recall")
        return float(self.metrics()[index])


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

    def update(self, y, y_pred: Optional[float]):
        if y is None:
            raise ValueError(f"Invalid ground-truth y = {y}")

        self._instance.setClassValue(y)
        example = InstanceExample(self._instance)

        # The learner did not produce a prediction for this instance, thus y_pred is None
        if y_pred is None:
            warnings.warn("The learner did not produce a prediction for this instance")
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
            _translate_metric_name("".join(measurement.getName()), to='capymoa')
            for measurement in performance_measurements
        ]
        return performance_names

    def metrics(self):
        return [
            measurement.getValue()
            for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()
        ]

    def metrics_dict(self):
        return {header: value for header, value in zip(self.metrics_header(), self.metrics())}

    def metrics_per_window(self):
        return pd.DataFrame(self.result_windows, columns=self.metrics_header()).copy()

    def predictions(self):
        return self.predictions

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
            raise ValueError(
                f"y_target_index must be an integer, not {type(y_target_index)}"
            )

        # Notice, in MOA the class value is an index, not the actual value
        # (e.g. not "one" but 0 assuming labels=["one", "two"])
        self._instance.setClassValue(y_target_index)
        example = InstanceExample(self._instance)

        self.moa_basic_evaluator.addResult(example, [score, 1-score])

        self.instances_seen += 1

        # If the window_size is set, then check if it should record the intermediary results.
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            performance_values = self.metrics()
            self.result_windows.append(performance_values)

    def metrics_header(self):
        performance_measurements = self.moa_basic_evaluator.getPerformanceMeasurements()
        performance_names = [
            _translate_metric_name("".join(measurement.getName()), to='capymoa')
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


class ClassificationWindowedEvaluator(ClassificationEvaluator):
    """
    Uses the ClassificationEvaluator to perform a windowed evaluation.

    IMPORTANT: The results for the last window are not always available through ```metrics()```, if the window_size does
    not perfectly divide the stream, the metrics corresponding to the last remaining instances in the last window can
    be obtained by invoking ```metrics()```
    """

    def __init__(
            self,
            schema=None,
            window_size=1000
    ):
        self.moa_evaluator = WindowClassificationPerformanceEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)

        super().__init__(
            schema=schema,
            window_size=window_size,
            moa_evaluator=self.moa_evaluator,
        )

    # This allows access to metrics that are generated dynamically like recall_0, f1_score_3, ...
    def __getattr__(self, metric):
        if metric in self.metrics_header():

            def metric_value():
                return self.metrics_per_window()[metric].tolist()

            return metric_value
        return None

    def accuracy(self):
        return self.metrics_per_window()['accuracy'].tolist()

    def kappa(self):
        return self.metrics_per_window()['kappa'].tolist()

    def kappa_t(self):
        return self.metrics_per_window()["kappa_t"].tolist()

    def kappa_m(self):
        return self.metrics_per_window()["kappa_m"].tolist()

    def f1_score(self):
        return self.metrics_per_window()["f1_score"].tolist()

    def precision(self):
        return self.metrics_per_window()["precision"].tolist()

    def recall(self):
        return self.metrics_per_window()["recall"].tolist()


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
        return self.metrics_per_window()['mae'].tolist()

    def rmse(self):
        return self.metrics_per_window()['rmse'].tolist()

    def rmae(self):
        return self.metrics_per_window()['rmae'].tolist()

    def r2(self):
        return self.metrics_per_window()['r2'].tolist()

    def adjusted_r2(self):
        return self.metrics_per_window()['adjusted_r2'].tolist()


class PredictionIntervalEvaluator(RegressionEvaluator):
    def __init__(self, schema=None, window_size=None, moa_evaluator=None):
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

    def update(self, y, y_pred):
        if y is None:
            raise ValueError(f"Invalid ground-truth y = {y}")

        self._instance.setClassValue(y)
        example = InstanceExample(self._instance)

        # if y_pred is None, it indicates the learner did not produce a prediction for this instace
        if y_pred is None:
            # if the y_pred is None, give a warning and then assign y_pred with an all zero prediction array
            warnings.warn("The learner did not produce a prediction interval for this instance")
            y_pred = [0, 0, 0]

        if len(y_pred) != len(self.pred_template):
            warnings.warn("The learner did not produce a valid prediction interval for this instance")

        for i in range(len(y_pred)):
            self.pred_template[i] = y_pred[i]

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
            _translate_metric_name("".join(measurement.getName()), to='capymoa')
            for measurement in performance_measurements
        ]
        return performance_names

    def metrics(self):
        return [
            measurement.getValue()
            for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()
        ]

    def metrics_per_window(self):
        return pd.DataFrame(self.result_windows, columns=self.metrics_header())

    def coverage(self):
        index = self.metrics_header().index("coverage")
        return self.metrics()[index]

    def average_length(self):
        index = self.metrics_header().index("average_length")
        return self.metrics()[index]

    def NMPIW(self):
        index = self.metrics_header().index("nmpiw")
        return self.metrics()[index]


class PredictionIntervalWindowedEvaluator(PredictionIntervalEvaluator):
    def __init__(self, schema=None, window_size=1000):
        self.moa_evaluator = WindowPredictionIntervalEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)

        super().__init__(
            schema=schema, window_size=window_size, moa_evaluator=self.moa_evaluator
        )

    def coverage(self):
        return self.metrics_per_window()['coverage'].tolist()

    def average_length(self):
        return self.metrics_per_window()['average length'].tolist()

    def NMPIW(self):
        return self.metrics_per_window()['NMPIW'].tolist()


def start_time_measuring():
    start_wallclock_time = time.time()
    start_cpu_time = time.process_time()

    return start_wallclock_time, start_cpu_time


def stop_time_measuring(start_wallclock_time, start_cpu_time):
    # Stop measuring time
    end_wallclock_time = time.time()
    end_cpu_time = time.process_time()

    # Calculate and print the elapsed time and CPU times
    elapsed_wallclock_time = end_wallclock_time - start_wallclock_time
    elapsed_cpu_time = end_cpu_time - start_cpu_time

    return elapsed_wallclock_time, elapsed_cpu_time


def prequential_evaluation(
        stream, learner, max_instances=None, window_size=1000, store_predictions=False, store_y=False, optimise=True
):
    """
    Calculates the metrics cumulatively (i.e. test-then-train) and in a window-fashion (i.e. windowed prequential
    evaluation). Returns both evaluators so that the user has access to metrics from both evaluators.
    """
    stream.restart()
    if _is_fast_mode_compilable(stream, learner, optimise):
        return _prequential_evaluation_fast(stream, learner,
                                            max_instances,
                                            window_size,
                                            store_y=store_y,
                                            store_predictions=store_predictions)

    predictions = None
    if store_predictions:
        predictions = []

    ground_truth_y = None
    if store_y:
        ground_truth_y = []

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    instancesProcessed = 1

    evaluator_cumulative = None
    evaluator_windowed = None
    if stream.get_schema().is_classification():
        evaluator_cumulative = ClassificationEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )
        if window_size is not None:
            evaluator_windowed = ClassificationWindowedEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
    else:
        if not isinstance(learner, MOAPredictionIntervalLearner):
            evaluator_cumulative = RegressionEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
            if window_size is not None:
                evaluator_windowed = RegressionWindowedEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
        else:
            evaluator_cumulative = PredictionIntervalEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
            if window_size is not None:
                evaluator_windowed = PredictionIntervalWindowedEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
    while stream.has_more_instances() and (
            max_instances is None or instancesProcessed <= max_instances
    ):
        instance = stream.next_instance()

        prediction = learner.predict(instance)

        if stream.get_schema().is_classification():
            y = instance.y_index
        else:
            y = instance.y_value

        evaluator_cumulative.update(y, prediction)
        if window_size is not None:
            evaluator_windowed.update(y, prediction)
        learner.train(instance)

        # Storing predictions if store_predictions was set to True during initialisation
        if predictions is not None:
            predictions.append(prediction)

        # Storing ground-truth if store_y was set to True during initialisation
        if ground_truth_y is not None:
            ground_truth_y.append(y)

        instancesProcessed += 1

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    # Add the results corresponding to the remainder of the stream in case the number of processed
    # instances is not perfectly divisible by the window_size (if it was, then it is already be in
    # the result_windows variable). The evaluator_windowed will be None if the window_size is None.
    if (
            evaluator_windowed is not None
            and evaluator_windowed.get_instances_seen() % window_size != 0
    ):
        evaluator_windowed.result_windows.append(evaluator_windowed.metrics())

    results = PrequentialResults(learner=str(learner),
                                 stream=stream,
                                 wallclock=elapsed_wallclock_time,
                                 cpu_time=elapsed_cpu_time,
                                 max_instances=max_instances,
                                 cumulative_evaluator=evaluator_cumulative,
                                 windowed_evaluator=evaluator_windowed,
                                 ground_truth_y=ground_truth_y,
                                 predictions=predictions)

    return results


# TODO: Include store_predictions and store_y logic
def prequential_ssl_evaluation(
        stream,
        learner,
        max_instances=None,
        window_size=1000,
        initial_window_size=0,
        delay_length=0,
        label_probability=0.01,
        random_seed=1,
        store_predictions=False,
        store_y=False,
        optimise=True,
):
    """
    If the learner is not an SSL learner, then it will be trained only on the labeled instances.
    """

    stream.restart()

    if _is_fast_mode_compilable(stream, learner, optimise):
        return _prequential_ssl_evaluation_fast(stream,
                                                learner,
                                                max_instances,
                                                window_size,
                                                initial_window_size,
                                                delay_length,
                                                label_probability,
                                                random_seed)

    # IMPORTANT: delay_length and initial_window_size have not been implemented in python yet
    # In MOA it is implemented so _prequential_ssl_evaluation_fast works just fine.
    if initial_window_size != 0:
        raise ValueError(
            "Initial window size must be 0 for this function as the feature is not implemented yet."
        )

    if delay_length != 0:
        raise ValueError(
            "Delay length must be 0 for this function as the feature is not implemented yet."
        )

    # Reset the random state
    mt19937 = np.random.MT19937()
    mt19937._legacy_seeding(random_seed)
    rand = np.random.Generator(mt19937)

    predictions = None
    if store_predictions:
        predictions = []

    ground_truth_y = None
    if store_y:
        ground_truth_y = []

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    instancesProcessed = 1

    evaluator_cumulative = None
    evaluator_windowed = None
    if stream.get_schema().is_classification():
        evaluator_cumulative = ClassificationEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )
        # If the window_size is None, then should not initialise or produce prequential (window) results.
        if window_size is not None:
            evaluator_windowed = ClassificationWindowedEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
    else:
        raise ValueError("The learning task is not classification")

    unlabeled_counter = 0

    while stream.has_more_instances() and (
            max_instances is None or instancesProcessed <= max_instances
    ):
        instance = stream.next_instance()

        prediction = learner.predict(instance)

        if stream.get_schema().is_classification():
            y = instance.y_index
        else:
            y = instance.y_value

        evaluator_cumulative.update(instance.y_index, prediction)
        if evaluator_windowed is not None:
            evaluator_windowed.update(instance.y_index, prediction)

        if rand.random(dtype=np.float64) >= label_probability:
            # if 0.00 >= label_probability:
            # Do not label the instance
            if isinstance(learner, ClassifierSSL):
                learner.train_on_unlabeled(instance)
                # Otherwise, just ignore the unlabeled instance
            unlabeled_counter += 1
        else:
            # Labeled instance
            learner.train(instance)

        # Storing predictions if store_predictions was set to True during initialisation
        if predictions is not None:
            predictions.append(prediction)

        # Storing ground-truth if store_y was set to True during initialisation
        if ground_truth_y is not None:
            ground_truth_y.append(y)

        instancesProcessed += 1

    # # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    # Add the results corresponding to the remainder of the stream in case the number of processed instances is not
    # perfectly divisible by the window_size (if it was, then it is already in the result_windows variable).
    if (
            evaluator_windowed is not None
            and evaluator_windowed.get_instances_seen() % window_size != 0
    ):
        evaluator_windowed.result_windows.append(evaluator_windowed.metrics())

    results = PrequentialResults(learner=str(learner),
                                 stream=stream,
                                 wallclock=elapsed_wallclock_time,
                                 cpu_time=elapsed_cpu_time,
                                 max_instances=max_instances,
                                 cumulative_evaluator=evaluator_cumulative,
                                 windowed_evaluator=evaluator_windowed,
                                 ground_truth_y=ground_truth_y,
                                 predictions=predictions,
                                 other_metrics={"unlabeled": unlabeled_counter,
                                                "unlabeled_ratio": unlabeled_counter / instancesProcessed})

    return results

##############################################################
###### OPTIMISED VERSIONS (use experimental MOA method) ######
##############################################################


def _prequential_evaluation_fast(stream, learner,
                                 max_instances=None,
                                 window_size=1000,
                                 store_y=False,
                                 store_predictions=False):
    """
    Prequential evaluation fast. This function should not be used directly, users should use prequential_evaluation.
    """

    predictions = None
    if store_predictions:
        predictions = []

    ground_truth_y = None
    if store_y:
        ground_truth_y = []

    if not _is_fast_mode_compilable(stream, learner):
        raise ValueError(
            "`prequential_evaluation_fast` requires the stream object to have a`Stream.moa_stream`"
        )

    if max_instances is None:
        max_instances = -1

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    basic_evaluator = None
    windowed_evaluator = None
    if stream.get_schema().is_classification():
        basic_evaluator = ClassificationEvaluator(schema=stream.get_schema())
        windowed_evaluator = ClassificationWindowedEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )
    else:
        # If it is not classification, could be regression or prediction interval
        if not isinstance(learner, MOAPredictionIntervalLearner):
            basic_evaluator = RegressionEvaluator(schema=stream.get_schema())
            windowed_evaluator = RegressionWindowedEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
        else:
            basic_evaluator = PredictionIntervalEvaluator(schema=stream.get_schema())
            windowed_evaluator = PredictionIntervalWindowedEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
    moa_results = EfficientEvaluationLoops.PrequentialEvaluation(
        stream.moa_stream,
        learner.moa_learner,
        basic_evaluator.moa_basic_evaluator,
        windowed_evaluator.moa_evaluator,
        max_instances,
        window_size,
        store_y,
        store_predictions,
    )

    # Reset the windowed_evaluator result_windows
    if moa_results is not None:
        windowed_evaluator.result_windows = []
        if moa_results.windowedResults is not None:
            for entry_idx in range(len(moa_results.windowedResults)):
                windowed_evaluator.result_windows.append(
                    moa_results.windowedResults[entry_idx]
                )

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    if store_y or store_predictions:
        for i in range(len(moa_results.targets if len(moa_results.targets) != 0 else moa_results.predictions)):
            if store_y:
                ground_truth_y.append(moa_results.targets[i])
            if store_predictions:
                predictions.append(moa_results.predictions[i])

    results = PrequentialResults(learner=str(learner),
                                 stream=stream,
                                 wallclock=elapsed_wallclock_time,
                                 cpu_time=elapsed_cpu_time,
                                 max_instances=max_instances,
                                 cumulative_evaluator=basic_evaluator,
                                 windowed_evaluator=windowed_evaluator,
                                 ground_truth_y=ground_truth_y,
                                 predictions=predictions)

    return results


def _prequential_ssl_evaluation_fast(
        stream,
        learner,
        max_instances=None,
        window_size=1000,
        initial_window_size=0,
        delay_length=0,
        label_probability=0.01,
        random_seed=1,
        store_y=False,
        store_predictions=False
):
    """
    Prequential SSL evaluation fast.
    """
    if not _is_fast_mode_compilable(stream, learner):
        raise ValueError(
            "`prequential_evaluation_fast` requires the stream object to have a`Stream.moa_stream`"
        )

    predictions = None
    if store_predictions:
        predictions = []

    ground_truth_y = None
    if store_y:
        ground_truth_y = []

    if max_instances is None:
        max_instances = -1

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    basic_evaluator = ClassificationEvaluator(schema=stream.get_schema())
    # Always create the windowed_evaluator, even if window_size is None.
    # TODO: may want to avoid creating it if window_size is None.
    windowed_evaluator = ClassificationWindowedEvaluator(
        schema=stream.get_schema(), window_size=window_size
    )

    # TODO: requires update to MOA to include store_y and store_predictions
    moa_results = EfficientEvaluationLoops.PrequentialSSLEvaluation(
        stream.moa_stream,
        learner.moa_learner,
        basic_evaluator.moa_basic_evaluator,
        windowed_evaluator.moa_evaluator,
        max_instances,
        window_size,
        initial_window_size,
        delay_length,
        label_probability,
        random_seed,
        True,
        # store_y,
        # store_predictions,
    )

    # Reset the windowed_evaluator result_windows
    if moa_results is not None:
        windowed_evaluator.result_windows = []
        if moa_results.windowedResults is not None:
            for entry_idx in range(len(moa_results.windowedResults)):
                windowed_evaluator.result_windows.append(
                    moa_results.windowedResults[entry_idx]
                )

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    if store_y or store_predictions:
        for i in range(len(moa_results.targets if len(moa_results.targets) != 0 else moa_results.predictions)):
            if store_y:
                ground_truth_y.append(moa_results.targets[i])
            if store_predictions:
                predictions.append(moa_results.predictions[i])

    results = PrequentialResults(learner=str(learner),
                                 stream=stream,
                                 wallclock=elapsed_wallclock_time,
                                 cpu_time=elapsed_cpu_time,
                                 max_instances=max_instances,
                                 cumulative_evaluator=basic_evaluator,
                                 windowed_evaluator=windowed_evaluator,
                                 ground_truth_y=ground_truth_y,
                                 predictions=predictions,
                                 other_metrics=dict(moa_results.otherMeasurements))

    return results


def prequential_evaluation_multiple_learners(
        stream, learners, max_instances=None, window_size=1000
):
    """
    Calculates the metrics cumulatively (i.e., test-then-train) and in a windowed-fashion for multiple streams and
    learners. It behaves as if we invoked prequential_evaluation() multiple times, but we only iterate through the
    stream once.
    This function is useful in situations where iterating through the stream is costly, but we still want to assess
    several learners on it.
    Returns the results in a dictionary format. Infers whether it is a Classification or Regression problem based on the
    stream schema.
    """
    results = {}

    stream.restart()

    for learner_name, learner in learners.items():
        results[learner_name] = {"learner": str(learner)}

    for learner_name, learner in learners.items():
        if stream.get_schema().is_classification():
            results[learner_name]["cumulative"] = ClassificationEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
            if window_size is not None:
                results[learner_name]["windowed"] = ClassificationWindowedEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
        else:
            if not isinstance(learner, MOAPredictionIntervalLearner):
                results[learner_name]["cumulative"] = RegressionEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
                if window_size is not None:
                    results[learner_name]["windowed"] = RegressionWindowedEvaluator(
                        schema=stream.get_schema(), window_size=window_size
                    )
            else:
                results[learner_name]["cumulative"] = PredictionIntervalEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
                if window_size is not None:
                    results[learner_name]["windowed"] = PredictionIntervalWindowedEvaluator(
                        schema=stream.get_schema(), window_size=window_size
                    )
        results[learner_name]['learner'] = learner_name
    instancesProcessed = 1

    while stream.has_more_instances() and (
            max_instances is None or instancesProcessed <= max_instances
    ):
        instance = stream.next_instance()

        for learner_name, learner in learners.items():
            # Predict for the current learner
            prediction = learner.predict(instance)

            if stream.get_schema().is_classification():
                y = instance.y_index
            else:
                y = instance.y_value

            results[learner_name]["cumulative"].update(y, prediction)
            if window_size is not None:
                results[learner_name]["windowed"].update(y, prediction)

            learner.train(instance)

        instancesProcessed += 1

    # Iterate through the results of each learner and add (if needed) the last window of results to it.
    if window_size is not None:
        for learner_name, result in results.items():
            if result["windowed"].get_instances_seen() % window_size != 0:
                result["windowed"].result_windows.append(result["windowed"].metrics())

    results['stream'] = stream
    results['max_instances'] = max_instances

    return results


def write_results_to_files(
        path: str = None,
        results=None,
        file_name: str = None,
        directory_name: str = None
):
    if results is None:
        raise ValueError('The results object is None')

    path = path if path.endswith('/') else (path + '/')

    if isinstance(results, ClassificationWindowedEvaluator) or isinstance(results, RegressionWindowedEvaluator):
        data = results.metrics_per_window()
        data.to_csv(('./' if path is None else path) + ('/windowed_results.csv' if file_name is None else file_name),
                    index=False)
    elif isinstance(results, ClassificationEvaluator) or isinstance(results, RegressionEvaluator):
        json_str = json.dumps(results.metrics_dict())
        data = json.loads(json_str)
        with open(('./' if path is None else path) + ('/cumulative_results.csv' if file_name is None else file_name),
                  'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data.keys())
            writer.writerow(data.values())
    elif isinstance(results, PrequentialResults):
        directory_name = 'prequential_results' if directory_name is None else directory_name
        if os.path.exists(path + '/' + directory_name):
            raise ValueError(f'Directory {directory_name} already exists, please use another name')
        else:
            os.makedirs(path + '/' + directory_name)

        write_results_to_files(path=path + '/' + directory_name, file_name=file_name, results=results.cumulative)
        write_results_to_files(path=path + '/' + directory_name, file_name=file_name, results=results.windowed)

        # If the ground truth and predictions are available, they will be writen to a file
        if results.get_ground_truth_y() is not None and results.get_predictions() is not None:
            y_vs_predictions = {'ground_truth_y': results.get_ground_truth_y(),
                                'predictions': results.get_predictions()}
            if len(y_vs_predictions) > 0:
                t_p = pd.DataFrame(y_vs_predictions)
                t_p.to_csv(('./' if path is None else path) + '/' + directory_name +
                           '/ground_truth_y_and_predictions.csv',
                           index=False)
    else:
        raise ValueError('Writing results to file is not supported for type ' + str(type(results)) + ' yet')


