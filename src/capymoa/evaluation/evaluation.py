import csv
import json
import os
import time
import warnings
from itertools import islice
from typing import Optional, Sized, Union

import numpy as np
import pandas as pd
from com.yahoo.labs.samoa.instances import Attribute, DenseInstance, Instances
from java.util import ArrayList
from moa.core import InstanceExample
from moa.evaluation import (
    BasicAUCImbalancedPerformanceEvaluator,
    BasicClassificationPerformanceEvaluator,
    BasicPredictionIntervalEvaluator,
    BasicRegressionPerformanceEvaluator,
    EfficientEvaluationLoops,
    WindowAUCImbalancedPerformanceEvaluator,
    WindowClassificationPerformanceEvaluator,
    WindowPredictionIntervalEvaluator,
    WindowRegressionPerformanceEvaluator,
)
from moa.streams import InstanceStream
from tqdm import tqdm

from capymoa._utils import _translate_metric_name, batched
from capymoa.base import (
    AnomalyDetector,
    BatchClassifier,
    BatchRegressor,
    Classifier,
    ClassifierSSL,
    Clusterer,
    MOAPredictionIntervalLearner,
    Regressor,
)
from capymoa.evaluation._progress_bar import resolve_progress_bar
from capymoa.evaluation.results import PrequentialResults
from capymoa.instance import LabeledInstance, RegressionInstance
from capymoa.stream import Schema, Stream


def _is_fast_mode_compilable(stream: Stream, learner, optimise=True) -> bool:
    # refuse prediction interval learner
    if not hasattr(learner, "moa_learner") or isinstance(
        learner.moa_learner, MOAPredictionIntervalLearner
    ):
        return False

    """Check if the stream is compatible with the efficient loops in MOA."""
    is_moa_stream = isinstance(stream.get_moa_stream(), InstanceStream)
    is_moa_learner = hasattr(learner, "moa_learner") and learner.moa_learner is not None

    return is_moa_stream and is_moa_learner and optimise


def _get_expected_length(
    stream: Stream, max_instances: Optional[int] = None
) -> Optional[int]:
    """Get the expected length of the stream."""
    if isinstance(stream, Sized) and max_instances is not None:
        return min(len(stream), max_instances)
    elif isinstance(stream, Sized) and max_instances is None:
        return len(stream)
    elif max_instances is not None:
        return max_instances
    else:
        return None


def _setup_progress_bar(
    msg: str,
    progress_bar: Union[bool, tqdm],
    stream: Stream,
    learner,
    max_instances: Optional[int],
):
    expected_length = _get_expected_length(stream, max_instances)
    progress_bar = resolve_progress_bar(
        progress_bar,
        f"{msg} {type(learner).__name__!r} on {type(stream).__name__!r}",
    )
    if progress_bar is not None and expected_length is not None:
        progress_bar.set_total(expected_length)
    return progress_bar


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

    def __repr__(self):
        return str(self)

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
            raise ValueError(
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


class ClusteringEvaluator:
    """
    Abstract clustering evaluator for CapyMOA.
    It is slightly different from the other evaluators because it does not have a moa_evaluator object.
    Clustering evaluation at this point is very simple and only uses the unsupervised metrics.
    """

    def __init__(self, update_interval=1000):
        """
        Only the update_interval is set here.
        :param update_interval: The interval at which the evaluator should update the measurements
        """
        self.instances_seen = 0
        self.update_interval = update_interval
        self.measurements = {name: [] for name in self.metrics_header()}
        self.clusterer_name = None

    def __str__(self):
        return str(self.metrics_dict())

    def get_instances_seen(self):
        return self.instances_seen

    def get_update_interval(self):
        return self.update_interval

    def get_clusterer_name(self):
        return self.clusterer_name

    def update(self, clusterer: Clusterer):
        if self.clusterer_name is None:
            self.clusterer_name = str(clusterer)
        self.instances_seen += 1
        if self.instances_seen % self.update_interval == 0:
            self._update_measurements(clusterer)

    def _update_measurements(self, clusterer: Clusterer):
        # update centers, weights, sizes, and radii
        if clusterer.implements_macro_clusters():
            macro = clusterer.get_clustering_result()
            if len(macro.get_centers()) > 0:
                self.measurements["macro"].append(macro)

        if clusterer.implements_micro_clusters():
            micro = clusterer.get_micro_clustering_result()
            if len(micro.get_centers()) > 0:
                self.measurements["micro"].append(micro)

        # calculate silhouette score
        # TODO: delegate silhouette to moa
        # Check how it is done among different clusterers

    def metrics_header(self):
        performance_names = ["macro", "micro"]
        return performance_names

    def metrics(self):
        # using the static list to keep the order of the metrics
        return [self.measurements[key] for key in self.metrics_header()]

    def get_measurements(self):
        return self.measurements


class ClassificationWindowedEvaluator(ClassificationEvaluator):
    """
    Uses the ClassificationEvaluator to perform a windowed evaluation.

    IMPORTANT: The results for the last window are not always available through ```metrics()```, if the window_size does
    not perfectly divide the stream, the metrics corresponding to the last remaining instances in the last window can
    be obtained by invoking ```metrics()```
    """

    def __init__(self, schema=None, window_size=1000):
        self.moa_evaluator = WindowClassificationPerformanceEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)

        super().__init__(
            schema=schema,
            window_size=window_size,
            moa_evaluator=self.moa_evaluator,
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        pass

    # This allows access to metrics that are generated dynamically like recall_0, f1_score_3, ...
    def __getattr__(self, metric):
        if metric in self.metrics_header():

            def metric_value():
                return self.metrics_per_window()[metric].tolist()

            return metric_value
        return None

    def accuracy(self):
        return self.metrics_per_window()["accuracy"].tolist()

    def kappa(self):
        return self.metrics_per_window()["kappa"].tolist()

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
        return self.metrics_per_window()["mae"].tolist()

    def rmse(self):
        return self.metrics_per_window()["rmse"].tolist()

    def rmae(self):
        return self.metrics_per_window()["rmae"].tolist()

    def r2(self):
        return self.metrics_per_window()["r2"].tolist()

    def adjusted_r2(self):
        return self.metrics_per_window()["adjusted_r2"].tolist()


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

        super().__init__(
            schema=schema, window_size=window_size, moa_evaluator=self.moa_evaluator
        )

    def coverage(self):
        return self.metrics_per_window()["coverage"].tolist()

    def average_length(self):
        return self.metrics_per_window()["average length"].tolist()

    def nmpiw(self):
        return self.metrics_per_window()["nmpiw"].tolist()


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


def _get_target(
    instance: Union[LabeledInstance, RegressionInstance],
) -> Union[int, np.double]:
    """Get the target value from an instance."""
    if isinstance(instance, LabeledInstance):
        return instance.y_index
    elif isinstance(instance, RegressionInstance):
        return instance.y_value
    else:
        raise ValueError("Unknown instance type")


def prequential_evaluation(
    stream: Stream,
    learner: Union[Classifier, Regressor],
    max_instances: Optional[int] = None,
    window_size: int = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: Union[bool, tqdm] = False,
    batch_size: int = 1,
) -> PrequentialResults:
    """Run and evaluate a learner on a stream using prequential evaluation.

    Calculates the metrics cumulatively (i.e. test-then-train) and in a
    window-fashion (i.e. windowed prequential evaluation). Returns both
    evaluators so that the user has access to metrics from both evaluators.

    :param stream: A data stream to evaluate the learner on. Will be restarted if
        ``restart_stream`` is True.
    :param learner: The learner to evaluate.
    :param max_instances: The number of instances to evaluate before exiting. If
        None, the evaluation will continue until the stream is empty.
    :param window_size: The size of the window used for windowed evaluation,
        defaults to 1000
    :param store_predictions: Store the learner's prediction in a list, defaults
        to False
    :param store_y: Store the ground truth targets in a list, defaults to False
    :param optimise: If True and the learner is compatible, the evaluator will
        use a Java native evaluation loop, defaults to True.
    :param restart_stream: If False, evaluation will continue from the current
        position in the stream, defaults to True. Not restarting the stream is
        useful for switching between learners or evaluators, without starting
        from the beginning of the stream.
    :param progress_bar: Enable, disable, or override the progress bar. Currently
        incompatible with ``optimize=True``.
    :param mini_batch: The size of the mini-batch to use for the learner.
    :return: An object containing the results of the evaluation windowed metrics,
        cumulative metrics, ground truth targets, and predictions.
    """
    if restart_stream:
        stream.restart()
    if batch_size != 1 and not isinstance(learner, (BatchClassifier, BatchRegressor)):
        raise ValueError(
            "The learner is not a batch learner, but mini_batch is set to a value greater than 1."
        )
    if _is_fast_mode_compilable(stream, learner, optimise):
        return _prequential_evaluation_fast(
            stream,
            learner,
            max_instances,
            window_size,
            store_y=store_y,
            store_predictions=store_predictions,
        )

    predictions = None
    if store_predictions:
        predictions = []

    ground_truth_y = None
    if store_y:
        ground_truth_y = []

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

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

    progress_bar = _setup_progress_bar(
        "Eval", progress_bar, stream, learner, max_instances
    )

    for i, batch in enumerate(batched(islice(stream, max_instances), batch_size)):
        yb_true = [_get_target(instance) for instance in batch]  # batch of targets
        yb_pred = []

        if isinstance(learner, (BatchClassifier, BatchRegressor)):
            xb = [instance.x for instance in batch]  # batch of features
            yb_pred = learner.batch_predict(np.stack(xb)).tolist()
            learner.batch_train(np.stack(xb), np.stack(yb_true))
        else:
            for instance in batch:
                yb_pred.append(learner.predict(instance))
                learner.train(instance)

        for y_true, y_pred in zip(yb_true, yb_pred, strict=True):
            evaluator_cumulative.update(y_true, y_pred)
            if window_size is not None:
                evaluator_windowed.update(y_true, y_pred)

            # Storing predictions if store_predictions was set to True during initialisation
            if predictions is not None:
                predictions.append(y_pred)

            # Storing ground-truth if store_y was set to True during initialisation
            if ground_truth_y is not None:
                ground_truth_y.append(y_true)

        if progress_bar is not None:
            progress_bar.update(len(batch))

    if progress_bar is not None:
        progress_bar.close()

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

    results = PrequentialResults(
        learner=str(learner),
        stream=stream,
        wallclock=elapsed_wallclock_time,
        cpu_time=elapsed_cpu_time,
        max_instances=max_instances,
        cumulative_evaluator=evaluator_cumulative,
        windowed_evaluator=evaluator_windowed,
        ground_truth_y=ground_truth_y,
        predictions=predictions,
    )

    return results


def prequential_ssl_evaluation(
    stream: Stream,
    learner: Union[ClassifierSSL, Classifier],
    max_instances: Optional[int] = None,
    window_size: int = 1000,
    initial_window_size: int = 0,
    delay_length: int = 0,
    label_probability: float = 0.01,
    random_seed: int = 1,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: Union[bool, tqdm] = False,
    batch_size: int = 1,
):
    """Run and evaluate a learner on a semi-supervised stream using prequential evaluation.

    :param stream: A data stream to evaluate the learner on. Will be restarted if
        ``restart_stream`` is True.
    :param learner: The learner to evaluate. If the learner is an SSL learner,
        it will be trained on both labeled and unlabeled instances. If the
        learner is not an SSL learner, then it will be trained only on the
        labeled instances.
    :param max_instances: The number of instances to evaluate before exiting.
        If None, the evaluation will continue until the stream is empty.
    :param window_size: The size of the window used for windowed evaluation,
        defaults to 1000
    :param initial_window_size: Not implemented yet
    :param delay_length: If greater than zero the labeled (``label_probability``%)
        instances will appear as unlabeled before reappearing as labeled after
        ``delay_length`` instances, defaults to 0
    :param label_probability: The proportion of instances that will be labeled,
        must be in the range [0, 1], defaults to 0.01
    :param random_seed: A random seed to define the random state that decides
        which instances are labeled and which are not, defaults to 1.
    :param store_predictions: Store the learner's prediction in a list, defaults
        to False
    :param store_y: Store the ground truth targets in a list, defaults to False
    :param optimise: If True and the learner is compatible, the evaluator will
        use a Java native evaluation loop, defaults to True.
    :param restart_stream: If False, evaluation will continue from the current
        position in the stream, defaults to True. Not restarting the stream is
        useful for switching between learners or evaluators, without starting
        from the beginning of the stream.
    :param progress_bar: Enable, disable, or override the progress bar. Currently
        incompatible with ``optimize=True``.
    :return: An object containing the results of the evaluation windowed metrics,
        cumulative metrics, ground truth targets, and predictions.
    """

    if restart_stream:
        stream.restart()

    if _is_fast_mode_compilable(stream, learner, optimise):
        return _prequential_ssl_evaluation_fast(
            stream,
            learner,
            max_instances,
            window_size,
            initial_window_size,
            delay_length,
            label_probability,
            random_seed,
        )

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

    progress_bar = _setup_progress_bar(
        "SSL Eval", progress_bar, stream, learner, max_instances
    )
    for i, instance in enumerate(stream):
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

        if progress_bar is not None:
            progress_bar.update(1)

        if max_instances is not None and i >= (max_instances - 1):
            break

    if progress_bar is not None:
        progress_bar.close()

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

    results = PrequentialResults(
        learner=str(learner),
        stream=stream,
        wallclock=elapsed_wallclock_time,
        cpu_time=elapsed_cpu_time,
        max_instances=max_instances,
        cumulative_evaluator=evaluator_cumulative,
        windowed_evaluator=evaluator_windowed,
        ground_truth_y=ground_truth_y,
        predictions=predictions,
        other_metrics={
            "unlabeled": unlabeled_counter,
            "unlabeled_ratio": unlabeled_counter / i,
        },
    )

    return results


def prequential_evaluation_anomaly(
    stream,
    learner,
    max_instances=None,
    window_size=1000,
    optimise=True,
    store_predictions=False,
    store_y=False,
    progress_bar: Union[bool, tqdm] = False,
):
    """
    Calculates the metrics cumulatively (i.e. test-then-train) and in a window-fashion (i.e. windowed prequential
    evaluation). Returns both evaluators so that the user has access to metrics from both evaluators.

    :param progress_bar: Enable, disable, or override the progress bar. Currently
        incompatible with ``optimize=True``.
    """
    stream.restart()
    if _is_fast_mode_compilable(stream, learner, optimise):
        return _prequential_evaluation_anomaly_fast(
            stream, learner, max_instances, window_size, store_y, store_predictions
        )

    predictions = None
    if store_predictions:
        predictions = []

    ground_truth_y = None
    if store_y:
        ground_truth_y = []

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    instances_processed = 1

    evaluator_cumulative = AnomalyDetectionEvaluator(
        schema=stream.get_schema(), window_size=window_size
    )
    evaluator_windowed = None
    if window_size is not None:
        evaluator_windowed = AnomalyDetectionWindowedEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )

    progress_bar = _setup_progress_bar(
        "AD Eval", progress_bar, stream, learner, max_instances
    )
    while stream.has_more_instances() and (
        max_instances is None or instances_processed <= max_instances
    ):
        instance = stream.next_instance()
        prediction = learner.score_instance(instance)
        y = instance.y_index
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

        instances_processed += 1
        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

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

    results = PrequentialResults(
        learner=str(learner),
        stream=stream,
        wallclock=elapsed_wallclock_time,
        cpu_time=elapsed_cpu_time,
        max_instances=max_instances,
        cumulative_evaluator=evaluator_cumulative,
        windowed_evaluator=evaluator_windowed,
        ground_truth_y=ground_truth_y,
        predictions=predictions,
    )

    return results


##############################################################
###### OPTIMISED VERSIONS (use experimental MOA method) ######
##############################################################


def _prequential_evaluation_fast(
    stream,
    learner,
    max_instances=None,
    window_size=1000,
    store_y=False,
    store_predictions=False,
):
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
        for i in range(
            len(
                moa_results.targets
                if len(moa_results.targets) != 0
                else moa_results.predictions
            )
        ):
            if store_y:
                ground_truth_y.append(moa_results.targets[i])
            if store_predictions:
                predictions.append(moa_results.predictions[i])

    results = PrequentialResults(
        learner=str(learner),
        stream=stream,
        wallclock=elapsed_wallclock_time,
        cpu_time=elapsed_cpu_time,
        max_instances=max_instances,
        cumulative_evaluator=basic_evaluator,
        windowed_evaluator=windowed_evaluator,
        ground_truth_y=ground_truth_y,
        predictions=predictions,
    )

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
    store_predictions=False,
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
        for i in range(
            len(
                moa_results.targets
                if len(moa_results.targets) != 0
                else moa_results.predictions
            )
        ):
            if store_y:
                ground_truth_y.append(moa_results.targets[i])
            if store_predictions:
                predictions.append(moa_results.predictions[i])

    results = PrequentialResults(
        learner=str(learner),
        stream=stream,
        wallclock=elapsed_wallclock_time,
        cpu_time=elapsed_cpu_time,
        max_instances=max_instances,
        cumulative_evaluator=basic_evaluator,
        windowed_evaluator=windowed_evaluator,
        ground_truth_y=ground_truth_y,
        predictions=predictions,
        other_metrics=dict(moa_results.otherMeasurements),
    )

    return results


def _prequential_evaluation_anomaly_fast(
    stream,
    learner,
    max_instances=None,
    window_size=1000,
    store_y=False,
    store_predictions=False,
):
    """
    Fast prequential evaluation for Anomaly Detectors.
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
    if not isinstance(learner, AnomalyDetector):
        raise ValueError("The learner is not an AnomalyDetector")
    basic_evaluator = AnomalyDetectionEvaluator(schema=stream.get_schema())
    windowed_evaluator = AnomalyDetectionWindowedEvaluator(
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
        for i in range(
            len(
                moa_results.targets
                if len(moa_results.targets) != 0
                else moa_results.predictions
            )
        ):
            if store_y:
                ground_truth_y.append(moa_results.targets[i])
            if store_predictions:
                predictions.append(moa_results.predictions[i])

    results = PrequentialResults(
        learner=str(learner),
        stream=stream,
        wallclock=elapsed_wallclock_time,
        cpu_time=elapsed_cpu_time,
        max_instances=max_instances,
        cumulative_evaluator=basic_evaluator,
        windowed_evaluator=windowed_evaluator,
        ground_truth_y=ground_truth_y,
        predictions=predictions,
    )

    return results


########################################################################################
###### EXPERIMENTAL (optimisation to go over the data once for several learners)  ######
########################################################################################


def prequential_evaluation_multiple_learners(
    stream,
    learners,
    max_instances=None,
    window_size=1000,
    store_predictions=False,
    store_y=False,
    progress_bar: Union[bool, tqdm] = False,
):
    """
    Calculates the metrics cumulatively (i.e., test-then-train) and in a windowed-fashion for multiple streams and
    learners. It behaves as if we invoked prequential_evaluation() multiple times, but we only iterate through the
    stream once.
    This function is useful in situations where iterating through the stream is costly, but we still want to assess
    several learners on it.
    Returns the results in a dictionary format. Infers whether it is a Classification or Regression problem based on the
    stream schema.

    :param progress_bar: Enable, disable, or override the progress bar.
    """
    results = {}

    stream.restart()

    for learner_name, learner in learners.items():
        predictions = [] if store_predictions else None
        ground_truth_y = [] if store_y else None

        if stream.get_schema().is_classification():
            cumulative_evaluator = ClassificationEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
            windowed_evaluator = (
                ClassificationWindowedEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
                if window_size is not None
                else None
            )
        else:
            if not isinstance(learner, MOAPredictionIntervalLearner):
                cumulative_evaluator = RegressionEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
                windowed_evaluator = (
                    RegressionWindowedEvaluator(
                        schema=stream.get_schema(), window_size=window_size
                    )
                    if window_size is not None
                    else None
                )
            else:
                cumulative_evaluator = PredictionIntervalEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
                windowed_evaluator = (
                    PredictionIntervalWindowedEvaluator(
                        schema=stream.get_schema(), window_size=window_size
                    )
                    if window_size is not None
                    else None
                )

        results[learner_name] = {
            "learner": learner,
            "cumulative_evaluator": cumulative_evaluator,
            "windowed_evaluator": windowed_evaluator,
            "predictions": predictions,
            "ground_truth_y": ground_truth_y,
            "start_wallclock_time": start_time_measuring()[0],
            "start_cpu_time": start_time_measuring()[1],
        }

    instancesProcessed = 1

    progress_bar = resolve_progress_bar(
        progress_bar, f"Eval {len(learners)} learners on {type(stream).__name__}"
    )
    expected_length = _get_expected_length(stream, max_instances)
    if progress_bar is not None and expected_length is not None:
        progress_bar.set_total(expected_length)

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

            results[learner_name]["cumulative_evaluator"].update(y, prediction)
            if window_size is not None:
                results[learner_name]["windowed_evaluator"].update(y, prediction)

            learner.train(instance)

            # Storing predictions if store_predictions was set to True during initialization
            if results[learner_name]["predictions"] is not None:
                results[learner_name]["predictions"].append(prediction)

            # Storing ground-truth if store_y was set to True during initialization
            if results[learner_name]["ground_truth_y"] is not None:
                results[learner_name]["ground_truth_y"].append(y)

        instancesProcessed += 1
        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    # Iterate through the results of each learner and add (if needed) the last window of results to it.
    if window_size is not None:
        for learner_name, result in results.items():
            if (
                result["windowed_evaluator"] is not None
                and result["windowed_evaluator"].get_instances_seen() % window_size != 0
            ):
                result["windowed_evaluator"].result_windows.append(
                    result["windowed_evaluator"].metrics()
                )

    # Creating PrequentialResults instances for each learner
    final_results = {}
    for learner_name, result in results.items():
        elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
            result["start_wallclock_time"], result["start_cpu_time"]
        )
        final_results[learner_name] = PrequentialResults(
            learner=str(result["learner"]),
            stream=stream,
            wallclock=elapsed_wallclock_time,
            cpu_time=elapsed_cpu_time,
            max_instances=max_instances,
            cumulative_evaluator=result["cumulative_evaluator"],
            windowed_evaluator=result["windowed_evaluator"],
            ground_truth_y=result["ground_truth_y"],
            predictions=result["predictions"],
        )

    return final_results


def write_results_to_files(
    path: str = None, results=None, file_name: str = None, directory_name: str = None
):
    if results is None:
        raise ValueError("The results object is None")

    path = path if path.endswith("/") else (path + "/")

    if isinstance(results, ClassificationWindowedEvaluator) or isinstance(
        results, RegressionWindowedEvaluator
    ):
        data = results.metrics_per_window()
        data.to_csv(
            ("./" if path is None else path)
            + ("/windowed_results.csv" if file_name is None else file_name),
            index=False,
        )
    elif isinstance(results, ClassificationEvaluator) or isinstance(
        results, RegressionEvaluator
    ):
        json_str = json.dumps(results.metrics_dict())
        data = json.loads(json_str)
        with open(
            ("./" if path is None else path)
            + ("/cumulative_results.csv" if file_name is None else file_name),
            "w",
            newline="",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data.keys())
            writer.writerow(data.values())
    elif isinstance(results, PrequentialResults):
        directory_name = (
            "prequential_results" if directory_name is None else directory_name
        )
        if os.path.exists(path + "/" + directory_name):
            raise ValueError(
                f"Directory {directory_name} already exists, please use another name"
            )
        else:
            os.makedirs(path + "/" + directory_name)

        write_results_to_files(
            path=path + "/" + directory_name,
            file_name=file_name,
            results=results.cumulative,
        )
        write_results_to_files(
            path=path + "/" + directory_name,
            file_name=file_name,
            results=results.windowed,
        )

        # If the ground truth and predictions are available, they will be writen to a file
        if (
            results.get_ground_truth_y() is not None
            and results.get_predictions() is not None
        ):
            y_vs_predictions = {
                "ground_truth_y": results.get_ground_truth_y(),
                "predictions": results.get_predictions(),
            }
            if len(y_vs_predictions) > 0:
                t_p = pd.DataFrame(y_vs_predictions)
                t_p.to_csv(
                    ("./" if path is None else path)
                    + "/"
                    + directory_name
                    + "/ground_truth_y_and_predictions.csv",
                    index=False,
                )
    else:
        raise ValueError(
            "Writing results to file is not supported for type "
            + str(type(results))
            + " yet"
        )
