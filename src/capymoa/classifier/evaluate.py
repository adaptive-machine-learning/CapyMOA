"""Evaluators and result types for classification.

:func:`~capymoa.classifier.evaluate_classifier` returns a :class:`ClassifierResults`. It
uses the evaluators below, which wrap MOA's evaluators. You can also use them in your
own test-then-train loop.
"""

from typing import NotRequired

import numpy as np
import pandas as pd
from com.yahoo.labs.samoa.instances import Attribute, DenseInstance, Instances
from java.util import ArrayList
from moa.core import InstanceExample
from moa.evaluation import (
    BasicClassificationPerformanceEvaluator,
    WindowClassificationPerformanceEvaluator,
)
from typing_extensions import TypedDict

from capymoa._utils import _translate_metric_name
from capymoa.evaluation.results import RunInfo, Windows
from capymoa.stream import Schema


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
    """Results of evaluating a classifier. See :func:`~capymoa.classifier.evaluate_classifier`.

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
    #: The metrics of each class in columns (see
    #: :class:`~capymoa.classifier.evaluate.PerClass`).
    per_class: PerClass
    #: The metrics of each window (see
    #: :class:`~capymoa.classifier.evaluate.ClassifierWindows`). Absent if
    #: ``window_size`` is absent.
    windowed: NotRequired[ClassifierWindows]


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
        self.moa_basic_evaluator.rocAucOption.set()
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

    def update(self, y_target_index: int, y_pred_index: int | None):
        """Update the evaluator with the ground-truth and the prediction.

        :param y_target_index: The ground-truth class index. This is NOT
            the actual class value, but the index of the class value in the
            schema.
        :param y_pred_index: The predicted class index. If the classifier
            abstains from making a prediction, this value can be None.
        :raises TypeError: If the values are not valid indexes in the schema.
        """
        if not isinstance(y_target_index, (np.integer, int)):
            raise TypeError(
                f"y_target_index must be an integer, not {type(y_target_index)}"
            )
        if not (y_pred_index is None or isinstance(y_pred_index, (np.integer, int))):
            raise TypeError(
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

    def roc_auc(self):
        index = self.metrics_header().index("roc_auc")
        return float(self.metrics()[index])


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

    def roc_auc(self):
        return self.metrics_per_window()["roc_auc"].tolist()
