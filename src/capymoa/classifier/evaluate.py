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

    accuracy: np.ndarray
    r"""Percent of correct predictions in each window.

    .. math::

       \text{accuracy} = \frac{\text{correct predictions}}{\text{all predictions}}

    Counts are per window.
    """
    #: Cohen's kappa in percent.
    kappa: np.ndarray
    #: Kappa temporal in percent.
    kappa_t: np.ndarray
    #: Kappa M in percent.
    kappa_m: np.ndarray
    f1_score: np.ndarray
    r"""Micro-averaged F1 score in percent (equals accuracy).

    .. math::

       F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}

    TP, FP and FN are counts over all classes (per window).
    """
    precision: np.ndarray
    r"""Micro-averaged precision in percent (equals accuracy).

    .. math::

       \text{precision} = \frac{TP}{TP + FP}

    TP, FP and FN are counts over all classes (per window).
    """
    recall: np.ndarray
    r"""Micro-averaged recall in percent (equals accuracy).

    .. math::

       \text{recall} = \frac{TP}{TP + FN}

    TP, FP and FN are counts over all classes (per window).
    """
    precision_macro: np.ndarray
    r"""Macro-averaged precision in percent: the mean of the precision of each class.

    .. math::

       \text{precision}_{macro} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c}

    ``C`` is the number of classes. TP, FP and FN are counts per class.
    """
    recall_macro: np.ndarray
    r"""Macro-averaged recall in percent: the mean of the recall of each class.

    .. math::

       \text{recall}_{macro} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}

    ``C`` is the number of classes. TP, FP and FN are counts per class.
    """
    f1_score_macro: np.ndarray
    r"""Macro-averaged F1 score in percent: the mean of the F1 score of each class.

    .. math::

       F_{1,macro} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}

    ``P_c`` and ``R_c`` are the precision and recall of class ``c``.
    """
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


class PerClassWindows(TypedDict):
    """Metrics of each class in each window, in columns.

    One row per window and class, so ``pd.DataFrame(windows)`` makes a long
    table. Rows are sorted by window, then by class.
    """

    #: The number of instances seen at the end of the window.
    instances: np.ndarray
    #: The label of the class.
    label: np.ndarray
    #: Precision in percent.
    precision: np.ndarray
    #: Recall in percent.
    recall: np.ndarray
    #: F1 score in percent.
    f1_score: np.ndarray


class ClassifierCumulative(TypedDict):
    """The metrics over the whole stream of a classifier."""

    accuracy: float
    r"""Percent of correct predictions.

    .. math::

       \text{accuracy} = \frac{\text{correct predictions}}{\text{all predictions}}

    Counts are over the whole stream.
    """
    #: Cohen's kappa in percent.
    kappa: float
    #: Kappa temporal in percent. Compares with a classifier that predicts the
    #: previous label.
    kappa_t: float
    #: Kappa M in percent. Compares with a classifier that predicts the
    #: majority label.
    kappa_m: float
    f1_score: float
    r"""Micro-averaged F1 score in percent (equals accuracy).

    .. math::

       F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}

    TP, FP and FN are counts over all classes (whole stream).
    """
    precision: float
    r"""Micro-averaged precision in percent (equals accuracy).

    .. math::

       \text{precision} = \frac{TP}{TP + FP}

    TP, FP and FN are counts over all classes (whole stream).
    """
    recall: float
    r"""Micro-averaged recall in percent (equals accuracy).

    .. math::

       \text{recall} = \frac{TP}{TP + FN}

    TP, FP and FN are counts over all classes (whole stream).
    """
    precision_macro: float
    r"""Macro-averaged precision in percent: the mean of the precision of each class.

    .. math::

       \text{precision}_{macro} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c}

    ``C`` is the number of classes. TP, FP and FN are counts per class over
    the whole stream.
    """
    recall_macro: float
    r"""Macro-averaged recall in percent: the mean of the recall of each class.

    .. math::

       \text{recall}_{macro} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}

    ``C`` is the number of classes. TP, FP and FN are counts per class over
    the whole stream.
    """
    f1_score_macro: float
    r"""Macro-averaged F1 score in percent: the mean of the F1 score of each class.

    .. math::

       F_{1,macro} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}

    ``P_c`` and ``R_c`` are the precision and recall of class ``c``.
    """
    #: Area under the ROC curve (0 to 1). Absent if it is not defined.
    roc_auc: NotRequired[float]


class ClassifierResults(RunInfo):
    """Results of evaluating a classifier. See :func:`~capymoa.classifier.evaluate_classifier`."""

    #: The metrics over the whole stream (see
    #: :class:`~capymoa.classifier.evaluate.ClassifierCumulative`).
    cumulative: ClassifierCumulative
    #: The metrics of each class over the whole stream, in columns (see
    #: :class:`~capymoa.classifier.evaluate.PerClass`).
    cumulative_per_class: PerClass

    #: The metrics of each window (see
    #: :class:`~capymoa.classifier.evaluate.ClassifierWindows`). Absent if
    #: ``window_size`` is absent.
    windowed: NotRequired[ClassifierWindows]
    #: The metrics of each class in each window, one row per window and class
    #: (see :class:`~capymoa.classifier.evaluate.PerClassWindows`). Absent if
    #: ``window_size`` is absent.
    windowed_per_class: NotRequired[PerClassWindows]


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
        # Python calls __getattr__ for every attribute that is not set, which
        # includes the __setstate__ probe that pickle makes before __init__ has
        # run. Reading self.moa_basic_evaluator would then ask for an attribute
        # that is missing, land back in this method, and recurse until the
        # stack limit. Stop when the evaluator is not there.
        if "moa_basic_evaluator" not in self.__dict__:
            raise AttributeError(metric)
        if metric in self.metrics_header():
            index = self.metrics_header().index(metric)

            def metric_value():
                return float(self.metrics()[index])

            return metric_value
        raise AttributeError(metric)

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
        # See ClassificationEvaluator.__getattr__ for why this guard is needed.
        if "moa_basic_evaluator" not in self.__dict__:
            raise AttributeError(metric)
        if metric in self.metrics_header():

            def metric_value():
                return self.metrics_per_window()[metric].tolist()

            return metric_value
        raise AttributeError(metric)

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
