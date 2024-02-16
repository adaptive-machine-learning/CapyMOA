# Python imports
import pandas as pd
import numpy as np
import time
import datetime as dt
import os
import random

# Library imports
from capymoa.stream.stream import NumpyStream
from capymoa.learner.learners import ClassifierSSL

# MOA/Java imports
from com.yahoo.labs.samoa.instances import Instances, Instance, Attribute, DenseInstance
from moa.core import Example, InstanceExample, Utils
from moa.evaluation import (
    BasicClassificationPerformanceEvaluator,
    WindowClassificationPerformanceEvaluator,
    BasicRegressionPerformanceEvaluator,
    WindowRegressionPerformanceEvaluator,
)
from java.util import ArrayList
from moa.evaluation import EfficientEvaluationLoops


class ClassificationEvaluator:
    """
    Wrapper for the Classification Performance Evaluator from MOA. By default uses the BasicClassificationPerformanceEvaluator
    """

    def __init__(
        self,
        schema=None,
        window_size=None,
        allow_abstaining=True,
        recall_per_class=False,
        precision_per_class=False,
        f1_precision_recall=False,
        f1_per_class=False,
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
        return str(
            {
                header: value
                for header, value in zip(self.metrics_header(), self.metrics())
            }
        )

    # # def metrics_with_header(self):
    #     return {header: value for header, value in zip(self.metrics_header(), self.metrics())}

    def get_instances_seen(self):
        return self.instances_seen

    def update(self, y, y_pred):
        """
        Updates the metrics based on the true label (y) and predicted label (y_label).

        Parameters:
        - y (class value (int, string, ...)): The true label.
        - y_pred (class value (int, string, ...)): The predicted label.

        Returns:
        None

        Notes:
        - This method assumes the predictions passed are class values instead of any internal representation, such as class indexes. 
        """

        # The class label should be valid, if an exception is thrown here, the code should stop. 
        y_index = self.schema.get_valid_index_for_label(y)
        # If the prediction is invalid, it could mean the classifier is abstaining from making a prediction; 
        #   thus, it is allowed to continue (unless parameterized differently).
        y_pred_index = 0
        try:
            y_pred_index = self.schema.get_valid_index_for_label(y_pred)
        except Exception as e:
            if self.allow_abstaining == False:
                raise

        if y_index is None:
            raise ValueError(f"Invalid ground-truth (y) value {y}")

        # Notice, in MOA the class value is an index, not the actual value (e.g. not "one" but 0 assuming labels=["one", "two"])
        self._instance.setClassValue(y_index)
        example = InstanceExample(self._instance)

        # Shallow copy of the pred_template
        # MOA evaluator accepts the result of getVotesForInstance which is similar to a predict_proba
        #    (may or may not be normalised, but for our purposes it doesn't matter)
        prediction_array = self.pred_template[:]

        # if y_pred is None, it indicates the learner did not produce a prediction for this instace, count as an error
        if y_pred_index is None:
            # Set y_pred_index to any valid prediction that is not y (force an incorrect prediction)
            # This does not affect recall or any other metrics, because the selected value is always incorrect.

            # Create an intermediary array with indices excluding the y
            indexesWithoutY = [
                i for i in range(len(self.schema.get_label_indexes())) if i != y_index
            ]
            random_y_pred = random.choice(indexesWithoutY)
            y_pred_index = self.schema.get_label_indexes()[random_y_pred]

        prediction_array[int(y_pred_index)] += 1
        self.moa_basic_evaluator.addResult(example, prediction_array)

        self.instances_seen += 1

        # If the window_size is set, then check if it should record the intermediary results.
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            performance_values = (
                self.metrics()
            )
            self.result_windows.append(performance_values)

    def metrics_header(self):
        performance_measurements = self.moa_basic_evaluator.getPerformanceMeasurements()
        performance_names = [
            "".join(measurement.getName()) for measurement in performance_measurements
        ]
        return performance_names

    def metrics(self):
        return [
            measurement.getValue()
            for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()
        ]

    def metrics_per_window(self):
        return pd.DataFrame(self.result_windows, columns=self.metrics_header())

    def accuracy(self):
        index = self.metrics_header().index("classifications correct (percent)")
        return self.metrics()[index]

    def kappa(self):
        index = self.metrics_header().index("Kappa Statistic (percent)")
        return self.metrics()[index]

    def kappa_temporal(self):
        index = self.metrics_header().index("Kappa Temporal Statistic (percent)")
        return self.metrics()[index]

    def kappa_M(self):
        index = self.metrics_header().index("Kappa M Statistic (percent)")
        return self.metrics()[index]


class ClassificationWindowedEvaluator(ClassificationEvaluator):
    """
    The results for the last window are always through ```metrics()```, if the window_size does not perfectly divides the stream, i.e.
    there are remaining instances are the last window, then we can obtain the results for this last window by invoking ```metrics()```
    """

    def __init__(
        self,
        schema=None,
        window_size=1000,
        recall_per_class=False,
        precision_per_class=False,
        f1_precision_recall=False,
        f1_per_class=False,
    ):
        self.moa_evaluator = WindowClassificationPerformanceEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)

        super().__init__(
            schema=schema,
            window_size=window_size,
            recall_per_class=recall_per_class,
            precision_per_class=precision_per_class,
            f1_precision_recall=f1_precision_recall,
            f1_per_class=f1_per_class,
            moa_evaluator=self.moa_evaluator,
        )


class RegressionEvaluator:
    """
    Wrapper for the Regression Performance Evaluator from MOA. By default uses the BasicRegressionPerformanceEvaluator
    """

    def __init__(self, schema=None, window_size=None, moa_evaluator=None):
        self.instances_seen = 0
        self.result_windows = []
        self.window_size = window_size

        self.moa_basic_evaluator = moa_evaluator
        if self.moa_basic_evaluator is None:
            self.moa_basic_evaluator = BasicRegressionPerformanceEvaluator()

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

        # Regression has only one output
        self.pred_template = [0]

        # Create the denseInstance just once and keep reusing it by changing the classValue (more efficient).
        self._instance = DenseInstance(self.schema.get_num_attributes() + 1)
        self._instance.setDataset(self._header)

    def __str__(self):
        return str(
            {
                header: value
                for header, value in zip(self.metrics_header(), self.metrics())
            }
        )

    def get_instances_seen(self):
        return self.instances_seen

    def update(self, y, y_pred):
        if y is None:
            raise ValueError(f"Invalid ground-truth y = {y}")

        self._instance.setClassValue(y)
        example = InstanceExample(self._instance)

        # if y_pred is None, it indicates the learner did not produce a prediction for this instace
        if y_pred is None:
            # In classification it is rather easy to deal with this, but

            # Create an intermediary array with indices excluding the y
            indexesWithoutY = [
                i for i in range(len(self.schema.get_label_indexes())) if i != y_index
            ]
            random_y_pred = random.choice(indexesWithoutY)
            y_pred_index = self.schema.get_label_indexes()[random_y_pred]

        # Different from classification, there is no need to make a shallow copy of the prediction array, just override the value.
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
            "".join(measurement.getName()) for measurement in performance_measurements
        ]
        return performance_names

    def metrics(self):
        return [
            measurement.getValue()
            for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()
        ]

    def metrics_per_window(self):
        return pd.DataFrame(self.result_windows, columns=self.metrics_header())

    def MAE(self):
        index = self.metrics_header().index("mean absolute error")
        return self.metrics()[index]

    def RMSE(self):
        index = self.metrics_header().index("root mean squared error")
        return self.metrics()[index]

    def RMAE(self):
        index = self.metrics_header().index("relative mean absolute error")
        return self.metrics()[index]

    def R2(self):
        index = self.metrics_header().index("coefficient of determination")
        return self.metrics()[index]

    def adjusted_R2(self):
        index = self.metrics_header().index("adjusted coefficient of determination")
        return self.metrics()[index]


class RegressionWindowedEvaluator(RegressionEvaluator):
    """
    The results for the last window are always through ```metrics()```, if the window_size does not perfectly divides the stream, i.e.
    there are remaining instances are the last window, then we can obtain the results for this last window by invoking ```metrics()```
    """

    def __init__(self, schema=None, window_size=1000):
        self.moa_evaluator = WindowRegressionPerformanceEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)

        super().__init__(
            schema=schema, window_size=window_size, moa_evaluator=self.moa_evaluator
        )


## Functions to measure runtime
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


def test_then_train_evaluation(
    stream,
    learner,
    max_instances=None,
    sample_frequency=None,
    evaluator=None,
    optimise=True,
):
    """
    Test-then-train evaluation. Returns a dictionary with the results.
    """

    if isinstance(stream, NumpyStream) == False and optimise:
        return test_then_train_evaluation_fast(
            stream, learner, max_instances, sample_frequency, evaluator
        )

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    instancesProcessed = 1

    if stream.has_more_instances() == False:
        stream.restart()

    if evaluator is None:
        schema = stream.get_schema()
        if schema.is_classification():
            evaluator = ClassificationEvaluator(
                schema=schema, window_size=sample_frequency
            )
        else:
            evaluator = RegressionEvaluator(schema=schema, window_size=sample_frequency)

    while stream.has_more_instances() and (
        max_instances is None or instancesProcessed <= max_instances
    ):
        instance = stream.next_instance()

        prediction = learner.predict(instance)
        evaluator.update(instance.y(), prediction)
        learner.train(instance)

        instancesProcessed += 1

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    results = {
        "learner": str(learner),
        "cumulative": evaluator,
        "wallclock": elapsed_wallclock_time,
        "cpu_time": elapsed_cpu_time,
        "max_instances":max_instances, 
        "stream":stream,
    }

    return results


def windowed_evaluation(stream, learner, max_instances=None, window_size=1000):
    """
    Prequential evaluation (window). Returns a dictionary with the results.
    """
    # Run test-then-train evaluation, but change the underlying evaluator
    evaluator = None
    if stream.get_schema().is_classification():
        evaluator = ClassificationWindowedEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )
    else:
        evaluator = RegressionWindowedEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )
    results = test_then_train_evaluation(
        stream,
        learner,
        max_instances=max_instances,
        sample_frequency=window_size,
        evaluator=evaluator,
    )

    results["windowed"] = results["cumulative"]
    # Add the results corresponding to the remainder of the stream in case the number of processed instances is not perfectly divisible by
    # the window_size (if it was, then it is already be in the result_windows variable).
    if evaluator.get_instances_seen() % window_size != 0:
        results["windowed"].result_windows.append(results["windowed"].metrics())

    results.pop(
        "cumulative", None
    )  # Remove previous entry with the cumulative results.

    # Ignore the last prediction values, because it doesn't matter as we are using a windowed evaluation.
    return results


def prequential_evaluation(
    stream, learner, max_instances=None, window_size=1000, optimise=True
):
    """
    Calculates the metrics cumulatively (i.e. test-then-train) and in a window-fashion (i.e. windowed prequential evaluation).
    Returns both evaluators so that the caller has access to metric from both evaluators.
    """
    if isinstance(stream, NumpyStream) == False and optimise:
        return prequential_evaluation_fast(stream, learner, max_instances, window_size)

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    instancesProcessed = 1

    if stream.has_more_instances() == False:
        stream.restart()

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
        evaluator_cumulative = RegressionEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )
        if window_size is not None:
            evaluator_windowed = RegressionWindowedEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )

    while stream.has_more_instances() and (
        max_instances is None or instancesProcessed <= max_instances
    ):
        instance = stream.next_instance()

        prediction = learner.predict(instance)

        evaluator_cumulative.update(instance.y(), prediction)
        if window_size is not None:
            evaluator_windowed.update(instance.y(), prediction)
        learner.train(instance)

        instancesProcessed += 1

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    # Add the results corresponding to the remainder of the stream in case the number of processed instances is not perfectly divisible by
    # the window_size (if it was, then it is already be in the result_windows variable).
    # The evaluator_windowed will be None if the window_size is None (it will not be created)
    if (
        evaluator_windowed is not None
        and evaluator_windowed.get_instances_seen() % window_size != 0
    ):
        evaluator_windowed.result_windows.append(evaluator_windowed.metrics())

    results = {
        "learner": str(learner),
        "cumulative": evaluator_cumulative,
        "windowed": evaluator_windowed,
        "wallclock": elapsed_wallclock_time,
        "cpu_time": elapsed_cpu_time,
        "max_instances":max_instances, 
        "stream":stream,
    }

    return results


def test_then_train_SSL_evaluation(
    stream,
    learner,
    max_instances=None,
    sample_frequency=None,
    initial_window_size=0,
    delay_length=0,
    label_probability=0.01,
    random_seed=1,
    evaluator=None,
    optimise=True,
):
    """
    Test-then-train SSL evaluation. Returns a dictionary with the results.
    """
    if isinstance(stream, NumpyStream) == False and optimise:
        return test_then_train_SSL_evaluation_fast(
            stream,
            learner,
            max_instances,
            sample_frequency,
            initial_window_size,
            delay_length,
            label_probability,
            random_seed,
            evaluator,
        )

    raise ValueError("test_then_train_SSL_evaluation(...) not fully implemented yet!")


def prequential_SSL_evaluation(
    stream,
    learner,
    max_instances=None,
    window_size=1000,
    initial_window_size=0,
    delay_length=0,
    label_probability=0.01,
    random_seed=1,
    optimise=True,
):
    """
    If the learner is not a SSL learner, then it will just train on labeled instances.
    """
    if isinstance(stream, NumpyStream) == False and optimise:
        return prequential_SSL_evaluation_fast(
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
    # In MOA it is implemented so prequential_SSL_evaluation_fast works just fine.
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

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    instancesProcessed = 1

    if stream.has_more_instances() == False:
        stream.restart()

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

        evaluator_cumulative.update(instance.y(), prediction)
        if evaluator_windowed is not None:
            evaluator_windowed.update(instance.y(), prediction)

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

        instancesProcessed += 1

    # # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    # Add the results corresponding to the remainder of the stream in case the number of processed instances is not perfectly divisible by
    # the window_size (if it was, then it is already in the result_windows variable).
    if (
        evaluator_windowed is not None
        and evaluator_windowed.get_instances_seen() % window_size != 0
    ):
        evaluator_windowed.result_windows.append(evaluator_windowed.metrics())

    # TODO: create a standard for the otherMeasurements like the PrequentialSSLEvaluation from MOA
    results = {
        "learner": str(learner),
        "cumulative": evaluator_cumulative,
        "windowed": evaluator_windowed,
        "wallclock": elapsed_wallclock_time,
        "cpu_time": elapsed_cpu_time,
        "max_instances":max_instances, 
        "stream":stream,
        "unlabeled": unlabeled_counter,
        "unlabeled_ratio": unlabeled_counter / instancesProcessed,
    }

    return results


##############################################################
###### OPTIMISED VERSIONS (use experimental MOA method) ######
##############################################################


def test_then_train_evaluation_fast(
    stream, learner, max_instances=None, sample_frequency=None, evaluator=None
):
    """
    Test-then-train evaluation using a MOA learner.
    """
    # If NumpyStream was used, the data already sits in Python memory.
    if isinstance(stream, NumpyStream):
        return test_then_train_evaluation(
            stream, learner, max_instances, sample_frequency
        )

    if max_instances is None:
        max_instances = -1

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    if evaluator is None:
        evaluator = ClassificationEvaluator(
            schema=stream.get_schema(), window_size=sample_frequency
        )

    if sample_frequency is not None:
        moa_results = EfficientEvaluationLoops.PrequentialEvaluation(
            stream.moa_stream,
            learner.moa_learner,
            None,
            evaluator.moa_basic_evaluator,
            max_instances,
            sample_frequency,
        )
        # Reset the windowed_evaluator result_windows
        if moa_results != None:
            evaluator.result_windows = []
            if moa_results.windowedResults != None:
                for entry_idx in range(len(moa_results.windowedResults)):
                    evaluator.result_windows.append(
                        moa_results.windowedResults[entry_idx]
                    )
    else:
        # Ignore the moa_results because there is no sample frequency (so no need to obtain the windowed results)
        # Set sample_frequency to -1 in the function call
        EfficientEvaluationLoops.PrequentialEvaluation(
            stream.moa_stream,
            learner.moa_learner,
            evaluator.moa_basic_evaluator,
            None,
            max_instances,
            -1,
        )

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    results = {
        "learner": str(learner),
        "cumulative": evaluator,
        "wallclock": elapsed_wallclock_time,
        "cpu_time": elapsed_cpu_time,
        "max_instances":max_instances, 
        "stream":stream,
    }

    return results


def prequential_evaluation_fast(stream, learner, max_instances=None, window_size=1000):
    """
    Prequential evaluation fast.
    """

    if isinstance(stream, NumpyStream):
        return prequential_evaluation(stream, learner, max_instances, window_size)

    if max_instances is None:
        max_instances = -1

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    basic_evaluator = None
    windowed_evaluator = None
    if stream.get_schema().is_classification():
        basic_evaluator = ClassificationEvaluator(schema=stream.get_schema())
        # Always create the windowed_evaluator, even if window_size is None. TODO: may want to avoid creating it if window_size is None.
        windowed_evaluator = ClassificationWindowedEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )
    else:
        # If it is not classification, must be regression
        basic_evaluator = RegressionEvaluator(schema=stream.get_schema())
        windowed_evaluator = RegressionWindowedEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )

    moa_results = EfficientEvaluationLoops.PrequentialEvaluation(
        stream.moa_stream,
        learner.moa_learner,
        basic_evaluator.moa_basic_evaluator,
        windowed_evaluator.moa_evaluator,
        max_instances,
        window_size,
    )

    # Reset the windowed_evaluator result_windows
    if moa_results != None:
        windowed_evaluator.result_windows = []
        if moa_results.windowedResults != None:
            for entry_idx in range(len(moa_results.windowedResults)):
                windowed_evaluator.result_windows.append(
                    moa_results.windowedResults[entry_idx]
                )

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    results = {
        "learner": str(learner),
        "cumulative": basic_evaluator,
        "windowed": windowed_evaluator,
        "wallclock": elapsed_wallclock_time,
        "cpu_time": elapsed_cpu_time,
        "max_instances":max_instances, 
        "stream":stream,
    }

    return results


def test_then_train_SSL_evaluation_fast(
    stream,
    learner,
    max_instances=None,
    sample_frequency=None,
    initial_window_size=0,
    delay_length=0,
    label_probability=0.01,
    random_seed=1,
    evaluator=None,
):
    """
    Test-then-train SSL evaluation.
    """
    # If NumpyStream was used, the data already sits in Python memory.
    if isinstance(stream, NumpyStream):
        raise ValueError("test_then_train_SSL_evaluation(...) to be implemented")
        # return test_then_train_SSL_evaluation(stream, learner, max_instances, sample_frequency,
        #                                 initial_window_size, delay_length, label_probability, random_seed, evaluator)

    if max_instances is None:
        max_instances = -1

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    if evaluator is None:
        evaluator = ClassificationEvaluator(
            schema=stream.get_schema(), window_size=sample_frequency
        )

    moa_results = None
    if sample_frequency is not None:
        moa_results = EfficientEvaluationLoops.PrequentialSSLEvaluation(
            stream.moa_stream,
            learner.moa_learner,
            None,
            evaluator.moa_basic_evaluator,
            max_instances,
            sample_frequency,
            initial_window_size,
            delay_length,
            label_probability,
            random_seed,
            True,
        )
        # Reset the windowed_evaluator result_windows
        if moa_results != None:
            evaluator.result_windows = []
            if moa_results.windowedResults != None:
                for entry_idx in range(len(moa_results.windowedResults)):
                    evaluator.result_windows.append(
                        moa_results.windowedResults[entry_idx]
                    )
    else:
        # Set sample_frequency to -1 in the function call
        moa_results = EfficientEvaluationLoops.PrequentialSSLEvaluation(
            stream.moa_stream,
            learner.moa_learner,
            evaluator.moa_basic_evaluator,
            None,
            max_instances,
            -1,
            initial_window_size,
            delay_length,
            label_probability,
            random_seed,
            True,
        )

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    results = {
        "learner": str(learner),
        "cumulative": evaluator,
        "wallclock": elapsed_wallclock_time,
        "cpu_time": elapsed_cpu_time,
        "max_instances":max_instances, 
        "stream":stream,
    }

    for measure in moa_results.otherMeasurements.keySet():
        measure_str = str(measure)  # Convert Java key to a Python string
        results[measure_str] = float(
            moa_results.otherMeasurements.get(measure)
        )  # Get the Java Double value

    return results


def prequential_SSL_evaluation_fast(
    stream,
    learner,
    max_instances=None,
    window_size=1000,
    initial_window_size=0,
    delay_length=0,
    label_probability=0.01,
    random_seed=1,
):
    """
    Prequential SSL evaluation fast.
    """
    if isinstance(stream, NumpyStream):
        return prequential_SSL_evaluation(stream, learner, max_instances, window_size)

    if max_instances is None:
        max_instances = -1

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    basic_evaluator = ClassificationEvaluator(schema=stream.get_schema())
    # Always create the windowed_evaluator, even if window_size is None. TODO: may want to avoid creating it if window_size is None.
    windowed_evaluator = ClassificationWindowedEvaluator(
        schema=stream.get_schema(), window_size=window_size
    )

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
    )

    # Reset the windowed_evaluator result_windows
    if moa_results != None:
        windowed_evaluator.result_windows = []
        if moa_results.windowedResults != None:
            for entry_idx in range(len(moa_results.windowedResults)):
                windowed_evaluator.result_windows.append(
                    moa_results.windowedResults[entry_idx]
                )

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    results = {
        "learner": str(learner),
        "cumulative": basic_evaluator,
        "windowed": windowed_evaluator,
        "wallclock": elapsed_wallclock_time,
        "cpu_time": elapsed_cpu_time,
        "max_instances":max_instances, 
        "stream":stream,
        "other_measurements": dict(moa_results.otherMeasurements),
    }

    return results


########################################################################################
###### EXPERIMENTAL (optimisation to go over the data once for several learners)  ######
########################################################################################


# TODO: review if we want to keep this method.
def prequential_evaluation_multiple_learners(
    stream, learners, max_instances=None, window_size=1000
):
    """
    Calculates the metrics cumulatively (i.e., test-then-train) and in a window-fashion (i.e., windowed prequential evaluation) for multiple streams and learners.
    Returns the results in a dictionary format. Infers whether it is a Classification or Regression problem based on the stream schema.
    """
    results = {}

    if stream.has_more_instances() == False:
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
            results[learner_name]["cumulative"] = RegressionEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
            if window_size is not None:
                results[learner_name]["windowed"] = RegressionWindowedEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
    instancesProcessed = 1

    while stream.has_more_instances() and (
        max_instances is None or instancesProcessed <= max_instances
    ):
        instance = stream.next_instance()

        for learner_name, learner in learners.items():
            # Predict for the current learner
            prediction = learner.predict(instance)

            results[learner_name]["cumulative"].update(instance.y(), prediction)
            if window_size is not None:
                results[learner_name]["windowed"].update(instance.y(), prediction)

            learner.train(instance)

        instancesProcessed += 1

    # Iterate through the results of each learner and add (if needed) the last window of results to it.
    if window_size is not None:
        for learner_name, result in results.items():
            if result["windowed"].get_instances_seen() % window_size != 0:
                result["windowed"].result_windows.append(result["windowed"].metrics())

    return results


# # USAGE EXAMPLES USING MOA LEARNERS
# from moa.classifiers.meta import AdaptiveRandomForest
# from moa.streams import ArffFileStream

# def example_ARF_on_RTG_2abrupt_with_TestThenTrain(dataset_path="/Users/gomeshe/Desktop/data/RTG_2abrupt.arff"):
#     arf10 = AdaptiveRandomForest()
#     arf10.getOptions().setViaCLIString("-s 10")
#     arf10.setRandomSeed(1)
#     arf10.prepareForUse()

#     sampleFrequency = 100

#     rtg_2abrupt = ArffFileStream(dataset_path, -1)
#     rtg_2abrupt.prepareForUse()

#     acc, wallclock, cpu_time, df = test_then_train(rtg_2abrupt, arf10, max_instances=2000, sample_frequency=sampleFrequency)

#     print(f"Test-Then-Train evaluation. Final accuracy: {acc:.4f}, Wallclock: {wallclock:.4f}, CPU Time: {cpu_time:.4f}")
#     print(df.to_string())

# def example_ARF_on_RTG_2abrupt_with_Prequential(dataset_path="/Users/gomeshe/Desktop/data/RTG_2abrupt.arff"):
#     arf10 = AdaptiveRandomForest()
#     arf10.getOptions().setViaCLIString("-s 10")
#     arf10.setRandomSeed(1)
#     arf10.prepareForUse()

#     sampleFrequency = 100

#     rtg_2abrupt = ArffFileStream(dataset_path, -1)
#     rtg_2abrupt.prepareForUse()

#     wallclock, cpu_time, df = prequential(rtg_2abrupt, arf10, max_instances=2000, window_size=sampleFrequency)

#     print(f"Prequential evaluation. Wallclock: {wallclock:.4f}, CPU Time: {cpu_time:.4f}")
#     print(df.to_string())

# if __name__ == "__main__":
#     print('example_ARF_on_RTG_2abrupt_with_TestThenTrain()')
#     example_ARF_on_RTG_2abrupt_with_TestThenTrain()
#     print('example_ARF_on_RTG_2abrupt_with_Prequential')
#     example_ARF_on_RTG_2abrupt_with_Prequential()
