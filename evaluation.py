import pandas as pd
import time
import datetime as dt
import resource
import os
import random

# Create the JVM and add the MOA jar to the classpath
from prepare_jpype import start_jpype
start_jpype()

# MOA/Java imports
from com.yahoo.labs.samoa.instances import Instances, Instance, Attribute, DenseInstance
from moa.core import Example, InstanceExample, Utils
from moa.evaluation import BasicClassificationPerformanceEvaluator, WindowClassificationPerformanceEvaluator, BasicRegressionPerformanceEvaluator, WindowRegressionPerformanceEvaluator
from java.util import ArrayList

# IDEA: STATIC METHOD TO CREATE A SCHEMA USING A MOA_HEADER. (e.g. withMOAHeader...)


# TODO: Move Schema to stream.py
class Schema:
    """
    This class is a wrapper for the MOA header, but it can be set up in Python directly by specifying the labels attribute.
    If moa_header is specified, then it overrides everything else. 
    In the future, we might want to have a way to specify the values for nominal attributes as well, so far just the labels. 
    The number of attributes is instrumental for Evaluators that need it, such as adjusted coefficient of determination. 
    """
    def __init__(self, moa_header=None, labels=None, num_attributes=1): 
        self.moa_header = moa_header
        self.label_values = labels
        self.label_indexes = None
        # Internally, we store the number of attributes + the class/target. This is because MOA methods expect the numAttributes 
        # to also account for the class/target.
        self.num_attributes_including_output = num_attributes+1
        
        self.regression = False;
        if self.moa_header is not None:
            # TODO: might want to iterate over the other attributes and create a dictionary representation for the nominal attributes.
            # There should be a way to configure that manually like setting the self.labels instead of using a MOA header.
            if self.moa_header.outputAttribute(1).isNominal():
                # Important: a Java.String is different from a Python str, so it is important to str(*) before storing the values.
                self.label_values = [str(g) for g in self.moa_header.outputAttribute(1).getAttributeValues()]
            else:
                # This is a regression task, there are no label values. 
                self.regression = True;
            # The numAttributes in MOA also account for the class label. 
            self.num_attributes_including_output = self.moa_header.numAttributes()
        # else logic: the label_values must be set, so that the first time the get_label_indexes is invoked, they are correctly created. 

    def get_label_values(self):
        if self.label_values is None:
            return None
        else:
            return self.label_values

    def get_label_indexes(self):
        if self.label_values is None:
            return None
        else:
            if self.label_indexes is None:
                self.label_indexes = list(range(len(self.label_values)))
            return self.label_indexes

    def get_moa_header(self):
        return self.moa_header

    def get_num_attributes(self):
        # ignoring the class/target value. 
        return self.num_attributes_including_output-1

    def get_valid_index_for_label(self, y):
        if self.label_indexes is None:
            raise ValueError("Schema was not properly initialised, please define a proper Schema.")

        # print(f"get_valid_index_for_label( y = {y} )")
        
        # Check of y is a string and if the labelValues contains strings. 
        # print(f"isinstance {type(y)}, {type(self.label_values[0])}")
        if isinstance(y, type(self.label_values[0])):
            if y in self.label_values:
                return self.label_values.index(y)

        # If it is not a valid value, then maybe it is an index
        if y in self.label_indexes:
            return y

        # This is neither a valid label value nor a valid index. 
        return None

    def is_regression(self):
        return self.regression

    def is_classification(self):
        return not self.regression


class ClassificationEvaluator: 
    """
    Wrapper for the Classification Performance Evaluator from MOA. By default uses the BasicClassificationPerformanceEvaluator
    """
    def __init__(self, schema=None, window_size=None, recall_per_class=False, precision_per_class=False, 
                 f1_precision_recall=False, f1_per_class=False, moa_evaluator=None):
        self.instances_seen = 0
        self.result_windows = []
        self.window_size = window_size
        
        self.moa_basic_evaluator = moa_evaluator
        if self.moa_basic_evaluator is None:
            self.moa_basic_evaluator = BasicClassificationPerformanceEvaluator()
        
        if recall_per_class: 
            self.moa_basic_evaluator.recallPerClassOption.set()
        if precision_per_class: 
            self.moa_basic_evaluator.precisionPerClassOption.set()
        if f1_precision_recall: 
            self.moa_basic_evaluator.precisionRecallOutputOption.set()
        if f1_per_class: 
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
                self._header.setClassIndex(0);
            else:
                raise ValueError("Schema was not initialised properly, please define a proper Schema.")
        else:
            raise ValueError("Schema is None, please define a proper Schema.")
        
        self.pred_template = [0] * len(self.schema.get_label_indexes())

        # Create the denseInstance just once and keep reusing it by changing the classValue (more efficient). 
        self._instance = DenseInstance(1)
        self._instance.setDataset(self._header)
    
    def __str__(self):
        return str({header: value for header, value in zip(self.metrics_header(), self.metrics())})
    
    def update(self, y, y_pred):
        
        # Check if the schema is valid. 
        y_index = self.schema.get_valid_index_for_label(y)
        y_pred_index = self.schema.get_valid_index_for_label(y_pred)

        # print(f"y_index = {y_index} y_pred_index = {y_pred_index}")
        
        if y_index is None:
            raise ValueError(f"Invalid ground-truth (y) value {y}")
            
        # Notice, in MOA the class value is a index, not the actual value (e.g. not "one" but 0 assuming labels=["one", "two"])
        
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
            indexesWithoutY = [i for i in range(len(self.schema.get_label_indexes())) if i != y_index]
            random_y_pred = random.choice(indexesWithoutY)
            y_pred_index = self.schema.get_label_indexes()[random_y_pred]
        
        prediction_array[int(y_pred_index)] += 1
        self.moa_basic_evaluator.addResult(example, prediction_array)

        self.instances_seen += 1

        # If the window_size is set, then check if it should record the intermediary results. 
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            performance_values = [measurement.getValue() for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()]
            self.result_windows.append(performance_values)

    def metrics_header(self):
        performance_measurements = self.moa_basic_evaluator.getPerformanceMeasurements()
        performance_names = ["".join(measurement.getName()) for measurement in performance_measurements]
        return performance_names

    def metrics(self):
        return [measurement.getValue() for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()]

    def metrics_per_window(self):
        return pd.DataFrame(self.result_windows, columns=self.metrics_header())
    
    def accuracy(self):
        index = self.metrics_header().index('classifications correct (percent)')
        return self.metrics()[index]

    def kappa(self):
        index = self.metrics_header().index('Kappa Statistic (percent)')
        return self.metrics()[index]

    def kappa_temporal(self):
        index = self.metrics_header().index('Kappa Temporal Statistic (percent)')
        return self.metrics()[index]

    def kappa_M(self):
        index = self.metrics_header().index('Kappa M Statistic (percent)')
        return self.metrics()[index]


class ClassificationWindowedEvaluator(ClassificationEvaluator):
    def __init__(self, schema=None, window_size=1000, recall_per_class=False, precision_per_class=False, 
                 f1_precision_recall=False, f1_per_class=False):
        self.moa_evaluator = WindowClassificationPerformanceEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)
        
        super().__init__(schema=schema, window_size=window_size, recall_per_class=recall_per_class, 
                         precision_per_class=precision_per_class, f1_precision_recall=f1_precision_recall, 
                         f1_per_class=f1_per_class, moa_evaluator=self.moa_evaluator)



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
                self._header.setClassIndex(self.schema.get_num_attributes());
                # print(self._header)
            else:
                raise ValueError("Schema was not set for a regression task")
        else:
            raise ValueError("Schema is None, please define a proper Schema.")
        
        # Regression has only one output
        self.pred_template = [0]

        # Create the denseInstance just once and keep reusing it by changing the classValue (more efficient). 
        self._instance = DenseInstance(self.schema.get_num_attributes()+1)
        self._instance.setDataset(self._header)
    
    def update(self, y, y_pred):

        if y is None:
            raise ValueError(f"Invalid ground-truth y = {y}")

        self._instance.setClassValue(y)
        example = InstanceExample(self._instance)
        
        # if y_pred is None, it indicates the learner did not produce a prediction for this instace
        if y_pred is None:
            # In classification it is rather easy to deal with this, but 

            # Create an intermediary array with indices excluding the y
            indexesWithoutY = [i for i in range(len(self.schema.get_label_indexes())) if i != y_index]
            random_y_pred = random.choice(indexesWithoutY)
            y_pred_index = self.schema.get_label_indexes()[random_y_pred]
        
        # Different from classification, there is no need to make a shallow copy of the prediction array, just override the value. 
        self.pred_template[0] = y_pred
        self.moa_basic_evaluator.addResult(example, self.pred_template)

        self.instances_seen += 1

        # If the window_size is set, then check if it should record the intermediary results. 
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            performance_values = [measurement.getValue() for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()]
            self.result_windows.append(performance_values)

    def metrics_header(self):
        performance_measurements = self.moa_basic_evaluator.getPerformanceMeasurements()
        performance_names = ["".join(measurement.getName()) for measurement in performance_measurements]
        return performance_names

    def metrics(self):
        return [measurement.getValue() for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()]

    def metrics_per_window(self):
        return pd.DataFrame(self.result_windows, columns=self.metrics_header())
    
    def MAE(self):
        index = self.metrics_header().index('mean absolute error')
        return self.metrics()[index]

    def RMSE(self):
        index = self.metrics_header().index('root mean squared error')
        return self.metrics()[index]

    def RMAE(self):
        index = self.metrics_header().index('relative mean absolute error')
        return self.metrics()[index]

    def R2(self):
        index = self.metrics_header().index('coefficient of determination')
        return self.metrics()[index]

    def adjusted_R2(self):
        index = self.metrics_header().index('adjusted coefficient of determination')
        return self.metrics()[index]


class RegressionWindowedEvaluator(RegressionEvaluator):
    def __init__(self, schema=None, window_size=1000):
        self.moa_evaluator = WindowRegressionPerformanceEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)
        
        super().__init__(schema=schema, window_size=window_size, moa_evaluator=self.moa_evaluator)


## Functions to measure runtime
def start_time_measuring(): 
    start_wallclock_time = time.time()
    start_cpu_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime

    return start_wallclock_time, start_cpu_time



def stop_time_measuring(start_wallclock_time, start_cpu_time):
    # Stop measuring time
    end_wallclock_time = time.time()
    end_cpu_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime
    
    # Calculate and print the elapsed time and CPU times
    elapsed_wallclock_time = end_wallclock_time - start_wallclock_time
    elapsed_cpu_time = end_cpu_time - start_cpu_time

    return elapsed_wallclock_time, elapsed_cpu_time


def test_then_train_evaluation(stream, learner, max_instances=None, sample_frequency=100, evaluator=None):
    '''
    Test-then-train evaluation using a MOA learner. 
    '''
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    instancesProcessed = 1

    if evaluator is None:
        schema = stream.get_schema()
        if schema.is_classification():
            evaluator = ClassificationEvaluator(schema=schema, window_size=sample_frequency)
        else:
            evaluator = RegressionEvaluator(schema=schema, window_size=sample_frequency)
    
    while stream.has_more_instances() and (max_instances is None or instancesProcessed <= max_instances):
        instance = stream.next_instance()
    
        prediction = learner.predict(instance)
        evaluator.update(instance.y(), prediction)
        learner.train(instance)

        instancesProcessed += 1
    
    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(start_wallclock_time, start_cpu_time)

    results = {'learner':str(learner), 'cumulative':evaluator, 
               'wallclock':elapsed_wallclock_time, 'cpu_time':elapsed_cpu_time}
    
    return results

def windowed_evaluation(stream, learner, max_instances=None, window_size=1000):
    '''
    Prequential evaluation (window) using a MOA learner. 
    '''
    # Run test-then-train evaluation, but change the underlying evaluator
    evaluator = None
    if stream.get_schema().is_classification():
        evaluator = ClassificationWindowedEvaluator(schema=stream.get_schema(), window_size=window_size)
    else:
        evaluator = RegressionWindowedEvaluator(schema=stream.get_schema(), window_size=window_size)
    results = test_then_train_evaluation(stream, learner, max_instances=max_instances, sample_frequency=window_size, evaluator=evaluator)

    results['windowed'] = results['cumulative']
    results.pop('cumulative', None) # Remove previous entry with the cumulative results. 

    # Ignore the last prediction values, because it doesn't matter as we are using a windowed evaluation.
    return results


def prequential_evaluation(stream, learner, max_instances=None, window_size=1000):
    '''
    Calculates the metrics cumulatively (i.e. test-then-train) and in a window-fashion (i.e. windowed prequential evaluation). 
    Returns both evaluators so that the caller has access to metric from both evaluators. 
    '''
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    instancesProcessed = 1
    
    evaluator_cumulative = None
    evaluator_windowed = None
    if stream.get_schema().is_classification:
        evaluator_cumulative = ClassificationEvaluator(schema=stream.get_schema(), window_size=window_size)
        evaluator_windowed = ClassificationWindowedEvaluator(schema=stream.get_schema(), window_size=window_size)
    else:
        evaluator_cumulative = RegressionEvaluator(schema=stream.get_schema(), window_size=window_size)
        evaluator_windowed = RegressionWindowedEvaluator(schema=stream.get_schema(), window_size=window_size)

    while stream.has_more_instances() and (max_instances is None or instancesProcessed <= max_instances):
        instance = stream.next_instance()

        prediction = learner.predict(instance)

        evaluator_cumulative.update(instance.y(), prediction)
        evaluator_windowed.update(instance.y(), prediction)
        learner.train(instance)

        instancesProcessed += 1

    # # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(start_wallclock_time, start_cpu_time)

    results = {'learner':str(learner), 'cumulative':evaluator_cumulative, 'windowed': evaluator_windowed, 
               'wallclock':elapsed_wallclock_time, 'cpu_time':elapsed_cpu_time}
    
    return results 
    # return evaluator_cumulative, evaluator_windowed, elapsed_wallclock_time, elapsed_cpu_time

def prequential_evaluation_multiple_learners(stream, learners, max_instances=None, window_size=1000):
    '''
    Calculates the metrics cumulatively (i.e., test-then-train) and in a window-fashion (i.e., windowed prequential evaluation) for multiple streams and learners. 
    Returns the results in a dictionary format. Infers whether it is a Classification or Regression problem based on the stream schema. 
    '''
    results = {}
    for learner_name, learner in learners.items():
        results[learner_name] = {'learner':str(learner)}
    
    for learner_name, learner in learners.items():
        if stream.get_schema().is_classification():
            results[learner_name]['cumulative'] = ClassificationEvaluator(schema=stream.get_schema(), window_size=window_size)
            results[learner_name]['windowed'] = ClassificationWindowedEvaluator(schema=stream.get_schema(), window_size=window_size)
        else:
            results[learner_name]['cumulative'] = RegressionEvaluator(schema=stream.get_schema(), window_size=window_size)
            results[learner_name]['windowed'] = RegressionWindowedEvaluator(schema=stream.get_schema(), window_size=window_size)
    instancesProcessed = 1
    while stream.has_more_instances() and (max_instances is None or instancesProcessed <= max_instances):
        instance = stream.next_instance()
        
        for learner_name, learner in learners.items():
            # Predict for the current learner
            prediction = learner.predict(instance)

            results[learner_name]['cumulative'].update(instance.y(), prediction)
            results[learner_name]['windowed'].update(instance.y(), prediction)
            
            learner.train(instance)
            
        instancesProcessed += 1
        
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