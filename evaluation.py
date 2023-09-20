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
from moa.evaluation import BasicClassificationPerformanceEvaluator, WindowClassificationPerformanceEvaluator
from moa.streams import ArffFileStream
from java.util import ArrayList


class Schema:
    """
    This class is a wrapper for the MOA header, but it can be set up in Python directly by specifying the labels attribute.
    If moa_header is specified, then it overrides everything else. 
    In the future, we might want to have a way to specify the values for nominal attributes as well, so far just the labels. 
    """
    def __init__(self, moa_header=None, labels=None): 
        self.moa_header = moa_header
        self.label_values = labels
        self.label_indexes = None
        
        if self.moa_header is not None:
            # Important: a Java.String is different from a Python str, so it is important to str(*) before storing the values.
            self.label_values = [str(g) for g in self.moa_header.outputAttribute(1).getAttributeValues()]
            
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
        
        # if y_pred is None, it indicates the learner absented from predicting this instace, count as an error
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


def test_then_train(stream, learner, max_instances=1000, sample_frequency=100, evaluator=None):
    '''
    Test-then-train evaluation using a MOA learner. 
    '''
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    
    instancesProcessed = 1
    learner.setModelContext(stream.getHeader())

    if evaluator is None:
        schema = Schema(moa_header=stream.getHeader())
        evaluator = ClassificationEvaluator(schema=schema, window_size=sample_frequency)
    
    while stream.hasMoreInstances() and instancesProcessed <= max_instances:
        trainInst = stream.nextInstance()
        testInst = trainInst
    
        prediction = learner.getVotesForInstance(testInst)
        evaluator.update(testInst.getData().classValue(), Utils.maxIndex(prediction))
        learner.trainOnInstance(trainInst)

        instancesProcessed += 1
    
    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(start_wallclock_time, start_cpu_time)

    return evaluator.accuracy(), elapsed_wallclock_time, elapsed_cpu_time, evaluator.metrics_per_window()


def prequential(stream, learner, max_instances=1000, window_size=100):
    '''
    Prequential evaluation (window) using a MOA learner. 
    '''
    schema = Schema(moa_header=stream.getHeader())
    evaluator = ClassificationWindowedEvaluator(schema=schema, window_size=window_size)
    
    _acc, wallclock, cpu_time, windowed_matrics = test_then_train(stream, learner, max_instances=max_instances, sample_frequency=window_size, evaluator=evaluator)

    # Ignore the last accuracy, because it doesn't matter (it is only useful for test-then-train). 
    return wallclock, cpu_time, windowed_matrics



# USAGE EXAMPLES USING MOA LEARNERS #
from moa.classifiers.meta import AdaptiveRandomForest

def example_ARF_on_RTG_2abrupt_with_TestThenTrain(dataset_path="/Users/gomeshe/Desktop/data/RTG_2abrupt.arff"):
    arf10 = AdaptiveRandomForest()
    arf10.getOptions().setViaCLIString("-s 10")
    arf10.setRandomSeed(1)
    arf10.prepareForUse()

    sampleFrequency = 100

    rtg_2abrupt = ArffFileStream(dataset_path, -1)
    rtg_2abrupt.prepareForUse()

    acc, wallclock, cpu_time, df = test_then_train(rtg_2abrupt, arf10, max_instances=2000, sample_frequency=sampleFrequency)

    print(f"Test-Then-Train evaluation. Final accuracy: {acc:.4f}, Wallclock: {wallclock:.4f}, CPU Time: {cpu_time:.4f}")
    print(df.to_string())

def example_ARF_on_RTG_2abrupt_with_Prequential(dataset_path="/Users/gomeshe/Desktop/data/RTG_2abrupt.arff"):
    arf10 = AdaptiveRandomForest()
    arf10.getOptions().setViaCLIString("-s 10")
    arf10.setRandomSeed(1)
    arf10.prepareForUse()

    sampleFrequency = 100

    rtg_2abrupt = ArffFileStream(dataset_path, -1)
    rtg_2abrupt.prepareForUse()

    wallclock, cpu_time, df = prequential(rtg_2abrupt, arf10, max_instances=2000, window_size=sampleFrequency)

    print(f"Prequential evaluation. Wallclock: {wallclock:.4f}, CPU Time: {cpu_time:.4f}")
    print(df.to_string())

if __name__ == "__main__":
    print('example_ARF_on_RTG_2abrupt_with_TestThenTrain()')
    example_ARF_on_RTG_2abrupt_with_TestThenTrain()
    print('example_ARF_on_RTG_2abrupt_with_Prequential')
    example_ARF_on_RTG_2abrupt_with_Prequential()