import pandas as pd
import time
import datetime as dt
import resource
import os
import random

# Create the JVM and add the MOA jar to the classpath
import prepare_jpype

# MOA/Java imports
from com.yahoo.labs.samoa.instances import Instances, Instance, Attribute, DenseInstance
from moa.core import Example, InstanceExample
from moa.evaluation import BasicClassificationPerformanceEvaluator, WindowClassificationPerformanceEvaluator
from moa.streams import ArffFileStream
from java.util import ArrayList

## River imports
from river import stream
from river import metrics


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


class Schema:
    """
    This class is a wrapper for the MOA header, but it can be set up in Python directly by specifying the labels attribute.
    If moa_header is specified, then it overrides everything else. 
    In the future, we might want to have a way to specify the values for nominal attributes as well, so far just the labels. 
    """
    def __init__(self, moa_header=None, labels=None):   
        self.moa_header = moa_header
        self.label_values = labels
        
        if self.moa_header is not None:
            self.label_values = [g for g in self.moa_header.outputAttribute(1).getAttributeValues()]
            
    def getLabelValues(self):
        if self.label_values is None:
            return []
        else:
            return self.label_values

    def getLabelIndexes(self):
        if self.label_values is None:
            return []
        else:
            return list(range(len(self.label_values)))

    def getMoaHeader(self):
        return self.moa_header


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
        if self.schema is not None:
            self.pred_template = [0] * len(self.schema.getLabelIndexes())
            for value in self.schema.getLabelIndexes():
                _attributeValues.append(value)
        else:
            raise ValueError("Schema cannot be None, please define a schema.")
        
        _classAttribute = Attribute("Class", _attributeValues)
        
        attSub = ArrayList()
        attSub.append(_classAttribute)
        # If there is no moa_header specified, then it will create one
        self._header = None
        if self.schema is not None and self.schema.getMoaHeader() is not None:
            self._header = self.schema.getMoaHeader()
        else:
            self._header = Instances("", attSub, 1)
            self._header.setClassIndex(0);

        # Create the denseInstance just once and keep reusing it by changing the classValue (more efficient). 
        self._instance = DenseInstance(1.0, [0])
        self._instance.setDataset(self._header)
    
    def update(self, y, y_pred):
        if self.schema.getLabelIndexes() is not None:
            # Check if y is a value present in the labels array, if yes, then convert it to a index on that list. 
            if y in self.schema.getLabelValues():
                y = self.schema.getLabelValues().index(y)
            
        # Notice, in MOA the class value is a index, not the actual value (e.g. not "one" but 0 assuming labels=["one", "two"])
        self._instance.setClassValue(y)
        example = InstanceExample(self._instance)
        
        # Shallow copy of the pred_template
        # MOA evaluator accepts the result of getVotesForInstance which is similar to a predict_proba 
        #    (may or may not be normalised, but for our purposes it doesn't matter)
        prediction_array = self.pred_template[:]
        
        # if y_pred is None, it indicates the learner absented from predicting this instace, count as an error
        if y_pred is None:
            # Set y_pred to any valid prediction that is not y (force an incorrect prediction)
            # This does not affect recall or any other metrics, because the selected value is always incorrect. 

            # Create an intermediary array with indices excluding the y
            indexesWithoutY = [i for i in range(len(self.schema.getLabelIndexes())) if i != y]
            random_y_pred = random.choice(indexesWithoutY)
            y_pred = self.schema.getLabelIndexes()[random_y_pred]
        else: 
            # y_pred is not None... 
            if self.schema.getLabelIndexes() is not None:
                # Find the index of the corresponding value (it y_pred is a value instead of an index). 
                if y_pred in self.schema.getLabelValues():
                    y_pred = self.schema.getLabelValues().index(y_pred)
        
        prediction_array[y_pred] += 1        
        self.moa_basic_evaluator.addResult(example, prediction_array)

        self.instances_seen += 1
        
        if self.window_size is not None and self.instances_seen % self.window_size == 0:
            performance_values = [measurement.getValue() for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()]
            self.result_windows.append(performance_values)

    def measurements(self):
        performance_measurements = self.moa_basic_evaluator.getPerformanceMeasurements()
        performance_names = ["".join(measurement.getName()) for measurement in performance_measurements]
        return performance_names

    def results(self):
        return [measurement.getValue() for measurement in self.moa_basic_evaluator.getPerformanceMeasurements()]

    def windowed_results(self):
        return pd.DataFrame(self.result_windows, columns=self.measurements())
    
    def accuracy(self):
        index = self.measurements().index('classifications correct (percent)')
        return self.results()[index]

    def kappa(self):
        index = self.measurements().index('Kappa Statistic (percent)')
        return self.results()[index]

    def kappaT(self):
        index = self.measurements().index('Kappa Temporal Statistic (percent)')
        return self.results()[index]

    def kappaM(self):
        index = self.measurements().index('Kappa M Statistic (percent)')
        return self.results()[index]


class ClassificationWindowedEvaluator(ClassificationEvaluator):
    def __init__(self, schema=None, window_size=1000, recall_per_class=False, precision_per_class=False, 
                 f1_precision_recall=False, f1_per_class=False):
        self.moa_evaluator = WindowClassificationPerformanceEvaluator()
        self.moa_evaluator.widthOption.setValue(window_size)
        
        super().__init__(schema=schema, window_size=window_size, recall_per_class=recall_per_class, 
                         precision_per_class=precision_per_class, f1_precision_recall=f1_precision_recall, 
                         f1_per_class=f1_per_class, moa_evaluator=self.moa_evaluator)


## Function to abstract the test and train loop using MOA
def test_train_loop_MOA(stream, learner, maxInstances=1000, sampleFrequency=100, evaluator=None):
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    
    instancesProcessed = 1
    learner.setModelContext(stream.getHeader())

    # evaluator.recallPerClassOption.set()
    if evaluator is None:
        evaluator = BasicClassificationPerformanceEvaluator()
        evaluator.prepareForUse()
    
    data = []
    performance_names = []
    performance_values = []
    # accuracy = 0
    
    while stream.hasMoreInstances() and instancesProcessed <= maxInstances:
        trainInst = stream.nextInstance()
        testInst = trainInst
    
        prediction = learner.getVotesForInstance(testInst)
        evaluator.addResult(testInst, prediction)
        learner.trainOnInstance(trainInst)
    
        if instancesProcessed == 1:
            performance_measurements = evaluator.getPerformanceMeasurements()
            performance_names = ["".join(measurement.getName()) for measurement in performance_measurements]
    
        if instancesProcessed % sampleFrequency == 0:
            performance_values = [measurement.getValue() for measurement in evaluator.getPerformanceMeasurements()]
            data.append(performance_values)

        instancesProcessed += 1

    accuracy = evaluator.getPerformanceMeasurements()[1].getValue()
    
    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(start_wallclock_time, start_cpu_time)

    # {<classifier>.getClass().getName(), <classifier>.getOptions().getAsCLIString()}
    return accuracy, elapsed_wallclock_time, elapsed_cpu_time, pd.DataFrame(data, columns=performance_names)


## Function to abstract the test and train loop using RIVER
def test_train_loop_RIVER(dataset, model, maxInstances=1000, sampleFrequency=100):
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    instancesProcessed = 1
    accuracy = metrics.Accuracy()
    
    X, Y = dataset[:, :-1], dataset[:, -1]
    
    data = []
    performance_names = ['Classified instances', 'accuracy']
    performance_values = []

    ds = stream.iter_array(X, Y)
    
    for (x, y) in ds:
        if instancesProcessed > maxInstances:
            break
        yp = model.predict_one(x)
        accuracy.update(y, yp)
        model.learn_one(x, y)

        if instancesProcessed % sampleFrequency == 0:
            performance_values = [instancesProcessed, accuracy.get()]
            data.append(performance_values)

        instancesProcessed += 1

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(start_wallclock_time, start_cpu_time)

    return accuracy.get(), elapsed_wallclock_time, elapsed_cpu_time, pd.DataFrame(data, columns=performance_names)


def example_ARF_on_RTG_2abrupt_with_WindowClassificationEvaluator(dataset_path="/Users/gomeshe/Desktop/data/RTG_2abrupt.arff"):
	arf10 = AdaptiveRandomForest()
	arf10.getOptions().setViaCLIString("-s 10")
	arf10.setRandomSeed(1)
	arf10.prepareForUse()

	sampleFrequency = 100

	rtg_2abrupt = ArffFileStream(dataset_path, -1)
	rtg_2abrupt.prepareForUse()

	evaluator = WindowClassificationPerformanceEvaluator()
	evaluator.widthOption.setValue(sampleFrequency)
	# evaluator.recallPerClassOption.set()
	evaluator.prepareForUse()

	# acc, df, wallclock = test_train_loop_MOA(rtg_2abrupt, arf10, evaluator, maxInstances=10000, sampleFrequency=10000)

	acc, wallclock, cpu_time, df = test_train_loop_MOA(rtg_2abrupt, arf10, maxInstances=2000, sampleFrequency=sampleFrequency, evaluator=evaluator)

	print(f"Last accuracy: {acc:.4f}, Wallclock: {wallclock:.4f}, CPU Time: {cpu_time:.4f}")
	print(df.to_string())


def example_ARF_on_RTG_2abrupt_with_BasicClassificationEvaluator(dataset_path="/Users/gomeshe/Desktop/data/RTG_2abrupt.arff"):
	arf10 = AdaptiveRandomForest()
	arf10.getOptions().setViaCLIString("-s 10")
	arf10.setRandomSeed(1)
	arf10.prepareForUse()

	sampleFrequency = 100

	rtg_2abrupt = ArffFileStream(dataset_path, -1)
	rtg_2abrupt.prepareForUse()

	evaluator = BasicClassificationPerformanceEvaluator()
	# evaluator.recallPerClassOption.set()
	evaluator.prepareForUse()

	# acc, df, wallclock = test_train_loop_MOA(rtg_2abrupt, arf10, evaluator, maxInstances=10000, sampleFrequency=10000)

	acc, wallclock, cpu_time, df = test_train_loop_MOA(rtg_2abrupt, arf10, maxInstances=2000, sampleFrequency=sampleFrequency, evaluator=evaluator)

	print(f"Last accuracy: {acc:.4f}, Wallclock: {wallclock:.4f}, CPU Time: {cpu_time:.4f}")
	print(df.to_string())

if __name__ == "__main__":
	print('example_ARF_on_RTG_2abrupt_with_WindowClassificationEvaluator()')
	example_ARF_on_RTG_2abrupt_with_WindowClassificationEvaluator()
	print('example_ARF_on_RTG_2abrupt_with_BasicClassificationEvaluator()')
	example_ARF_on_RTG_2abrupt_with_BasicClassificationEvaluator()