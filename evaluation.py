import pandas as pd
import time
import datetime as dt
import resource
import os

# Create the JVM and add the MOA jar to the classpath
import prepare_jpype

# MOA imports
from moa.classifiers.meta import AdaptiveRandomForest
from moa.core import Example
from moa.evaluation import BasicClassificationPerformanceEvaluator, WindowClassificationPerformanceEvaluator
from moa.streams import ArffFileStream

## River imports
from river import stream
from river import metrics

## Functions fo measuring runtime
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

## Function to abstract the test and train loop using MOA
def test_train_loop_MOA(stream, learner, maxInstances=1000, sampleFrequency=100, evaluator=BasicClassificationPerformanceEvaluator()):
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    
    instancesProcessed = 1
    learner.setModelContext(stream.getHeader())

    # evaluator.recallPerClassOption.set()
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
def test_train_loop_RIVER(model, dataset, maxInstances=1000, sampleFrequency=100):
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