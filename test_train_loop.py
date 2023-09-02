import pandas as pd
import subprocess
import jpype

import time
import datetime as dt

import jpype.imports
from jpype.types import *

# Starts the JVM
jpype.startJVM()

# Add the moa jar to the class path
jpype.addClassPath('/Users/gomeshe/Dropbox/ciencia_computacao/dev/Using-MOA-API/moa.jar')

from moa.classifiers.meta import AdaptiveRandomForest
from moa.core import Example
from moa.evaluation import BasicClassificationPerformanceEvaluator
from moa.streams import ArffFileStream


## Function to abstract the test and train loop
## The JVM must have been started already
def test_train_loop(stream, learner, evaluator, maxInstances=1000, sampleFrequency=100):
	start = time.perf_counter()
	instancesProcessed = 1

	learner.setModelContext(stream.getHeader())
	
	data = []
	performance_names = []
	performance_values = []
	
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

		accuracy = evaluator.getPerformanceMeasurements()[1]

		instancesProcessed += 1
	
	return accuracy, pd.DataFrame(data, columns=performance_names), dt.timedelta(seconds=time.perf_counter() - start).seconds


if __name__ == "__main__":
	# If JAVA_HOME is not set, then jpype will fail. 
	subprocess.call("ECHO $JAVA_HOME", shell=True)

	arf10 = AdaptiveRandomForest()
	arf10.getOptions().setViaCLIString("-s 10")
	arf10.setRandomSeed(1)
	arf10.prepareForUse()

	rtg_2abrupt = ArffFileStream("/Users/gomeshe/Desktop/data/RTG_2abrupt.arff", -1)
	rtg_2abrupt.prepareForUse()

	evaluator = BasicClassificationPerformanceEvaluator()
	evaluator.recallPerClassOption.set()
	evaluator.prepareForUse()

	acc, df, wallclock = test_train_loop(rtg_2abrupt, arf10, evaluator, maxInstances=10000, sampleFrequency=10000)

	print('wall clock: ', wallclock)
	print(acc)
	print(df.to_string())