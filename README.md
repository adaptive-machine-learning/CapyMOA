# MOABridge
Python wrapper for MOA to allow efficient use of existing algorithms with a more modern API

To use the project
1. Set the config.ini to point to the correct path of the moa.jar
2. Make sure JAVA_HOME is set
3. The requirements are missing river and jpype
4. ```pip install jpype1``` (yes, there is a 1 there), and to run the river experiments you need to ```pip install river```


# TODO (updated 12/10/2023)


1. [Doc] ```*evaluation_``` functions from ```evaluation``` package always invoke ```restart()``` on the streams (guarantee no evaluation is executed in a stream that has already been partially processed). 
2. [Tests/Examples] Add tests and examples using regression for prequential, test-then-train and windowed. Notebook: 
3. [New] Add prequential multiple streams and learners (will need to restart streams, allow us to provide `fair` timing information)
4. ~[Examples] Create a simple Demo notebook for Classification and Regression.~
5. ~[Tests/Examples] Double check if all notebooks still work~
6. [New] Create the module structure.  
7. [New] Override the ```__str__``` or create a ```describe()``` method to return the learner+key hyperparameters (for plotting and identification purposes)
8. [New] Create ```__str__``` for the ```Stream()``` classes (to identify the dataset). 
9. [Doc] ```windowed_evaluation``` and ```prequential_evaluation``` needs to add the last window of results when such window is smaller than ```window_size``` (the remaining instances). Important: ```*Evaluator``` classes do not do anything about adding this last window, if we use, for example ```ClassificationWindowEvaluator``` directly, then we need to know that the results for the last window is accessible through ```metrics()``` and not in the last row of the ```metrics_per_window()``` dataframe. 
10. [Tests] Update the ```benchmarking.py``` to use the new API to read files and evaluation
11. [New] Implement test_then_train_SSL_evaluation (Python version) and complete prequential_SSL_evaluation (Python) - needs logic to handle initial_window and delay_length (Java version already implement that)
12. [Doc] Document the interaction between the **fast** evaluation implementations that rely on MOA code. 
13. [New] Search and analyse other ways of sharing data between Java and Python (NumpyStream -> CSV can be improved). 
14. [New] Pipeline logic. Pipeline that process instances incrementally. Example: normalise values before testing or training with an instance. 
15. [New] Clustering API: encapsulate the clusterer and clustering algorithms from MOA
16. [New] Visualization of feature importances (encapsulate the ClassifierWithFeatureImportance class from MOA)
17. [New] Port SO-KNL to ensembles module like AdaptiveRandomForest


**Updated all the notebooks on 10/10/2023, removed some that were outdated**