# MOABridge
Python wrapper for MOA to allow efficient use of existing algorithms with a more modern API

To use the project
1. In ```config.ini``` set the path to the ```moa.jar``` that you are using (download it from the project, see [```src/capymoa/jar/accessing_jar.txt```](src/capymoa/jar/accessing_jar.txt)). It should work with any MOA jar released after 2023. Alternatively, run ```make download``` to download the jar and datasets.
2. Make sure `JAVA_HOME` is set. 
3. Use ```conda env create -f environment.yml``` to create a conda environment with all the requirements. 
	* Optional: Use pip install on the requirements. 
	* If you are using a custom installation, make sure to install jpype through ```pip install jpype1``` (yes, there is a 1 there)
4. Activate the conda environment ```conda activate MOABridge```
5. Add the project to your python path in your ```.bashrc```, 
   ```.bash_profile```, ```.zshrc``` or ```.profile``` with the following command (replace ```<MY PATH HERE>``` with the path to the project):
```sh
export PYTHONPATH=$PYTHONPATH:<MY PATH HERE>/MOABridge/src
```
6. Try the DEMO notebook ```jupyter notebook DEMO.ipynb```. The notebook must be
   started with the correct ```PYTHONPATH``` and ```JAVA_HOME``` variables set.
   To double check run ```echo $PYTHONPATH``` and ```echo $JAVA_HOME``` in the
   terminal before starting the notebook.


# Functionality
* Full support for classification, regression and semi-supervised classification. 
* Read CSV or ARFF files, or use synthetic generators from MOA.

# Tutorial notebooks
These notebooks show how to do things. Data is available in the ```/data/``` directory (some of which will need to be downloaded, see instrucitons there). 

* **DEMO.ipynb**: Contains simple examples on how to execute classification and regression, using MOA objets to configure synthetic generators or classifiers/regressors. 
* **Evaluation_and_Data_Reading.ipynb**: Many examples showing how to perform different evaluations for classification and regression using different methods (i.e. a loop or buildin functions). 
* **Learners_API_Examples.ipynb**: Similar to the DEMO, but shows more capabilities of the evaluator and learner objects.
* **Using_sklearn_pytorch.ipynb**: Shows how one can use the API to run sklearn algorithms (those that implement ```partial_fit```) and PyTorch models. 

# Test notebooks
These show how some parts of the library were developed and provide comparisons of different options on how to do things. 

* **Efficient_Evaluation.ipynb**: Some simple benchmarks comparing different versions of test_then_train_evaluation and prequential_evaluation. Interesting to developers looking to improve that aspect of the platform. 
* **Using_jpype_MOA_example.ipynb**: Example using MOA directly from jpype without the library in-between. Interesting to developers looking for a full example of how it is done without the library. 
* **Data_Reading.ipynb**: Data reading examples. More interesting to developers looking to improve the data capabilities. 



# TODO (updated 20/10/2023)
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
17. [New] Port SO-KNL to ensembles module like AdaptiveRandomForest (SO-KNL not yet in MOA, still a PR)
18. [New] Add logic to interpret nominal values (strings) in the class label for ```stream_from_file``` when using a CSV.


**Updated all the notebooks on 20/10/2023, removed some that were outdated**
