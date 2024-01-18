# CapyMOA
Python wrapper for MOA to allow efficient use of existing algorithms with a more modern API

To use the project
1. Use ```conda env create -f environment.yml``` to create a conda environment with all the requirements. 
	* Optional: Use pip install on the requirements. 
	* If you are using a custom installation, make sure to install jpype through ```pip install jpype1``` (yes, there is a 1 there)
 	* If you are using windows, use ```environment_wds.yml``` instead of ```environment.yml```
2. Run ```make download``` to get the MOA jar.
   	* Optional: In ```config.ini``` set the path to the ```moa.jar``` that you are using if you are using a custom ```moa.jar```, otherwise the jar specified in [```src/capymoa/jar/accessing_jar.txt```](src/capymoa/jar/accessing_jar.txt)) will be used. It should work with any MOA jar released after 2023, but some functions may not work (as they haven't been merged into moa yet, such as the SSL supporting functions). 
3. Make sure `JAVA_HOME` is set. 
4. Activate the conda environment ```conda activate CapyMOA```
5. Add the project to your python path in your ```.bashrc```, 
   ```.bash_profile```, ```.zshrc``` or ```.profile``` with the following command (replace ```<MY PATH HERE>``` with the path to the project):
```sh
export PYTHONPATH=$PYTHONPATH:<MY PATH HERE>/CapyMOA/src
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

**Updated all the notebooks on 16/01/2024, removed some that were outdated**
