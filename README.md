# CapyMOA
Python wrapper for MOA to allow efficient use of existing algorithms with a more modern API

> [!IMPORTANT]
> * **[How to install CapyMOA](docs/installation.md)**
> * **[How to add documentation](docs/contributing/docs.md)**
> * **[How to add tests](docs/contributing/tests.md)**
> * **[How to add new algorithms or methods](docs/contributing/learners.md)**


# Functionality
* Full support for classification, regression and semi-supervised classification. 
* Read CSV or ARFF files, or use synthetic generators from MOA.

# Tutorial notebooks
These notebooks show how to do things. Data is available in the ```/data/``` directory (some of which will need to be downloaded, see instrucitons there). 

* [`00_getting_started.ipynb`](notebooks/00_getting_started.ipynb): Contains simple examples on how to execute classification and regression, using MOA objets to configure synthetic generators or classifiers/regressors. 
* [`01_evaluation_and_data_reading.ipynb`](notebooks/01_evaluation_and_data_reading.ipynb): Many examples showing how to perform different evaluations for classification and regression using different methods (i.e. a loop or buildin functions). 
* [`02_learners_api_examples.ipynb`](notebooks/02_learners_api_examples.ipynb): Shows more capabilities of the evaluator and learner objects.
* [`03_using_sklearn_pytorch.ipynb`](notebooks/03_using_sklearn_pytorch.ipynb): Shows how one can use the API to run sklearn algorithms (those that implement ```partial_fit```) and PyTorch models.
* [`04_drift_streams.ipynb`](notebooks/04_drift_streams.ipynb): Shows how to setup
   simulated concept drifts in data streams.

# Test notebooks
These show how some parts of the library were developed and provide comparisons of different options on how to do things. 

* **Efficient_Evaluation.ipynb**: Some simple benchmarks comparing different versions of test_then_train_evaluation and prequential_evaluation. Interesting to developers looking to improve that aspect of the platform. 
* **Using_jpype_MOA_example.ipynb**: Example using MOA directly from jpype without the library in-between. Interesting to developers looking for a full example of how it is done without the library. 
* **Data_Reading.ipynb**: Data reading examples. More interesting to developers looking to improve the data capabilities. 

**Updated all the notebooks on 16/01/2024, removed some that were outdated**
