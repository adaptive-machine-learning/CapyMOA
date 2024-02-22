# CapyMOA
Python wrapper for MOA to allow efficient use of existing algorithms with a more modern API


# Developer Installation

1. Install Java

   CapyMOA requires Java to be installed and accessible in your environment. You
   can check if Java is installed by running the following command in your terminal:
   ```bash
   java -version
   ```

   If Java is not installed, you can download OpenJDK (Open Java Development Kit) 
   from [this link](https://openjdk.org/install/), or alternatively the Oracle 
   JDK from [this link](https://www.oracle.com/java).

   Now that Java is installed, you should see an output similar to the following
   when you run the command `java -version`:
   ```txt
   openjdk version "17.0.9" 2023-10-17
   OpenJDK Runtime Environment (build 17.0.9+8)
   OpenJDK 64-Bit Server VM (build 17.0.9+8, mixed mode)
   ```

   > If you have multiple versions of Java installed, you can set the `JAVA_HOME`
   > environment variable to the path of the Java version you want to use.


1. **(Optional)** Setup a conda environment.

   It is recommended to use a conda environment since using a
   conda environment isolates project dependencies and avoids conflicts.
   > Follow the instructions [at this link](https://docs.conda.io/projects/miniconda/en/latest/) to install miniconda.

   Setup the conda environment by running one of the following commands:
   ```sh
   conda env create -f environment.yml     # For linux
   conda env create -f environment_wds.yml # For windows
   ```
   Ensure the environment is activated:
   ```sh
   conda activate CapyMOA
   ```

2. **Install CapyMOA.**

   Use pip to install the project in editable mode with the development dependencies.
   ```bash
   pip install --editable ".[dev]"
   ```

3. **(Optional)** Run the tests.
   ```bash
   python -m pytest
   ```

4. **(Optional)** Try the example notebooks.
   ```
   python -m jupyter notebook notebooks/DEMO.ipynb
   ```

5. **(Optional)** Neural Network support.

   The CapyMOA algorithms using deep learning require PyTorch. It
   is not installed by default because different versions are required
   for different hardware. If you want to use these algorithms, follow the
   instructions [at this link](https://pytorch.org/get-started/locally/) to
   get the correct version for your hardware.

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
