# Benchmark Package Usage Guide

This guide will walk you through the steps required to use the benchmark package, including setting up environments, configuring datasets, and properly updating variables to execute desired benchmarks.

## 1. Create Appropriate Environments

Before using the benchmark package, ensure you have the correct environment set up. You can create and activate a conda environment as follows:

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create a new environment and install necessary dependencies.

Example:
```bash
conda create --name benchmark_env python=3.8
conda activate benchmark_env
pip install -r requirements.txt
```

Ensure all required libraries for benchmarking are listed in the `requirements.txt` file.

## 2. Ensure Required Datasets are Available

Make sure that all the required datasets are available in the `benchmarking_datasets` directory. The datasets you want to use for benchmarking should be placed inside this directory. The structure should be as follows:

```
benchmarking_datasets/
    dataset1.csv
    dataset2.csv
    ...
```

Ensure that the datasets are properly formatted and ready for use.

## 3. Update the `dataset_path` Variables

After placing the datasets in the `benchmarking_datasets` directory, update the `dataset_path` variables in the benchmark script to correctly point to the respective dataset files.

Ensure that the path variables are updated for each benchmark script to refer to the correct dataset.

## 4. Update the Learner Variables

Different machine learning libraries may have different learner (algorithm) names. Update the `learner` variables in the benchmarking script based on the specific library you are using. 

For example:
- For `scikit-learn`, use `RandomForestClassifier`.
- For `river`, use `AdaptiveRandomForestClassifier`.

Make sure the correct learner name is assigned in each script for consistency.

## 5. Update the `arguments` Variables

Each learner may have different parameter combinations for benchmarking. The `arguments` variable should be updated to reflect these combinations in the format:

```
'learner name' : [list of different parameter combinations]
```

Example:

```
'RandomForestClassifier': [
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 20}
]
```

Ensure that you list all relevant combinations for the learners you are benchmarking.

## 6. Modify the `__name__` Function

To execute only the desired benchmarks, you will need to modify the `__name__` function in the script. Update it to only include the learners, datasets, and arguments you wish to run.

This ensures that only the specific benchmarks are executed when the script is run, preventing unnecessary or irrelevant executions.

---

Following these steps will ensure that your benchmarking environment, datasets, learners, and parameters are properly configured, allowing you to run the desired benchmarks efficiently. Make sure to revisit and update these variables as needed for different benchmarking scenarios.
``` 

This Markdown file should be clear and structured for users to follow along with your benchmarking package setup and usage.