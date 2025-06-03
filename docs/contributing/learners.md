# Adding Learners
This document describes adding a new classifier, regressor, or 
another learner to CapyMOA. Before doing this, you should have read the
[installation guide](../installation.rst) to set up your development environment.

## Where does my new learner go?
You should add your new learner to the appropriate directory:
- Classifiers go in `src/capymoa/classifier`.
- Regressors go in `src/capymoa/regressor`.
- Anomaly detectors go in `src/capymoa/anomaly`.
- Semi-supervised classifiers go in `src/capymoa/ssl/classifier`.

Each standalone learner should be in its own file, prefixed with `_` to indicate that they are not meant to be imported directly. Instead, they are imported by an `__init__.py` file. The `__init__.py` file is a special file that tells Python to treat the directory as a package.

For example, to add a new classifier class called `MyNewLearner`, you should implement it in `src/capymoa/classifier/_my_new_learner.py` and add it to the `src/capymoa/classifier/__init__.py` file. The `__init__.py` will look like this:
```python
from ._my_new_learner import MyNewLearner
...
__all__ = [
    'MyNewLearner',
    ...
]
```

The prefix and init files allow users to import all classifiers, regressors, 
or semi-supervised from one package while splitting the code into multiple files. You can, for example, import your new learner with the following:
```python
from capymoa.classifier import MyNewLearner
```

## What does a learner implement?

A learner should implement the appropriate interface:
* {py:class}`capymoa.base.Classifier` for classifiers.
* {py:class}`capymoa.base.Regressor` for regressors.
* {py:class}`capymoa.base.AnomalyDetector` for anomaly detectors.
* {py:class}`capymoa.base.ClassifierSSL` for semi-supervised classifiers.
* {py:class}`capymoa.base.Classifier` for online continual learning classifiers.
  Optionally also inheriting from {py:class}`capymoa.ocl.base.TrainTaskAware` or
  {py:class}`capymoa.ocl.base.TestTaskAware` to support learners that are aware of
  task identities or task boundaries.

If your method is a wrapper around a MOA learner, you should use the appropriate
base class:
* {py:class}`capymoa.base.MOAClassifier` for classifiers.
* {py:class}`capymoa.base.MOARegressor` for regressors.
* {py:class}`capymoa.base.MOAAnomalyDetector` for anomaly detectors.

## How do I test my new learner?
You should add a test to ensure your learner achieves and continues to achieves
the expected performance in future versions. CapyMOA provides parametrized
tests for classifiers, regressors, and semi-supervised classifiers. You should
not need to write any new test code. Instead, you should add your test's
parameters to the appropriate test file:

- `tests/test_classifiers.py` for classifiers.
- `tests/test_ssl_classifiers.py` for semi-supervised classifiers.
- `tests/ocl/test_learners.py` for online continual learning learners.
- `tests/test_regressors.py` for regressors.
- `tests/test_anomaly.py` for anomaly detectors.

To run your tests, use the following command:
```bash
python -m pytest -k MyNewLearner
```
The `-k MyNewLearner` flag tells PyTest to run tests containing `MyNewLearner` in the test ID.

* If you want to add documented exemplar usage of your learner, you can add doctests.
See the [testing guide](tests.md) for more information.

* If you need custom test code for your learner, you can add a new test file in
`tests`.

## How do I document my new learner?
You should add a docstring to your learner that describes the learner and its
parameters. The docstring should be in the Sphinx format. Check the 
[documentation guide](docs.rst) for more information and an example.

## How to debug failed GitHub Actions?
Before submitting your pull request, you may wish to run all tests to
ensure your changes will succeed in GitHub Actions. You can run all tests with:
```bash
invoke test
```
If you run into issues with GitHub actions failing to build documentation.
Follow the instructions in the [documentation guide](docs.rst) to build the
documentation locally. The documentation build settings are intentionally strict
to ensure the documentation builds correctly.
