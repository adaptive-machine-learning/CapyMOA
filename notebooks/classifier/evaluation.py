# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluating classifiers in CapyMOA
#
# This notebook further explores **high-level evaluation functions**, **data abstraction** and **classifiers**.
#
# * **High-level evaluation functions**
#     * We show how to evaluate one learner with `evaluate_classifier()`, many learners in one pass over the stream by passing a dictionary to `evaluate_classifier()`, and how to work with the results.
#     * We also discuss particularities about how these evaluation functions relate to how research has developed in the field, and how evaluation is commonly performed and presented.
#
# * **Supervised Learning**
#     * We clarify important information concerning the usage of **classifiers** and their predictions.
#     * For the equivalent walkthrough using regressors, see notebooks/regressor/evaluation.py.
#  
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 28/11/2025**

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, mock_datasets

if is_nb_fast():
    mock_datasets()

# %% [markdown]
# ## The difference between evaluators
#
# * The following example implements a **while loop** that updates a `ClassificationWindowedEvaluator` and a `ClassificationEvaluator` for the same learner. 
# * The `ClassificationWindowedEvaluator` updates the metrics according to tumbling windows which 'forgets' older correct and incorrect predictions. This allows us to observe how well the learner performs in shorter windows. 
# * The `ClassificationEvaluator` updates the metrics, taking into account all the correct and incorrect predictions made. It is useful to observe the overall performance after processing hundreds of thousands of instances.
#
# * **Two important points**:
#     1. Regarding **window_size** in `ClassificationEvaluator`: A `ClassificationEvaluator` also allows us to specify a window size, but it only controls the frequency at which cumulative metrics are calculated.
#     2. If we access metrics directly (not through `metrics_per_window()`) in `ClassificationWindowedEvaluator` we will be looking at the metrics corresponding to the last window.
#  
# For further insight into the specifics of the evaluators, please refer to the documentation at https://www.capymoa.org.

# %%
from capymoa.classifier import AdaptiveRandomForestClassifier
from capymoa.datasets import Electricity
from capymoa.classifier.evaluate import (
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
)

stream = Electricity()

ARF = AdaptiveRandomForestClassifier(schema=stream.get_schema(), ensemble_size=10)

# The window_size in ClassificationWindowedEvaluator specifies the amount of instances used per evaluation.
windowedEvaluatorARF = ClassificationWindowedEvaluator(
    schema=stream.get_schema(), window_size=4500
)
# The window_size ClassificationEvaluator just specifies the frequency at which the cumulative metrics are stored.
classificationEvaluatorARF = ClassificationEvaluator(
    schema=stream.get_schema(), window_size=4500
)

while stream.has_more_instances():
    instance = stream.next_instance()
    prediction = ARF.predict(instance)
    windowedEvaluatorARF.update(instance.y_index, prediction)
    classificationEvaluatorARF.update(instance.y_index, prediction)
    ARF.train(instance)

# Showing only the 'classifications correct (percent)' (i.e. accuracy)
print(
    "[ClassificationWindowedEvaluator] Windowed accuracy reported for every window_size windows"
)
print(windowedEvaluatorARF.accuracy())

print(
    f"[ClassificationEvaluator] Cumulative accuracy: {classificationEvaluatorARF.accuracy()}"
)
# We could report the cumulative accuracy every window_size instances with the following code, but that is normally not very insightful.
# display(classificationEvaluatorARF.metrics_per_window())

# %% [markdown]
# ## High-level evaluation functions
#
# For classification, the high-level function is `evaluate_classifier()`. It runs the test-then-train loop and updates a `ClassificationEvaluator` and a `ClassificationWindowedEvaluator` for you, so you do not need to update them yourself.
#
# Each research domain has its own function and result type:
#
# | Domain | Function | Results |
# |---|---|---|
# | classification | `capymoa.classifier.evaluate_classifier` | `ClassifierResults` |
# | regression | `capymoa.regressor.evaluate_regressor` | `RegressorResults` |
# | prediction intervals | `capymoa.uncertainty.evaluate_prediction_interval` | `PredictionIntervalResults` |
# | semi-supervised | `capymoa.ssl.evaluate_ssl` | `SSLResults` |
# | anomaly detection | `capymoa.anomaly.evaluate_anomaly` | `AnomalyResults` |
#
# Each function evaluates one learner. Pass a dictionary of learners instead, and the function evaluates them all in one pass over the stream.
#
# If you do not know the domain of a learner in advance, use `capymoa.evaluation.prequential_evaluation()`. It checks the type of the learner and calls the matching function from the table.
#
# **Result of a high-level function**
#
# * The return from `evaluate_classifier()` is a `ClassifierResults`: a plain, typed `dict` with a fixed set of keys. Hover over a key in your IDE or see the API documentation for what each one means and its unit.
#
# **Common characteristics for all high-level evaluation functions**
#
# * `evaluate_classifier()` specifies a `max_instances` parameter, which by default is `None`. Depending on the source of the data (e.g. a real stream or a synthetic stream) the function will never stop! The intuition behind this is that streams are infinite, we process them as such. Therefore, it is a good idea to specify `max_instances` unless you are using a snapshot of a stream (i.e. a `Dataset` like `Electricity`)
#
# **Evaluation practices in the literature (and practice)**
#
# Interested readers might want to peruse section **6.1.1 Error Estimation** from [Machine Learning for Data Streams](https://moa.cms.waikato.ac.nz/book-html/) book. We further expand the relationships between the literature and our evaluation functions in the documentation: https://www.capymoa.org.

# %% [markdown]
# ### evaluate_classifier()
#
# The `evaluate_classifier()` function performs a windowed evaluation and a cumulative evaluation at once. Internally, it maintains a `ClassificationWindowedEvaluator` (for the windowed metrics) and `ClassificationEvaluator` (for the cumulative metrics). This allows us to have access to the **cumulative** and **windowed** results without running two separate evaluation functions.
#
# * The metrics over the whole stream (**cumulative**) are keys of the results, for example `results["accuracy"]`.
#
# * The `windowed` key holds the metrics of each window in columns: a dictionary with an `instances` entry and one entry per metric, each an array with one value per window. `pd.DataFrame(results["windowed"])` makes a table from it. `per_class` works the same way, with one value per class.
#
# * `run info` keys such as `learner`, `stream`, `instances`, `wallclock` and `cpu_time` are in every result.
#
# * Invoking `plot_windowed_results()` with a result will plot its `windowed` results.
#
# * For plotting and analysis purposes, one might want to set `store_predictions=True` and `store_y=True` on the `evaluate_classifier()` function, which will include all the predictions and ground truth y in the `y_pred` and `y_true` keys. It is important to note that this can be costly in terms of memory depending on the size of the stream. Otherwise, the result has no `y_pred` or `y_true` key.

# %%
from capymoa.classifier import HoeffdingTree
from capymoa.datasets import ElectricityTiny
from capymoa.classifier import evaluate_classifier
from capymoa.evaluation.plot import plot_windowed_results

elec_stream = ElectricityTiny()
ht = HoeffdingTree(schema=elec_stream.get_schema(), grace_period=50)

results_ht = evaluate_classifier(
    stream=elec_stream,
    learner=ht,
    window_size=100,
    optimise=True,
    store_predictions=False,
    store_y=False,
)

print("\tRun information:")
print(f"results_ht['learner']: {results_ht['learner']}")
print(f"results_ht['wallclock']: {results_ht['wallclock']}")
print(f"results_ht['cpu_time']: {results_ht['cpu_time']}")

print("\n\tThe cumulative metrics:")
print(f"results_ht['accuracy'] = {results_ht['accuracy']}")

print("\n\tPer class metrics:")
import pandas as pd

display(pd.DataFrame(results_ht["per_class"]))

print("\n\tAll the windowed results:")
display(pd.DataFrame(results_ht["windowed"]))

plot_windowed_results(results_ht, metric="accuracy")

# %% [markdown]
# ### Evaluating a single stream using multiple learners
#
# Passing a dictionary of names to learners to `evaluate_classifier()` further encapsulates experiments by executing multiple learners on a single stream.
#
# * This behaves as if we invoked `evaluate_classifier()` multiple times (without the Java loop), but internally it only iterates through the stream once. This is useful if we are faced with a situation where accessing each instance of the stream is costly, then this will be more convenient than just invoking `evaluate_classifier()` multiple times.
#
# * The result is a dictionary from the name of each learner to its results. The name is also the `learner` key of each result.
#
# * The training and testing of each learner is interleaved, so `wallclock` and `cpu_time` are for the whole pass and are the same for all the learners. Do not use them to compare the speed of learners.

# %%
from capymoa.classifier import AdaptiveRandomForestClassifier, OnlineBagging
from capymoa.datasets import Electricity
from capymoa.classifier import evaluate_classifier
from capymoa.evaluation.plot import plot_windowed_results

stream = Electricity()

# Define the learners + an alias (dictionary key)
learners = {
    "OB": OnlineBagging(schema=stream.get_schema(), ensemble_size=10),
    "ARF": AdaptiveRandomForestClassifier(schema=stream.get_schema(), ensemble_size=10),
}

results = evaluate_classifier(stream, learners, window_size=4500)

print(
    f"OB final accuracy = {results['OB']['accuracy']} and ARF final accuracy = {results['ARF']['accuracy']}"
)
plot_windowed_results(results["OB"], results["ARF"], metric="accuracy")

# %% [markdown]
# ## Working with results
#
# The results of every domain are typed dictionaries, which makes them easy to compare, plot and keep.
#
# * A list of results makes a table with one row per run. Since the keys are fixed, the columns always line up.
# * `windowed` is a dict of columns, so `pd.DataFrame(result["windowed"])` makes a table of the windows. Stack the tables of many runs with a `learner` column to plot them with `seaborn.lineplot(..., hue="learner")`.
# * A result holds only plain values, dicts, lists and NumPy arrays, so you can store it as you like. For example, with `pickle`.

# %%
import pandas as pd

# One row per learner.
display(pd.DataFrame(list(results.values()))[["learner", "accuracy", "kappa"]])

# One row per learner and window.
windows = pd.concat(
    [pd.DataFrame(r["windowed"]).assign(learner=name) for name, r in results.items()]
)
display(windows.head())

# %%
import seaborn as sns

sns.lineplot(windows, x="instances", y="accuracy", hue="learner");

# %%
import pickle

restored = pickle.loads(pickle.dumps(results["OB"]))
print(restored["accuracy"] == results["OB"]["accuracy"])
