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
#     * We demonstrate how to use `prequential_evaluation()` and how to further encapsulate prequential evaluation using `prequential_evaluation_multiple_learners`.
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
from capymoa.evaluation import ClassificationEvaluator, ClassificationWindowedEvaluator

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
# In CapyMOA, for supervised learning, there is one primary evaluation function designed to handle the manipulation of evaluators, i.e. the `prequential_evaluation()`. This function streamlines the process, ensuring users need not directly update them. Essentially, this function executes the evaluation loop and updates the relevant evaluators:
#
# * `prequential_evaluation()` utilises `ClassificationEvaluator` and `ClassificationWindowedEvaluator`.
#
# Previously, CapyMOA included two other functions: `cumulative_evaluation()` and `windowed_evaluation()`. However, since `prequential_evaluation()` incorporates the functionality of both we decided to remove those functions and focus on `prequential_evaluation()`.
# It's important to note that `prequential_evaluation()` is applicable to both `Regression` and `Prediction Intervals` besides `Classification`. The functionality and interpretation remain the same across these cases, but the metrics differ.
#
# **Result of a high-level function**
#
# * The return from `prequential_evaluation()` is a `PrequentialResults` object which provides access to the `cumulative` and `windowed` metrics as well as some other metrics (like wall-clock and cpu time).
#
# **Common characteristics for all high-level evaluation functions**
#
# * `prequential_evaluation()` specifies a `max_instances` parameter, which by default is `None`. Depending on the source of the data (e.g. a real stream or a synthetic stream) the function will never stop! The intuition behind this is that streams are infinite, we process them as such. Therefore, it is a good idea to specify `max_instances` unless you are using a snapshot of a stream (i.e. a `Dataset` like `Electricity`)
#
# **Evaluation practices in the literature (and practice)**
#
# Interested readers might want to peruse section **6.1.1 Error Estimation** from [Machine Learning for Data Streams](https://moa.cms.waikato.ac.nz/book-html/) book. We further expand the relationships between the literature and our evaluation functions in the documentation: https://www.capymoa.org.

# %% [markdown]
# ### prequential_evaluation()
#
# The `prequential_evaluation()` function performs a windowed evaluation and a cumulative evaluation at once. Internally, it maintains a `ClassificationWindowedEvaluator` (for the windowed metrics) and `ClassificationEvaluator` (for the cumulative metrics). This allows us to have access to the **cumulative** and **windowed** results without running two separate evaluation functions. 
#
# * The results returned from `prequential_evaluation()` allows access to the evaluator objects `ClassificationWindowedEvaluator` (attribute `windowed`) and `ClassificationEvaluator` (attribute `cumulative`) directly. 
#   
# * Notice that the computational overhead of training and assessing the same model twice outweighs the minimum overhead of updating the two evaluators within the function. Thus, it is advisable to use the `prequential_evaluation()` function instead of creating separate `while` loops for evaluation.
#
# * Advanced users might intuitively request metrics directly from the `results` object, which will return the `cumulative` metrics. For example, assuming `results = prequential_evaluation(...)`, `results.accuracy()` will return the `cumulative` accuracy. 
# **IMPORTANT**: There are no IDE hints for these metrics as they are accessed dynamically via `__getattr__`. It is advisable that users access metrics explicitly through `results.cumulative` (or `results['cumulative']`) or `results.windowed` (or `results['windowed']`).
#
# * Invoking `results.metrics_per_window()` from a `results` object will return the dataframe with the `windowed` results.
#
# * `results.write_to_file()` will output the `cumulative` and `windowed` results to a directory.
#
# * `results.cumulative.metrics_dict()` will return all the cumulative metrics identifiers and their corresponding values in a dictionary structure.
#
# * Invoking `plot_windowed_results()` with a `PrequentialResults` object will plot its `windowed` results.
#
# * For plotting and analysis purposes, one might want to set `store_predictions=True` and `store_y=True` on the `prequential_evaluation()` function, which will include all the predictions and ground truth y in the `PrequentialResults` object. It is important to note that this can be costly in terms of memory depending on the size of the stream.

# %%
from capymoa.classifier import HoeffdingTree
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.visualization import plot_windowed_results

elec_stream = ElectricityTiny()
ht = HoeffdingTree(schema=elec_stream.get_schema(), grace_period=50)

results_ht = prequential_evaluation(
    stream=elec_stream,
    learner=ht,
    window_size=100,
    optimise=True,
    store_predictions=False,
    store_y=False,
)


print("\tDifferent ways of accessing metrics:")

print(
    f"results_ht['wallclock']: {results_ht['wallclock']} results_ht.wallclock(): {results_ht.wallclock()}"
)
print(
    f"results_ht['cpu_time']: {results_ht['cpu_time']} results_ht.cpu_time(): {results_ht.cpu_time()}"
)

print(f"results_ht.cumulative.accuracy() = {results_ht.cumulative.accuracy()}")
print(f"results_ht.cumulative['accuracy'] = {results_ht.cumulative['accuracy']}")
print(f"results_ht['cumulative'].accuracy() = {results_ht['cumulative'].accuracy()}")
print(f"results_ht.accuracy() = {results_ht.accuracy()}")

print("\n\tAll the cumulative results:")
print(results_ht.cumulative.metrics_dict())

print("\n\tAll the windowed results:")
display(results_ht.metrics_per_window())
# OR display(results_ht.windowed.metrics_per_window())

# results_ht.write_to_file() -> this will save the results to a directory

plot_windowed_results(results_ht, metric="accuracy")

# %% [markdown]
# ### Evaluating a single stream using multiple learners
#
# `prequential_evaluation_multiple_learners()` further encapsulates experiments by executing multiple learners on a single stream. 
#
# * This function behaves as if we invoked `prequential_evaluation()` multiple times, but internally it only iterates through the stream once. This is useful if we are faced with a situation where accessing each instance of the stream is costly, then this function will be more convenient than just invoking `prequential_evaluation()` multiple times. 
#
# * This method does not calculate `wallclock` or `cpu_time` because the training and testing of each learner is interleaved, thus timing estimations are unreliable. Thus, the results dictionaries do not contain the keys `wallclock` and `cpu_time`.

# %%
from capymoa.classifier import AdaptiveRandomForestClassifier, OnlineBagging
from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation_multiple_learners
from capymoa.evaluation.visualization import plot_windowed_results

stream = Electricity()

# Define the learners + an alias (dictionary key)
learners = {
    "OB": OnlineBagging(schema=stream.get_schema(), ensemble_size=10),
    "ARF": AdaptiveRandomForestClassifier(schema=stream.get_schema(), ensemble_size=10),
}

results = prequential_evaluation_multiple_learners(stream, learners, window_size=4500)

print(
    f"OB final accuracy = {results['OB'].cumulative.accuracy()} and ARF final accuracy = {results['ARF'].cumulative.accuracy()}"
)
plot_windowed_results(results["OB"], results["ARF"], metric="accuracy")
