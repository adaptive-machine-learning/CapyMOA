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
# # Getting started with classification
#
# This notebook shows some basic usage of CapyMOA for streaming classification.
#
# * There are more detailed notebooks and documentation available; our goal here is just to present some high-level functions and demonstrate a subset of CapyMOA's functionalities.
# * For simplicity, we simulate data streams in the following examples using datasets and employing synthetic generators. One could also read data directly from a CSV or ARFF (See [stream_from_file](https://capymoa.org/api/modules/capymoa.stream.html#capymoa.stream.stream_from_file) function).
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org
#
# **last update on 05/08/2026**

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, mock_datasets

if is_nb_fast():
    mock_datasets()

# %% [markdown]
# ## Test-then-train loop
#
# * Classification for data streams traditionally assumes instances are available
#   to the classifier in an incremental fashion and labels become available before
#   a new instance becomes available.
# * It is common to simulate this behavior using a **while loop**, often referred
#   to as a **test-then-train loop** which contains 4 distinct steps:
#     1. Fetches the next instance from the stream
#     2. Makes a prediction
#     3. Train the model with the instance
#     4. Update a mechanism to keep track of metrics
#
# **Some remarks about the test-then-train loop**:
#
# * We must not train before testing, meaning that steps 2 and 3 should not be interchanged, as this would invalidate our interpretation concerning how the model performs on unseen data, leading to unreliable evaluations of its efficacy. 
# * Steps 3 and 4 can be completed in any order without altering the result. 
# * What if labels are not immediately available? Then you might want to read about delayed labeling and partially labeled data, see [A Survey on Semi-supervised Learning for Delayed Partially Labelled Data Streams](https://dl.acm.org/doi/full/10.1145/3523055)
# * More information on classification for data streams is available at section **2.2 Classification** from the [Machine Learning for Data Streams](https://moa.cms.waikato.ac.nz/book-html/) book

# %%
from capymoa.classifier import OnlineBagging
from capymoa.datasets import Electricity
from capymoa.evaluation import ClassificationEvaluator

elec_stream = Electricity()
ob_learner = OnlineBagging(schema=elec_stream.get_schema(), ensemble_size=5)
ob_evaluator = ClassificationEvaluator(schema=elec_stream.get_schema())

for instance in elec_stream:
    prediction = ob_learner.predict(instance)
    ob_learner.train(instance)
    ob_evaluator.update(instance.y_index, prediction)

print(ob_evaluator.accuracy())

# %% [markdown]
# ### High-level evaluation functions
#
# * If our goal is just to evaluate learners, it would be tedious to keep writing **test-then-train loops**. 
# Thus, it makes sense to encapsulate that loop inside **high-level evaluation functions**. 
#
# * Furthermore, sometimes we are interested in **cumulative metrics** and sometimes we care about **windowed metrics**. For example, if we want to know how accurate our model is so far, considering all the instances it has seen, then we would look at its **cumulative metrics**. However, we might also be interested in how well the model is performing every **n** number of instances, so that we can, for example, identify periods in which our model was really struggling to produce correct predictions. 
#
# * In this example, we use the `prequential_evaluation` function, which provides us with both the cumulative and the windowed metrics! 
#
# * Some remarks:
#     * If you want to know more about other **high-level evaluation functions**, **evaluators**, or which **metrics** are available, check the **evaluation** notebook (notebooks/classifier/evaluation.py).
#     * The **results** from evaluation functions such as **prequential_evaluation** follow a standard and are discussed thoroughly in the **Evaluation documentation** at http://www.capymoa.org.
#     * Sometimes authors refer to the **cumulative** metrics as **test-then-train** metrics, such as **test-then-train accuracy** (or TTT accuracy for short). They all refer to the same concept.
#     * Shouldn't we recreate the stream object `elec_stream`? No, `prequential_evaluation()`, by default, will automatically `restart()` streams when they are reused.
#
# In the below example `prequential_evaluation` is used with a `HoeffdingTree` classifier on the `Electricity` data stream.

# %%
from capymoa.classifier import HoeffdingTree
from capymoa.evaluation import prequential_evaluation

ht = HoeffdingTree(schema=elec_stream.get_schema(), grace_period=50)

# Obtain the results from the high-level function.
# Note that we need to specify a window_size as we obtain both windowed and cumulative results.
# The results from a high-level evaluation function are represented as a PrequentialResults object.
results_ht = prequential_evaluation(stream=elec_stream, learner=ht, window_size=4500)

print(
    f"Cumulative accuracy = {results_ht.cumulative.accuracy()}, wall-clock time: {results_ht.wallclock()}"
)

# The windowed results are conveniently stored in a pandas DataFrame.
display(results_ht.windowed.metrics_per_window())

# %% [markdown]
# ### Comparing results among classifiers
#
# * CapyMOA provides `plot_windowed_results` as an easy visualisation function for quickly comparing **windowed metrics**.
# * In the example below, we create three classifiers: HoeffdingAdaptiveTree, HoeffdingTree and AdaptiveRandomForest, and plot the results using `plot_windowed_results`.
# * More details about `plot_windowed_results` options are described in the documentation at http://www.capymoa.org.

# %%
from moa.classifiers.trees import HoeffdingAdaptiveTree

from capymoa.base import MOAClassifier
from capymoa.classifier import AdaptiveRandomForestClassifier, HoeffdingTree
from capymoa.evaluation.visualization import plot_windowed_results

# Create the wrapper for HoeffdingAdaptiveTree (from MOA).
HAT = MOAClassifier(
    schema=elec_stream.get_schema(), moa_learner=HoeffdingAdaptiveTree, CLI="-g 50"
)
HT = HoeffdingTree(schema=elec_stream.get_schema(), grace_period=50)
ARF = AdaptiveRandomForestClassifier(
    schema=elec_stream.get_schema(), ensemble_size=10, number_of_jobs=4
)

results_HAT = prequential_evaluation(stream=elec_stream, learner=HAT, window_size=4500)
results_HT = prequential_evaluation(stream=elec_stream, learner=HT, window_size=4500)
results_ARF = prequential_evaluation(stream=elec_stream, learner=ARF, window_size=4500)

# Comparing models based on their cumulative accuracy.
print(f"HAT accuracy = {results_HAT.cumulative.accuracy()}")
print(f"HT accuracy = {results_HT.cumulative.accuracy()}")
print(f"ARF accuracy = {results_ARF.cumulative.accuracy()}")

# Plotting the results. Note that we ovewrote the ylabel, but that doesn't change the metric.
plot_windowed_results(
    results_HAT,
    results_HT,
    results_ARF,
    metric="accuracy",
    xlabel="# Instances (window)",
)

# %% [markdown]
# ## Concept drift
#
# * One of the most challenging and defining aspects of data streams is the phenomenon known as **concept drifts**.
# * In CapyMOA, we designed the simplest and most complete API for simulating, visualising and assessing concept drifts.
# * In the example below, we focus on a simple way of simulating and visualising a drifting stream. There is a tutorial focusing entirely on how concept drift can be simulated, detected and assessed in a separate notebook (See notebooks/drift: `Simulating Concept Drifts with the DriftStream API`).

# %% [markdown]
# ### Plotting drift detection results
#
# * This example uses the DriftStream building API, precisely the **positional version** where drifts are specified according to their exact location in the stream.
# * **Integration with the visualisation function.** The DriftStream object carries meta-information about the drift which is passed along the stream and thus becomes available to `plot_windowed_results`.
#
# * The following plot contains two drifts: 1 abrupt and 1 gradual, such that the abrupt drift is located at instance 5000 and the gradual drift starts at instance 9000 and ends at 12000. This information is provided to the stream via `GradualDrift(start=9000, end=12000)`.
#
# * More details concerning concept drifts in CapyMOA can be found in the documentation at http://www.capymoa.org.

# %%
from capymoa.classifier import OnlineBagging
from capymoa.stream.drift import AbruptDrift, DriftStream, GradualDrift
from capymoa.stream.generator import SEA

# Generating a synthetic stream with 1 abrupt drift and 1 gradual drift.
stream_sea2drift = DriftStream(
    stream=[
        SEA(function=1),
        AbruptDrift(position=5000),
        SEA(function=3),
        GradualDrift(start=9000, end=12000),
        SEA(function=1),
    ]
)

OB = OnlineBagging(schema=stream_sea2drift.get_schema(), ensemble_size=10)

# Since this is a synthetic stream, max_instances is needed to determine the amount of instances to be generated.
results_sea2drift_OB = prequential_evaluation(
    stream=stream_sea2drift, learner=OB, window_size=100, max_instances=15000
)

plot_windowed_results(results_sea2drift_OB, metric="accuracy")
