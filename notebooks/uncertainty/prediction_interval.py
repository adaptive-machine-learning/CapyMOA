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
# # Prediction Intervals for Data Streams
#
# * This notebook covers how to utilise prediction intervals for regression tasks in CapyMOA.
# * Two methods for obtaining prediction intervals are currently available in CapyMOA: MVE and AdaPI.
#
# More details about prediction intervals for streaming data can be found in the AdaPI paper: 
#
# [Yibin Sun, Bernhard Pfahringer, Heitor Murilo Gomes & Albert Bifet. "Adaptive Prediction Interval for Data Stream Regression." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, 2024.](https://link.springer.com/chapter/10.1007/978-981-97-2259-4_10)
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last updated on 28/11/2025**

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, mock_datasets

if is_nb_fast():
    mock_datasets()

# %%
from capymoa.datasets import Fried

# load data
fried_stream = Fried()

# %% [markdown]
# ## Basic prediction interval usage
#
# * Here is an example of prediction intervals in CapyMOA.
# * Currently available prediction interval learners require a regressor base model.

# %%
from capymoa.regressor import SOKNL
from capymoa.uncertainty import MVE

# build prediction interval learner in regular manner
soknl = SOKNL(schema=fried_stream.get_schema(), ensemble_size=10)
mve = MVE(schema=fried_stream.get_schema(), base_learner=soknl)

# build prediction interval learner in in-line manner
mve_inline = MVE(
    schema=fried_stream.get_schema(),
    base_learner=SOKNL(schema=fried_stream.get_schema(), ensemble_size=10),
)

# %% [markdown]
# ## Creating evaluators
#
# * There are currently two types of prediction interval evaluators implemented: basic (cumulative) and windowed.

# %%
from capymoa.evaluation.evaluation import (
    PredictionIntervalEvaluator,
    PredictionIntervalWindowedEvaluator,
)

# build prediction interval (basic and windowed) evaluators
mve_evaluator = PredictionIntervalEvaluator(schema=fried_stream.get_schema())
mve_windowed_evaluator = PredictionIntervalWindowedEvaluator(
    schema=fried_stream.get_schema(), window_size=1000
)

# %% [markdown]
# ## Running test-then-train/prequential tasks manually
#
# * **Don't forget to train the models (call .train() function) at the end!**

# %%
# run test-then-train/prequential tasks
while fried_stream.has_more_instances():
    instance = fried_stream.next_instance()
    prediction = mve.predict(instance)
    mve_evaluator.update(instance.y_value, prediction)
    mve_windowed_evaluator.update(instance.y_value, prediction)
    mve.train(instance)

# %% [markdown]
# ## Results from both evaluators

# %%
# show results
print(
    f"MVE basic evaluation:\ncoverage: {mve_evaluator.coverage()}, NMPIW: {mve_evaluator.nmpiw()}"
)
print(
    f"MVE windowed evaluation in last window:\ncoverage: {mve_windowed_evaluator.coverage()}, \nNMPIW: {mve_windowed_evaluator.nmpiw()}"
)

# %% [markdown]
# ## Wrap things up with prequential evaluation
#
# * Prediction interval tasks also can be wrapped into prequential evaluation in CapyMOA.

# %%
from capymoa.evaluation import prequential_evaluation
from capymoa.uncertainty import AdaPI

# restart stream
fried_stream.restart()
# specify regressive model
regressive_learner = SOKNL(schema=fried_stream.get_schema(), ensemble_size=10)
# build prediction interval models
mve_learner = MVE(schema=fried_stream.get_schema(), base_learner=regressive_learner)
adapi_learner = AdaPI(
    schema=fried_stream.get_schema(), base_learner=regressive_learner, limit=0.001
)
# gather results
mve_results = prequential_evaluation(
    stream=fried_stream, learner=mve_learner, window_size=1000
)
adapi_results = prequential_evaluation(
    stream=fried_stream, learner=adapi_learner, window_size=1000
)

# show overall results
print(
    f"MVE coverage: {mve_results.cumulative.coverage()}, NMPIW: {mve_results.cumulative.nmpiw()}"
)
print(
    f"AdaPI coverage: {adapi_results.cumulative.coverage()}, NMPIW: {adapi_results.cumulative.nmpiw()}"
)

# %% [markdown]
# ## Plots are also supported

# %%
from capymoa.evaluation.visualization import plot_windowed_results

# plot over time comparison
plot_windowed_results(mve_results, adapi_results, metric="coverage")
plot_windowed_results(mve_results, adapi_results, metric="nmpiw")

# %% [markdown]
# ### Plotting prediction intervals over time
#
# * We also provide a visualisation tool for plotting prediction intervals over time.
# * The function `plot_prediction_interval` can be used to plot the prediction intervals over time.
# * This function can take one of two prediction interval results as input.

# %% [markdown]
# ### Single result plotting example
#
# * In order to plot the prediction interval over time, we need to have stored the predictions and the ground truth values in the prediction interval results.
# * The shaded area represents the prediction interval, while the solid line represents the regressor's predictions.
# * The star markers represent the ground truth values that are covered by the intervals.
# * On the other hand, the cross markers represent the ground truth values that are outside the intervals. 
# * The colors can be adjusted by the `colors` parameter in the function as a list.
# * `start` and `end` parameters can be used to specify the range of the plot.
# * The `ground truth` and `predictions` can be omitted by setting the `plot_truth` and `plot_predictions` parameters to `False`.
#
# **We have to set `optimise` to `False` to avoid subscribing problems**.

# %%
new_mve_learner = MVE(
    schema=fried_stream.get_schema(),
    base_learner=SOKNL(schema=fried_stream.get_schema(), ensemble_size=10),
)
new_mve_results = prequential_evaluation(
    stream=fried_stream,
    learner=new_mve_learner,
    window_size=1000,
    optimise=False,
    store_predictions=True,
    store_y=True,
)

# %%
from capymoa.evaluation.visualization import plot_prediction_interval

plot_prediction_interval(new_mve_results, start=300, end=500, colors=["coral"])

# %% [markdown]
# ### Two results plotting example
#
# * For comparison purposes, we can also plot two prediction interval results over time.
# * We don't offer the ability to take more than two since it makes the plot too messy to read.

# %%
# Let's add another results
new_adapi_learner = AdaPI(
    schema=fried_stream.get_schema(),
    base_learner=SOKNL(schema=fried_stream.get_schema(), ensemble_size=10),
    limit=0.001,
)
new_adapi_results = prequential_evaluation(
    stream=fried_stream,
    learner=new_adapi_learner,
    window_size=1000,
    optimise=False,
    store_predictions=True,
    store_y=True,
)

# %%
plot_prediction_interval(
    new_mve_results,
    new_adapi_results,
    start=300,
    end=500,
    colors=["coral", "teal"],
    plot_predictions=False,
)

# %% [markdown]
# * New plus markers represent the ground truth values that are covered by the narrower but not the wider intervals. 
# * The function automatically puts the wider area to the back to make the narrower intervals more visible.
