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
# # Evaluating regressors in CapyMOA
#
# This notebook further explores **high-level evaluation functions** as applied to **regressors**.
#
# * **High-level evaluation functions**
#     * We use `evaluate_regressor()`, which works like `evaluate_classifier()` introduced for classification (see notebooks/classifier/evaluation.py), and show how they apply to regression with only minor differences.
#     * We also show how to plot **predictions vs. ground truth** over time, which is particularly useful for regression tasks.
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
# ## Regression
#
# * We introduce a simple example using regression just to show how similar it is to assess regressors using the **high-level evaluation functions**.
# * The example below uses `evaluate_regressor()`. Its results are a `RegressorResults` (see `capymoa.regressor.evaluate`), a typed dictionary with the keys `mae`, `rmse`, `rmae`, `r2` and `adjusted_r2` besides the run information and the `windowed` table.
# * One difference between classification and regression evaluation in CapyMOA is that the evaluators are different. Instead of `ClassificationEvaluator` and `ClassificationWindowedEvaluator` functions use `RegressionEvaluator` and `RegressionWindowedEvaluator`.

# %%
from capymoa.datasets import Fried
from capymoa.regressor import evaluate_regressor
from capymoa.evaluation.plot import plot_windowed_results
from capymoa.regressor import AdaptiveRandomForestRegressor, KNNRegressor

stream = Fried()
kNN_learner = KNNRegressor(schema=stream.get_schema(), k=5)
ARF_learner = AdaptiveRandomForestRegressor(
    schema=stream.get_schema(), ensemble_size=10
)

kNN_results = evaluate_regressor(
    stream=stream, learner=kNN_learner, window_size=5000
)
ARF_results = evaluate_regressor(
    stream=stream, learner=ARF_learner, window_size=5000
)

print(
    f"{kNN_results['learner']} [cumulative] RMSE = {kNN_results['rmse']} and \
    {ARF_results['learner']}  [cumulative] RMSE = {ARF_results['rmse']}"
)

plot_windowed_results(kNN_results, ARF_results, metric="rmse")

# %% [markdown]
# ### Evaluating a single stream using multiple learners
#
# * `evaluate_regressor` evaluates multiple regressors when you pass a dictionary of learners; the example below shows how it can be used.

# %%
from capymoa.regressor import evaluate_regressor

# Define the learners + an alias (dictionary key)
learners = {
    "kNNReg_k5": KNNRegressor(schema=stream.get_schema(), k=5),
    "kNNReg_k2": KNNRegressor(schema=stream.get_schema(), k=2),
    "kNNReg_k5_median": KNNRegressor(schema=stream.get_schema(), CLI="-k 5 -m"),
    "ARFReg_s5": AdaptiveRandomForestRegressor(
        schema=stream.get_schema(), ensemble_size=5
    ),
}

results = evaluate_regressor(stream, learners)

print("Cumulative results for each learner:")
for learner_id in learners:
    if learner_id in results:
        result = results[learner_id]
        print(
            f"{learner_id}, RMSE: {result['rmse']:.2f}, adjusted R2: {result['adjusted_r2']:.2f}"
        )

# Tip: a list of results makes a table with one row per learner.
import pandas as pd

display(pd.DataFrame(list(results.values()))[["learner", "rmse", "adjusted_r2"]])
plot_windowed_results(
    results["kNNReg_k5"],
    results["kNNReg_k2"],
    results["kNNReg_k5_median"],
    results["ARFReg_s5"],
    metric="rmse",
)

plot_windowed_results(
    results["kNNReg_k5"],
    results["kNNReg_k2"],
    results["kNNReg_k5_median"],
    results["ARFReg_s5"],
    metric="adjusted_r2",
)

# %% [markdown]
# ### Plotting predictions vs. ground truth over time
#
# * In regression it is sometimes desirable to plot **predictions vs. ground truth** to observe what is happening with the stream. If we create a custom loop and use the evaluators directly, it is trivial to store the ground truth and predictions, and then proceed to plot them. However, to make people's life easier the `plot_predictions_vs_ground_truth` function can be used.
#
# * For massive streams with millions of instances, it can be unbearable to plot all at once, thus we can specify a `plot_interval` (that we want to investigate) to `plot_predictions_vs_ground_truth`. By default, the plot function will attempt to plot everything, i.e. with `plot_interval=None`, which is seldom a good idea.

# %%
from capymoa.datasets import Fried
from capymoa.regressor import evaluate_regressor
from capymoa.regressor.plot import plot_predictions_vs_ground_truth
from capymoa.regressor import AdaptiveRandomForestRegressor, KNNRegressor

stream = Fried()
kNN_learner = KNNRegressor(schema=stream.get_schema(), k=5)
ARF_learner = AdaptiveRandomForestRegressor(
    schema=stream.get_schema(), ensemble_size=10
)

# When we specify store_predictions and store_y, the results will also include all the predictions and all the ground truth y.
# It is useful for debugging and outputting the predictions elsewhere.
kNN_results = evaluate_regressor(
    stream=stream,
    learner=kNN_learner,
    window_size=5000,
    store_predictions=True,
    store_y=True,
)
# We don't need to store the ground-truth for every experiment, since it is always the same for the same stream.
ARF_results = evaluate_regressor(
    stream=stream, learner=ARF_learner, window_size=5000, store_predictions=True
)


# Plot only 200 predictions (see plot_interval).
plot_predictions_vs_ground_truth(
    kNN_results,
    ARF_results,
    plot_interval=(0, 200),
)
