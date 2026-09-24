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
# # Semi-supervised Learning
#
# * Preparing and executing partially and delayed labeling experiments.
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 05/08/2026**

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, mock_datasets

if is_nb_fast():
    mock_datasets()

# %%
from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_ssl_evaluation
from capymoa.evaluation.visualization import plot_windowed_results

# %% [markdown]
# ## Learning using a SSL classifier
#
# * This example uses the OSNN algorithm to learn from a stream with only 1% labeled data.
# * We utilise the `prequential_ssl_evaluation()` function to simulate the absence of labels (`label_probability`) and delays (`delay_length`).
# * The results yielded by `prequential_ssl_evaluation()` include more information in comparison to `prequential_evaluation()`, such as the number of unlabeled instances (`unlabeled`) and the unlabeled ratio (`unlabeled_ratio`).

# %%
help(prequential_ssl_evaluation)

# %%
from capymoa.ssl import OSNN

stream = Electricity()

osnn = OSNN(schema=stream.get_schema(), optim_steps=10)

results_osnn = prequential_ssl_evaluation(
    stream=stream,
    learner=osnn,
    label_probability=0.01,
    window_size=100,
    max_instances=2000,
)

# The results are stored in a dictionary.
display(results_osnn)

print(
    results_osnn["cumulative"].accuracy()
)  # Test-then-train accuracy, i.e. cumulatively, not windowed.

# Plotting over time (default: classifications correct (percent) i.e. accuracy)
results_osnn.learner = "OSNN"
plot_windowed_results(results_osnn, metric="accuracy")

# %% [markdown]
# ## Using a supervised model
#
# * If a supervised model is used with `prequential_ssl_evaluation()`, it will only be trained on the labeled data.

# %%
from capymoa.classifier import StreamingRandomPatches

srp10 = StreamingRandomPatches(schema=stream.get_schema(), ensemble_size=10)

results_srp10 = prequential_ssl_evaluation(
    stream=stream,
    learner=srp10,
    label_probability=0.01,
    window_size=100,
    max_instances=2000,
)
print(results_srp10["cumulative"].accuracy())

# %% [markdown]
# ## SLEADE
#
# * SLEADE is another semi-supervised learning algorithm

# %%
from capymoa.ssl import SLEADE

stream = Electricity()

sleade = SLEADE(schema=stream.get_schema(), ensemble_size=10)

results_sleade = prequential_ssl_evaluation(
    stream=stream,
    learner=sleade,
    label_probability=0.01,
    window_size=100,
    max_instances=2000,
)

print(results_sleade["cumulative"].accuracy())

# %% [markdown]
# ## Comparing a SSL classifier to a supervised classifier

# %%
# Plotting all the results together
# Adding an experiment_id to the results dictionary allows controlling the legend of each learner.
results_osnn.learner = "OSNN"
results_srp10.learner = "SRP10"
results_sleade.learner = "SLEADE"

plot_windowed_results(results_osnn, results_srp10, results_sleade, metric="accuracy")

# %% [markdown]
# ## Delay example
#
# * In this section we compare the effect of delay on a stream.
# * It is particularly interesting to see the effect after a drift takes place.

# %%
from capymoa.classifier import HoeffdingTree
from capymoa.stream.drift import AbruptDrift, DriftStream
from capymoa.stream.generator import SEA

## Creating a stream with drift
sea2drifts = DriftStream(
    stream=[
        SEA(function=1),
        AbruptDrift(position=25000),
        SEA(function=2),
        AbruptDrift(position=50000),
        SEA(function=3),
    ]
)


ht_immediate = HoeffdingTree(schema=sea2drifts.get_schema())
ht_delayed = HoeffdingTree(schema=sea2drifts.get_schema())

results_ht_immediate = prequential_ssl_evaluation(
    stream=sea2drifts,
    learner=ht_immediate,
    label_probability=0.1,
    window_size=1000,
    max_instances=100000,
)

results_ht_delayed_1000 = prequential_ssl_evaluation(
    stream=sea2drifts,
    learner=ht_delayed,
    label_probability=0.01,
    delay_length=1000,  # adding the delay
    window_size=1000,
    max_instances=100000,
)

results_ht_immediate.learner = "HT_immediate"
results_ht_delayed_1000.learner = "HT_delayed_1000"

print(f"Accuracy immediate: {results_ht_immediate['cumulative'].accuracy()}")
print(
    f"Accuracy delayed by 1000 instances: {results_ht_delayed_1000['cumulative'].accuracy()}"
)

plot_windowed_results(results_ht_immediate, results_ht_delayed_1000, metric="accuracy")
