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
# # Anomaly Detection
#
# This notebook shows some basic usage of CapyMOA for anomaly detection tasks.
#
# Algorithms: `HalfSpaceTrees`, `Autoencoder` and `Online Isolation Forest`
#
# Important notes: Prior to version 0.8.2, a lower anomaly score indicated a higher likelihood of an anomaly. This has been updated so that a higher anomaly score now indicates a higher likelihood of an anomaly, aligning with the standard anomaly detection literature.
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 28/11/2025**

# %% [markdown]
# ## Creating simple anomalous data with `sklearn`
#
# * Generating a few examples and some simple anomalous data using `sklearn`.

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from capymoa.stream import NumpyStream

# generate normal data points
n_samples = 10000
n_features = 2
n_clusters = 3

# generate anomalous data points
n_anomalies = 100  # the anomaly rate is 1%

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
if is_nb_fast():
    n_samples = 1000
    n_anomalies = 10

# %%
X, y = make_blobs(
    n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42
)

anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, n_features))

# combine the normal data points with anomalies
X = np.vstack([X, anomalies])
y = np.hstack([y, [1] * n_anomalies])  # Label anomalies with 1
y[:n_samples] = 0  # Label normal points with 0

# shuffle the data
idx = np.random.permutation(n_samples + n_anomalies)
X = X[idx]
y = y[idx]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.show()

# create a NumpyStream from the combined dataset
feature_names = [f"feature_{i}" for i in range(n_features)]
target_name = "class"

# %% [markdown]
# ## Unsupervised anomaly detection for data streams
#
# * Recent research has been focused on unsupervised anomaly detection for data streams, as it is often difficult to obtain labeled data for training.
# * Instead of using evaluation functions, we first use a basic **test-then-train loop** from scratch to evaluate the model's performance.
# * Please note that higher scores indicate higher anomaly likelihood.

# %%
from capymoa.anomaly import HalfSpaceTrees
from capymoa.evaluation import AnomalyDetectionEvaluator

stream_ad = NumpyStream(
    X,
    y,
    dataset_name="AnomalyDetectionDataset",
    feature_names=feature_names,
    target_name=target_name,
    target_type="categorical",
)
learner = HalfSpaceTrees(stream_ad.get_schema())
evaluator = AnomalyDetectionEvaluator(stream_ad.get_schema())
while stream_ad.has_more_instances():
    instance = stream_ad.next_instance()
    score = learner.score_instance(instance)
    evaluator.update(instance.y_index, score)
    learner.train(instance)

auc = evaluator.auc()
print(f"AUC: {auc:.2f}")

# %% [markdown]
# ## High-level evaluation functions
#
# * CapyMOA provides `prequential_evaluation_anomaly` as a high level function to assess anomaly detectors.

# %% [markdown]
# ### `prequential_evaluation_anomaly`
# In this example, we use the `prequential_evaluation_anomaly` function with `plot_windowed_results` to plot AUC for HalfSpaceTrees on the synthetic data stream.

# %%
from capymoa.anomaly import HalfSpaceTrees
from capymoa.evaluation import prequential_evaluation_anomaly
from capymoa.evaluation.visualization import plot_windowed_results

stream_ad = NumpyStream(
    X,
    y,
    dataset_name="AnomalyDetectionDataset",
    feature_names=feature_names,
    target_name=target_name,
    target_type="categorical",
)
hst = HalfSpaceTrees(schema=stream_ad.get_schema())

results_hst = prequential_evaluation_anomaly(
    stream=stream_ad, learner=hst, window_size=1000
)

print(f"AUC: {results_hst.auc()}")
display(results_hst.windowed.metrics_per_window())
plot_windowed_results(results_hst, metric="auc", save_only=False)

# %% [markdown]
# ### Autoencoder

# %%
from capymoa.anomaly import Autoencoder
from capymoa.evaluation import prequential_evaluation_anomaly
from capymoa.evaluation.visualization import plot_windowed_results

stream_ad = NumpyStream(
    X,
    y,
    dataset_name="AnomalyDetectionDataset",
    feature_names=feature_names,
    target_name=target_name,
    target_type="categorical",
)
ae = Autoencoder(schema=stream_ad.get_schema(), hidden_layer=1)

results_ae = prequential_evaluation_anomaly(
    stream=stream_ad, learner=ae, window_size=1000
)

print(f"AUC: {results_ae.auc()}")
display(results_ae.windowed.metrics_per_window())
plot_windowed_results(results_ae, metric="auc", save_only=False)

# %% [markdown]
# ### Online Isolation Forest

# %%
from capymoa.anomaly import OnlineIsolationForest
from capymoa.evaluation import prequential_evaluation_anomaly
from capymoa.evaluation.visualization import plot_windowed_results

stream_ad = NumpyStream(
    X,
    y,
    dataset_name="AnomalyDetectionDataset",
    feature_names=feature_names,
    target_name=target_name,
    target_type="categorical",
)
oif = OnlineIsolationForest(schema=stream_ad.get_schema(), num_trees=10)

results_oif = prequential_evaluation_anomaly(
    stream=stream_ad, learner=oif, window_size=1000
)

print(f"AUC: {results_oif.auc()}")
display(results_oif.windowed.metrics_per_window())
plot_windowed_results(results_oif, metric="auc", save_only=False)

# %% [markdown]
# ## Comparing algorithms

# %%
plot_windowed_results(
    results_hst, results_ae, results_oif, metric="auc", save_only=False
)
