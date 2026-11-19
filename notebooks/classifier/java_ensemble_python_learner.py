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
# # Java ensembles with a Python base learner
#
# Many ensembles in CapyMOA, such as {class}`~capymoa.classifier.OnlineBagging`, run
# in Java (MOA). They are fast and well tested. This notebook shows how to use them with
# a base learner you write in Python.
#
# * We write a small Python classifier.
# * We pass it as the `base_learner` of a Java ensemble.
# * We compare the ensemble with the single learner.
#
# To learn more about writing a Python learner, see the
# [Creating a new classifier](new_learner.ipynb) tutorial.
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 19/11/2026**

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, mock_datasets

if is_nb_fast():
    mock_datasets()

# %% [markdown]
# ## A Python base learner
#
# Our learner is a nearest centroid classifier. It keeps one centroid per class, an
# exponential moving average of the instances of that class. It predicts the class of
# the closest centroid.
#
# We extend {class}`~capymoa.base.Classifier` and implement `train`, `predict` and
# `predict_proba`, like any other Python learner. Before it sees a class it can not
# predict, so `predict_proba` returns `None`.

# %%
import numpy as np

from capymoa.base import Classifier


class NearestCentroid(Classifier):
    def __init__(self, schema, random_seed=None, decay=0.99):
        """
        :param schema: Describes the structure of the datastream.
        :param random_seed: Unused, the learner has no randomness.
        :param decay: Weight of the old centroid when it is updated.
        """
        super().__init__(schema=schema, random_seed=random_seed)
        self.decay = decay
        n_classes = schema.get_num_classes()
        n_features = schema.get_num_attributes()
        self.centroids = np.zeros((n_classes, n_features))
        self.seen = np.zeros(n_classes, dtype=bool)

    def __str__(self):
        return f"NearestCentroid(decay={self.decay})"

    def train(self, instance):
        y = instance.y_index
        if not self.seen[y]:
            self.centroids[y] = instance.x
            self.seen[y] = True
        else:
            self.centroids[y] = (
                self.decay * self.centroids[y] + (1 - self.decay) * instance.x
            )

    def predict_proba(self, instance):
        if not self.seen.any():
            return None
        distance = np.linalg.norm(self.centroids - instance.x, axis=1)
        score = np.where(self.seen, np.exp(-distance), 0.0)
        return score / score.sum()

    def predict(self, instance):
        proba = self.predict_proba(instance)
        return None if proba is None else int(np.argmax(proba))


# %% [markdown]
# ## Use it in a Java ensemble
#
# Pass an instance of the learner as `base_learner`. CapyMOA wraps it in a proxy
# ({func}`~capymoa.core.moa.proxy.to_java_proxy`), so MOA can call it like a Java
# learner.
#
# MOA does not use the object you pass. It builds new learners from the class name and
# {meth}`~capymoa.base.LearnerParamsMixin.get_params`, one for each ensemble member.
# This has a few effects:
#
# * Constructor parameters must be simple values (numbers, strings, lists, ...) or other
#   CapyMOA learners, so they can be captured by `get_params()`.
# * Any training you did before is not used, and the object you pass is not changed.
# * The class must be importable by its name. A class defined in a notebook works, since
#   it lives in the `__main__` module.

# %%
from capymoa.classifier import OnlineBagging
from capymoa.datasets import Electricity

stream = Electricity()
schema = stream.get_schema()

centroid = NearestCentroid(schema, decay=0.8)
print("Parameters:", centroid.get_params())

bagging = OnlineBagging(schema, base_learner=centroid, ensemble_size=10, random_seed=1)
print("Ensemble:", bagging)

# %% [markdown]
# ## Compare the learners
#
# We compare three learners on the `Electricity` stream:
#
# * The single nearest centroid classifier.
# * {class}`~capymoa.classifier.OnlineBagging` with 10 nearest centroid members.
# * {class}`~capymoa.classifier.LeveragingBagging` with 10 nearest centroid members.
#
# Passing a dictionary to {func}`~capymoa.classifier.evaluate_classifier` runs all
# learners in one pass over the stream.

# %%
from capymoa.classifier import LeveragingBagging, evaluate_classifier
from capymoa.evaluation.plot import plot_windowed_results

learners = {
    "NearestCentroid": NearestCentroid(schema, decay=0.8),
    "OnlineBagging": bagging,
    "LeveragingBagging": LeveragingBagging(
        schema, base_learner=centroid, ensemble_size=10, random_seed=1
    ),
}

results = evaluate_classifier(
    stream, learners, max_instances=10_000, window_size=1_000
)

for name, result in results.items():
    print(f"{name}: {result['cumulative']['accuracy']:.2f}% accuracy")

plot_windowed_results(*results.values(), metric="accuracy")

# %% [markdown]
# `OnlineBagging` gives about the same accuracy as the single learner. Bagging helps
# learners that change a lot when their training data changes, such as decision trees.
# A centroid is an average, so it changes little when bagging resamples the stream, and
# all members end up alike.
#
# `LeveragingBagging` does better. It resamples more (Poisson with a mean of 6 instead of
# 1), which makes the members differ more. It also uses ADWIN to detect drift and resets
# its worst member when the stream changes.

# %% [markdown]
# ## Other ensembles
#
# The same works for other Java ensembles that take a `base_learner`, such as
# {class}`~capymoa.classifier.OnlineAdwinBagging`,
# {class}`~capymoa.classifier.OzaBoost` and
# {class}`~capymoa.classifier.StreamingRandomPatches`.
# Regressors work the same way, for example a Python regressor as the base learner of a
# Java regression ensemble.

# %% [markdown]
# ## Limits
#
# * **Speed.** Java calls Python for every instance and every member. A Python base
#   learner is much slower than a Java one, so keep ensembles small.
# * **Instance weights.** Python learners take no instance weights. The proxy trains on
#   an instance `round(weight)` times. This matches the Poisson weights of bagging, but
#   only approximates the weights used in boosting.
# * **Saving.** {func}`~capymoa.core.io.save_model` uses Java serialization, which can
#   not save a Python learner inside a Java ensemble.
# * **MOA-only methods.** Some MOA methods, such as the options or the model size, are
#   not supported. They raise
#   {class}`~capymoa.core.moa.proxy.ProxyNotImplementedError` (see
#   {mod}`capymoa.core.moa.proxy`).
