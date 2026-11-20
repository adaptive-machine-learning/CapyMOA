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
# # Python Base Learners in Java Ensembles
#
# CapyMOA lets you use custom learners in existing ensembles, including ensembles
# written in Java. This lets you extend MOA's Java algorithms without writing any Java.
# In this tutorial we write a simple custom classifier and use it in
# {class}`~capymoa.classifier.LeveragingBagging`.

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, mock_datasets

if is_nb_fast():
    mock_datasets()

# %% [markdown]
# ## Python Base Learner
#
# For this example, our base learner is the
# [nearest centroid classifier](https://en.wikipedia.org/wiki/Nearest_centroid_classifier).

# %%
import numpy as np

from capymoa.base import Classifier


class NearestCentroid(Classifier):
    def __init__(self, schema, random_seed=None, decay=0.99):
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


# %% [markdown]
# ## Ensemble
#
# Pass an instance of the learner as `base_learner`. CapyMOA automatically wraps it in
# an adapter ({func}`~capymoa.core.moa.proxy.to_java_proxy`) so MOA can use it.
#
# The ensemble does not use the object you pass. Instead, it uses it as a prototype to
# configure its members. This lets each member have its own seed, state, and schema.
#
# This has some limits:
#
# * Constructor parameters must be simple values (numbers, strings, lists, ...) or other
#   CapyMOA learners, so they can be captured by
#   {meth}`~capymoa.base.LearnerParamsMixin.get_params`.
# * Model state is discarded, so any training done before is lost.

# %%
from capymoa.classifier import LeveragingBagging
from capymoa.datasets import Electricity

stream = Electricity()
schema = stream.get_schema()

centroid = NearestCentroid(schema, decay=0.8)
print("Parameters:", centroid.get_params())

bagging = LeveragingBagging(
    schema, base_learner=centroid, ensemble_size=10, random_seed=1
)
print("Ensemble:", bagging)

# %% [markdown]
# The ensemble performs much better than the nearest centroid classifier on its own.

# %%
from capymoa.classifier import evaluate_classifier
from capymoa.evaluation.plot import plot_windowed_results

learners = {
    "NearestCentroid": NearestCentroid(schema, decay=0.8),
    "LeveragingBagging": bagging,
}

results = evaluate_classifier(stream, learners, max_instances=10_000, window_size=1_000)

for name, result in results.items():
    print(f"{name}: {result['cumulative']['accuracy']:.2f}% accuracy")

plot_windowed_results(*results.values(), metric="accuracy")

# %% [markdown]
# ## Limits
#
# * **Speed.** Python base learners (members) can be slow when used in large numbers.
# * **Instance weights.** Some MOA learners weight training instances. CapyMOA's Java
#   proxy applies a weight by training on the instance repeatedly.
# * **Saving.** {func}`~capymoa.core.io.save_model` uses Java serialization, which cannot
#   save a Python learner inside a Java ensemble.
# * **MOA-only methods.** Some MOA methods, such as the options or the model size, are
#   not supported. They raise {class}`~capymoa.core.moa.proxy.ProxyNotImplementedError`
#   (see {mod}`capymoa.core.moa.proxy`). Please open an issue if you hit one.
