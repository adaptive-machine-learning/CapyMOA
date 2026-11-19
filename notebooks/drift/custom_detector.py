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
# # A custom drift detector in Adaptive Random Forest
#
# {class}`~capymoa.classifier.AdaptiveRandomForestClassifier` (ARF) runs in Java (MOA).
# Each tree has a drift detector that watches its errors. When the detector fires, the
# tree is replaced. This notebook shows how to write a drift detector in Python and use
# it inside ARF.
#
# * We write a detector that compares two windows with a t-test.
# * We test it on a simple stream.
# * We use it in ARF and compare it with DDM, a common detector, and with no detection.
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 19/11/2026**

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast

# %% [markdown]
# ## A t-test drift detector
#
# We keep the detector as simple as we can, to show what is needed. Any other test or
# distance would work the same way.
#
# The detector splits its input into windows of `window_size` values. When a window is
# full, it compares the window with the one before using Welch's t-test
# ({func}`scipy.stats.ttest_ind`). If the p-value is below `alpha`, the means differ and
# it signals a drift.
#
# ARF gives each detector a stream of errors: `1` if the tree was wrong and `0` if it
# was right. So the detector checks if the error rate has changed.
#
# We extend {class}`~capymoa.drift.base_detector.BaseDriftDetector` and implement
# `add_element`. It sets `in_concept_change`, which
# {meth}`~capymoa.drift.base_detector.BaseDriftDetector.detected_change` returns, and
# records where drifts happen in `detection_index`.

# %%
from scipy.stats import ttest_ind

from capymoa.drift.base_detector import BaseDriftDetector


class TTestDetector(BaseDriftDetector):
    def __init__(self, window_size: int = 100, alpha: float = 0.01):
        """
        :param window_size: Number of values in each window.
        :param alpha: Signal a drift when the p-value is below this.
        """
        super().__init__()
        self.window_size = window_size
        self.alpha = alpha
        self.reference = None
        self.current = []

    def add_element(self, element: float) -> None:
        self.idx += 1
        self.in_concept_change = False

        self.current.append(element)
        if len(self.current) < self.window_size:
            return

        # The window is full. Compare it with the window before.
        if self.reference is not None:
            p_value = ttest_ind(self.reference, self.current, equal_var=False).pvalue
            if p_value < self.alpha:
                self.in_concept_change = True
                self.detection_index.append(self.idx)
        self.reference = self.current
        self.current = []


# %% [markdown]
# ## Test on a simple stream
#
# We make a stream of errors. The error rate is 10% for the first 2000 values and 40%
# after, so there is a drift at value **2000**.

# %%
import matplotlib.pyplot as plt
import numpy as np

DRIFT_AT = 2000

rng = np.random.default_rng(42)
errors = (rng.random(4000) < 0.1).astype(float)
errors[DRIFT_AT:] = rng.random(2000) < 0.4

detector = TTestDetector(window_size=100, alpha=0.01)
for error in errors:
    detector.add_element(error)

print("Drifts detected at:", detector.detection_index)

plt.figure(figsize=(9, 3))
plt.plot(np.convolve(errors, np.ones(100) / 100, mode="valid"), label="error rate")
plt.axvline(DRIFT_AT, color="red", linestyle="--", label="true drift")
for i in detector.detection_index:
    plt.axvline(i, color="green", linestyle=":", label="detected drift")
plt.xlabel("# Instances")
plt.ylabel("Error rate (moving average)")
plt.legend()
plt.show()

# %% [markdown]
# The detector finds the drift one window after it happens, with no false alarms.

# %% [markdown]
# ## Use it in Adaptive Random Forest
#
# ARF takes two detectors for each tree:
#
# * `warning_detection_method`: when it fires, the tree starts training a background tree.
# * `drift_detection_method`: when it fires, the background tree replaces the tree.
#
# We use a {class}`~capymoa.stream.drift.DriftStream` made of three random tree concepts
# with abrupt drifts at 5000 and 10000. We compare three versions of ARF with 10 trees:
#
# * ARF with our t-test detector. The warning detector uses a larger `alpha`, so it
#   fires before the drift detector.
# * ARF with {class}`~capymoa.drift.detectors.DDM`, a common drift detector.
# * ARF with drift detection turned off.
#
# CapyMOA wraps the Python detector in a proxy
# ({func}`~capymoa.core.moa.proxy.to_java_proxy`), so MOA can call it like a Java
# detector. MOA builds a new detector for each tree from the class name and
# {meth}`~capymoa.base.LearnerParamsMixin.get_params`, so trees do not share state.

# %%
from capymoa.classifier import AdaptiveRandomForestClassifier, evaluate_classifier
from capymoa.drift.detectors import DDM
from capymoa.evaluation.plot import plot_windowed_results
from capymoa.stream.drift import AbruptDrift, DriftStream
from capymoa.stream.generator import RandomTreeGenerator

max_instances = 15_000

stream = DriftStream(
    stream=[
        RandomTreeGenerator(tree_random_seed=1),
        AbruptDrift(position=5000),
        RandomTreeGenerator(tree_random_seed=2),
        AbruptDrift(position=10000),
        RandomTreeGenerator(tree_random_seed=3),
    ]
)
schema = stream.get_schema()

learners = {
    "ARF(t-test)": AdaptiveRandomForestClassifier(
        schema,
        ensemble_size=10,
        drift_detection_method=TTestDetector(window_size=100, alpha=0.01),
        warning_detection_method=TTestDetector(window_size=100, alpha=0.05),
        random_seed=1,
    ),
    "ARF(DDM)": AdaptiveRandomForestClassifier(
        schema,
        ensemble_size=10,
        drift_detection_method=DDM(),
        warning_detection_method=DDM(),
        random_seed=1,
    ),
    "ARF(no detection)": AdaptiveRandomForestClassifier(
        schema, ensemble_size=10, disable_drift_detection=True, random_seed=1
    ),
}

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
if is_nb_fast():
    max_instances = 1_000

# %%
results = evaluate_classifier(
    stream, learners, max_instances=max_instances, window_size=1_000
)

for name, result in results.items():
    print(f"{name}: {result['cumulative']['accuracy']:.2f}% accuracy")

plot_windowed_results(*results.values(), metric="accuracy")

# %% [markdown]
# Before the first drift, all three versions are close. After each drift, ARF without
# detection keeps its old trees and recovers slowly. With our t-test detector, ARF
# replaces trees that start to fail and recovers almost as well as with DDM.
#
# ## Limits
#
# * **Fixed windows.** The detector only checks once per window, so it finds a drift up
#   to `window_size` values late. ADWIN picks its window size by itself.
# * **False alarms.** Each full window is a new test. With `alpha=0.01`, about 1 in 100
#   tests on a stable stream signals a drift by chance.
# * **Speed.** Java calls Python for every instance and every tree, so a Python detector
#   is slower than a Java one.
