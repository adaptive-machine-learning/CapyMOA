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
# # Custom Drift Detectors
#
# Some CapyMOA learners accept a custom drift detector, so they can adapt to
# distribution shifts. This includes using Python drift detectors in Java (MOA)
# learners, with CapyMOA's Java proxy ({mod}`capymoa.core.moa.proxy`). In this tutorial
# we write a basic drift detector and use it in Adaptive Random Forest.

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast

# %% [markdown]
# ## A t-test drift detector
#
# Drift detectors often follow these steps:
#
# 1. Collect and group data.
# 2. Describe each group.
# 3. Measure the dissimilarity between groups.
# 4. Raise an alarm if the dissimilarity passes a threshold.

# %%
import numpy as np
from scipy.stats import ttest_ind_from_stats

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
        self.idx = 0
        self.i = 0
        self.win_present = np.zeros((window_size,))
        self.win_past = np.zeros((window_size,))

    def add_element(self, element: float) -> None:
        self.idx += 1
        self.in_concept_change = False

        # 1. Collect and group data into present and past ring buffers.
        i = self.i % self.window_size
        self.win_past[i] = self.win_present[i]  # Move from present to past window.
        self.win_present[i] = element  # Add new element to present window.
        self.i += 1

        # Wait for both windows to fill before running the t-test.
        if self.i >= self.window_size * 2:
            # 2. Describe each group. Computing the mean and std incrementally would be
            # faster.
            mean_present, std_present = (
                np.mean(self.win_present),
                np.std(self.win_present),
            )
            mean_past, std_past = np.mean(self.win_past), np.std(self.win_past)

            # 3. Run the t-test on the group statistics.
            _, p_value = ttest_ind_from_stats(
                mean1=mean_present,
                std1=std_present,
                nobs1=self.window_size,
                mean2=mean_past,
                std2=std_past,
                nobs2=self.window_size,
                equal_var=True,
                alternative="greater",
            )

            # 4. Raise an alarm if the error rate increases.
            if p_value < self.alpha:
                # Reset, so both windows must fill again.
                self.i = 0
                self.in_concept_change = True
                self.detection_index.append(self.idx)


# %% [markdown]
# Test `TTestDetector` on a simple stream:

# %%
import matplotlib.pyplot as plt

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
# Use `TTestDetector` as the drift detector in Adaptive Random Forest:

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
# ## Limits
#
# Using Python drift detectors in Java learners has some limits:
#
# * The Java learner does not use the object you pass. Instead, it uses the object's
#   configuration to construct new detectors. So constructor parameters must be simple
#   values (numbers, strings, lists, ...) that
#   {meth}`~capymoa.base.LearnerParamsMixin.get_params` can capture, and any state in
#   the passed object is lost.
# * {func}`~capymoa.core.io.save_model` cannot save a Python detector inside a Java
#   learner because it relies on Java serialization.
# * Large numbers of Python detectors (such as one per ensemble member) can be slow.
# * Unsupported MOA methods (such as options, estimation, or delay) raise
#   {class}`~capymoa.core.moa.proxy.ProxyNotImplementedError` (see
#   {mod}`capymoa.core.moa.proxy`).
#
# If you run into any of these limits, please open an issue on the CapyMOA repository so
# we can prioritize addressing it for a future release and figure out a workaround for
# your research.
#
# ```{admonition} See also
# :class: seealso
#
# - {mod}`capymoa.core.moa.proxy`: the adapter that lets MOA call Python detectors.
# - {doc}`/notebooks/classifier/java_ensemble_python_learner`: use the same approach to
#   plug a Python learner into a Java ensemble.
# - {doc}`/contributing/moa/moa_integration`: go the other way, write a method in
#   Java and use it from Python.
# ```
