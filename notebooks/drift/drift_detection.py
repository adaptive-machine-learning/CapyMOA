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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Drift Detection in CapyMOA
#
# In this tutorial, we show how to conduct drift detection using CapyMOA.
#
# * A first example using ADWIN, and how to inspect what a detector reports.
# * Comparing the detectors available in CapyMOA on the same stream.
# * Evaluating detectors against known drift locations, including detection delay.
# * Multivariate drift detection using ABCD.
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 07/08/2026**

# %% [markdown]
# ## A first example with ADWIN
#
# We start with the simplest possible stream: a single sequence of numbers with one abrupt change in the middle. The first 1000 values are drawn from `{0, 1}` and the last 1000 from `{6, ..., 11}`, so a drift occurs at instance **1000**.
#
# Knowing where the drift is lets us judge the detector later. We seed the generator so the numbers quoted in this tutorial are the ones you will get when you run it.

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DRIFT_AT = 1000

rng = np.random.default_rng(42)
data_stream = rng.integers(0, 2, size=2000).astype(float)
data_stream[DRIFT_AT:] = rng.integers(6, 12, size=1000)

# %%
plt.figure(figsize=(9, 3))
plt.plot(data_stream, linewidth=0.5, alpha=0.8)
plt.axvline(DRIFT_AT, color="red", linestyle="--", label=f"drift at {DRIFT_AT}")
plt.title("A univariate stream with one abrupt drift")
plt.xlabel("# Instances")
plt.ylabel("Value")
plt.legend()
plt.show()

# %% [markdown]
# ADWIN (ADaptive WINdowing) is a popular detection algorithm. It maintains a window of recent values and splits it in every possible way, signalling a change when the means of two sub-windows differ by more than a statistical bound. The `delta` parameter is the confidence level: smaller values make it more conservative.
#
# **Reference:** Bifet, A., & Gavalda, R. (2007). *Learning from time-changing data with adaptive windowing.* SIAM International Conference on Data Mining, pp. 443-448.

# %%
from capymoa.drift.detectors import ADWIN

detector = ADWIN(delta=0.001)

for i in range(len(data_stream)):
    detector.add_element(data_stream[i])
    if detector.detected_change():
        print(f"Change detected in data: {data_stream[i]} - at index: {i}")

# %%
plt.figure(figsize=(9, 3))
plt.plot(data_stream, linewidth=0.5, alpha=0.6)
plt.axvline(DRIFT_AT, color="red", linestyle="--", label=f"true drift ({DRIFT_AT})")
for n, idx in enumerate(detector.detection_index):
    plt.axvline(idx, color="green", label="detection" if n == 0 else None)
plt.title("ADWIN detections against the true drift location")
plt.xlabel("# Instances")
plt.ylabel("Value")
plt.legend()
plt.show()

# %% [markdown]
# ### What the detector records
#
# Beyond signalling changes as they happen, every CapyMOA detector keeps a record of what it has seen. Three attributes are worth knowing:
#
# - **`detection_index`**: the instance indices at which the detector signalled a change.
# - **`warning_index`**: indices where the detector entered a *warning* zone, meaning it suspects a change but has not committed to one. Warnings are typically used to start building a replacement model before the drift is confirmed.
# - **`idx`**: how many instances the detector has processed in total.
#
# Note that ADWIN reports **no warnings**. Not every algorithm has a warning zone. ADWIN, CUSUM, PageHinkley and SEED signal changes directly, while DDM, RDDM, EWMAChart, STEPD, OPTWIN and the HDDM family do use one. The comparison in the next section makes this visible.

# %%
print(f"Detections ....... {detector.detection_index}")
print(f"Warnings ......... {detector.warning_index}")
print(f"Instances seen ... {detector.idx}")

delay = detector.detection_index[0] - DRIFT_AT
print(f"\nThe first detection came {delay} instances after the drift at {DRIFT_AT}.")

# %% [markdown]
# ## Comparing the available detectors
#
# CapyMOA ships a number of drift detectors, all with the same interface, so the loop above works unchanged for any of them. Running all of them over the same stream shows how differently they behave out of the box.
#
# Notice that some detectors fire far more often than others. `HDDMAverage` and `HDDMWeighted` report dozens of changes where the stream contains exactly one. That is not a defect so much as a default: we have not tuned any hyperparameters here, and these detectors are simply more sensitive. The warning column also shows which algorithms have a warning zone at all.

# %%
from capymoa.drift import detectors

results = {}
for detector_name in detectors.__all__:
    detector_cls = getattr(detectors, detector_name)
    if getattr(detector_cls, "REQUIRES_FIT", True):
        continue
    if detector_name == "STUDD":
        continue

    d = detector_cls()
    for i in range(len(data_stream)):
        d.add_element(float(data_stream[i]))
        d.detected_change()

    results[detector_name] = {
        "detections": len(d.detection_index),
        "warnings": len(d.warning_index),
        "first_detection": d.detection_index[0] if d.detection_index else None,
    }

pd.DataFrame(results).T

# %% [markdown]
# ## Evaluating drift detectors
#
# Assuming the drift locations are known, you can evaluate detectors using **EvaluateDetector** class.
#
# This class takes a parameter called **max_delay**, which is the maximum number of instances for which we consider a detector to have detected a change. After **max_delay** instances, we assume that the change is obvious and has been missed by the detector. This evaluation approach was recently investigated (see the KDD reference below).
#
# The `EvaluateDetector` class takes two arguments for evaluating detectors:
# - The `locations` of the drift (ground-truth) i.e. `trues`
# - The `locations` of the drift detections i.e. `preds`
#
# **Reference:** Cerqueira, V., Gomes, H. M., Heyden, M., Pfahringer, B., & Bifet, A. (2026). *A Framework for Evaluating and Benchmarking Concept Drift Detection Methods.* ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

# %%
from capymoa.drift.eval_detector import EvaluateDriftDetector

drift_eval = EvaluateDriftDetector(max_delay=200)

trues = np.array([DRIFT_AT])
preds = detector.detection_index

metrics = drift_eval.calc_performance(trues, preds, tot_n_instances=detector.idx)
metrics

# %% [markdown]
# ### Detection delay: `mdt` and `ndt`
#
# Two of those metrics describe *how late* the detector was, averaged over the drifts it actually found:
#
# - **`mdt`**: the mean delay, in instances.
# - **`ndt`**: that same delay divided by `max_delay`, so 0 means the drift was caught as it began and 1 means it was caught just as it would have become obvious anyway.
#
# ADWIN detected the drift 24 instances after it happened, and we allowed 200, so `mdt` is `24.0` and `ndt` is `24 / 200 = 0.12`. The detector used about an eighth of the delay we were willing to tolerate.
#
# The reason to have both is that `mdt` alone cannot be compared across streams. The table below evaluates a fixed set of detection points against the same drift at 1000, under three different tolerances. Look at the `detected_at = 1150` rows: `mdt` is 150 instances in both cases where the drift was found, but `ndt` is **0.75** against a tolerance of 200 and **0.30** against a tolerance of 500. The detector behaved identically; only the standard it was held to changed.
#
# That is why `ndt` is the one to average over datasets. A raw delay of 150 instances means something quite different depending on how much delay was acceptable in the first place. `ndt` reaches exactly 1.0 when the detection lands on the deadline, and both metrics are `NaN` when the drift was missed altogether, since there is then no delay to average.

# %%
rows = []
for max_delay in (100, 200, 500):
    for detected_at in (1000, 1150, 1200):
        m = EvaluateDriftDetector(max_delay=max_delay).calc_performance(
            trues=np.array([1000]), preds=np.array([detected_at]), tot_n_instances=2000
        )
        rows.append(
            {
                "max_delay": max_delay,
                "detected_at": detected_at,
                "detected": m.tp == 1,
                "mdt": m.mdt,
                "ndt": m.ndt,
            }
        )

pd.DataFrame(rows)

# %% [markdown]
# ## Multivariate drift detection
#
# Everything so far monitored a single sequence of numbers. ABCD (Adaptive Bernstein Change Detector) instead monitors a **multivariate** input: it fits an encoder-decoder model of the incoming feature vectors, tracks how well that model reconstructs them, and signals when the reconstruction error shifts.
#
# Because ABCD watches the *input distribution*, it needs drift that actually changes the inputs. A stream whose labels change while its feature distribution stays put is invisible to it.
#
# **Reference:** Heyden, M., Fouché, E., Arzamasov, V., Fenn, T., Kalinke, F., & Böhm, K. (2024). *Adaptive Bernstein change detector for high-dimensional data streams.* Data Mining and Knowledge Discovery, 38(3), 1334-1363.

# %% [markdown]
# ### Building a stream with known drift points
#
# To evaluate a multivariate detector we need multivariate data whose drift locations we know. We can build exactly that with the `DriftStream` API from [Simulating concept drifts with the DriftStream API](https://capymoa.org/notebooks/drift/drift_streams.html).
#
# `RandomRBFGenerator` places a fixed number of centroids in feature space, then produces each instance by picking a centroid and adding Gaussian noise. Its `model_random_seed` is what decides where those centroids land, so `rbf(1)`, `rbf(99)` and `rbf(7)` are three different *layouts* of centroids, and therefore three genuinely different input distributions. The `instance_random_seed` only controls which centroid is drawn each time, so on its own it would give us more samples of the same distribution rather than a new concept.
#
# This is the same recipe the ABCD paper uses to build its synthetic RBF stream, where changes are created by incrementing the generator seed so that the centroids move.
#
# Note that we use `AbruptDrift` and the _Position API_ for drift generation (See more about simulating drifts on [Simulating concept drifts with the DriftStream API](https://capymoa.org/notebooks/drift/drift_streams.html)). Because we placed the drifts ourselves, the ground truth is exact, so the result feeds straight into `EvaluateDriftDetector`.

# %%
from capymoa.drift.detectors import ABCD
from capymoa.stream.drift import AbruptDrift, DriftStream
from capymoa.stream.generator import RandomRBFGenerator

TRUE_DRIFTS = [2000, 4000]


def rbf(seed):
    return RandomRBFGenerator(
        model_random_seed=seed,
        instance_random_seed=seed,
        number_of_attributes=6,
        number_of_centroids=20,
    )


stream = DriftStream(
    stream=[
        rbf(1),
        AbruptDrift(position=TRUE_DRIFTS[0]),
        rbf(99),
        AbruptDrift(position=TRUE_DRIFTS[1]),
        rbf(7),
    ]
)
# get_concept_counts() replays the routing without generating the data,
# confirming how many instances each concept actually contributed.
print(f"instances per concept: {stream.get_concept_counts()}")

# %%
n_check_instances = 3000

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
if is_nb_fast():
    n_check_instances = 300

# %%
# Confirming the seeds really do move the input distribution: the mean of each
# feature differs from concept to concept, which is what ABCD has to notice.
for seed in (1, 99, 7):
    concept = rbf(seed)
    concept.restart()
    x = np.array([concept.next_instance().x for _ in range(n_check_instances)])
    print(f"rbf({seed:2d}) feature means: {np.round(x.mean(axis=0), 3)}")

# %% [markdown]
# ### Running ABCD on the stream
#
# ABCD's `maximum_absolute_value` bounds how large an input value it expects. The default of `1` is conservative for this data and makes the statistical test slow to react, so we set it lower here; section 4.3 looks at that parameter directly.
#
# We feed the detector `instance.x`, the feature vector, since ABCD is unsupervised and never sees the label. Plotting the reconstruction loss alongside the true drift positions shows what the detector is reacting to: the loss climbs after each change in centroid layout, and the detector fires once that climb is large enough to pass its test.

# %%
abcd = ABCD(model_id="pca", maximum_absolute_value=0.2)

loss_values = []
i = 0
n_abcd_instances = 6000

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
if is_nb_fast():
    n_abcd_instances = 600

# %%
while stream.has_more_instances() and i < n_abcd_instances:
    instance = stream.next_instance()
    i += 1
    abcd.add_element(instance.x)
    loss_values.append(abcd.loss())
    if abcd.detected_change():
        print(f"Change detected at index: {i}")

# %%
plt.figure(figsize=(9, 3))
plt.plot(pd.Series(loss_values).rolling(50).mean())
for n, true_drift in enumerate(TRUE_DRIFTS):
    plt.axvline(
        true_drift, color="red", linestyle="--", label="true drift" if n == 0 else None
    )
for n, idx in enumerate(abcd.detection_index):
    plt.axvline(idx, color="green", label="detection" if n == 0 else None)
plt.title("ABCD reconstruction loss, with true drifts and detections")
plt.xlabel("# Instances")
plt.ylabel("Reconstruction loss")
plt.legend()
plt.show()

# %% [markdown]
# #### Evaluating the detections
#
# Because we know where the drifts really were, we can score ABCD exactly as we scored ADWIN in section 3. ABCD is _slow_ to react since establishing that a 6-dimensional distribution has moved takes more evidence than establishing it for a single series, so we allow a correspondingly larger `max_delay`. However, the `max_delay` should in practice be set according to the expectations with respect to the algorithms, e.g. we are assuming that for this particular problem waiting for 500 instances for a detection to be confirmed is reasonable.

# %%
abcd_eval = EvaluateDriftDetector(max_delay=500)
abcd_metrics = abcd_eval.calc_performance(
    trues=np.array(TRUE_DRIFTS), preds=abcd.detection_index, tot_n_instances=i
)

print(f"detections ... {abcd.detection_index}")
print(f"true drifts .. {TRUE_DRIFTS}")
print(f"recall ....... {abcd_metrics.recall}")
print(f"mdt .......... {abcd_metrics.mdt}")
print(f"ndt .......... {abcd_metrics.ndt}")

# %% [markdown]
# ### Sensitivity, and what tuning cannot fix
#
# Every detector exposes some control over how much evidence it demands before it commits. ADWIN has `delta`, a confidence level. ABCD has `maximum_absolute_value`, which bounds the reconstruction error it expects to see and so decides how large a deviation has to be before the Bernstein bound is exceeded. Lowering it makes the test more sensitive.
#
# The useful question is not "which value is best" but "what does turning this knob actually buy". Now that we have ground truth and the metrics from section 3, we can measure it rather than guess. The sweep below runs the same stream through ABCD at four settings and evaluates each one.
#
# Two things are worth reading out of the result. Reaction speed does improve as the parameter falls: `ndt` drops steadily, so the detector is using less of its allowed delay. But `recall` stays at 0.5 throughout, because no setting catches the second drift inside our 500-instance window. Sensitivity buys speed on evidence that exists; it cannot manufacture evidence that does not.
#
# The second drift is simply the harder one. At `maximum_absolute_value=0.2` ABCD does fire at instance 4599, 599 instances after the drift, which is real detection but later than we said we would accept. Raising `max_delay` to 600 turns that same run into a recall of 1.0. Which is correct depends entirely on how quickly your application actually needs to know about changes.

# %%
n_sweep_instances = 6000

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
if is_nb_fast():
    n_sweep_instances = 600

# %%
sweep = []
for max_abs in (1.0, 0.5, 0.2, 0.1):
    stream.restart()
    tuned = ABCD(model_id="pca", maximum_absolute_value=max_abs)

    n = 0
    while stream.has_more_instances() and n < n_sweep_instances:
        tuned.add_element(stream.next_instance().x)
        n += 1

    metrics = EvaluateDriftDetector(max_delay=500).calc_performance(
        trues=np.array(TRUE_DRIFTS),
        preds=np.array(tuned.detection_index),
        tot_n_instances=n,
    )
    sweep.append(
        {
            "maximum_absolute_value": max_abs,
            "detections": tuned.detection_index,
            "recall": metrics.recall,
            "mdt": metrics.mdt,
            "ndt": round(metrics.ndt, 3),
        }
    )

pd.DataFrame(sweep)
