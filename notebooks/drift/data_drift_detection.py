# ---
# jupyter:
#   jupytext:
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
# # Data Drift Detection in CapyMOA
#
# This tutorial shows how to detect **data drift** (changes in the input data
# distribution) using detectors in `capymoa.drift.detectors`.
#
# Unlike *concept drift* detectors, which track prediction errors, data drift
# detectors compare recent observations against a reference distribution to
# determine whether the data-generating process has changed.
#
# In this tutorial we cover:
#
# * A first example with the Kolmogorov–Smirnov test.
# * Inspecting the `DataDriftResult` object.
# * Comparing statistical test detectors (KS, Anderson–Darling, Cramér–von Mises).
# * Comparing distance-based detectors (KL, JS, PSI, Hellinger, Wasserstein, Energy Distance).
# * Multivariate detectors (MMD, D3, BNDM).
# * Categorical features with the Chi-Square test.
# * Batch comparison with `compare()`.
# * Auto-fit mode for fully streaming workflows.
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 15/09/2026**

# %% nbsphinx="hidden"
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast

# Reduce sizes when running in fast mode (CI / nbmake)
if is_nb_fast():
    _N_REF = 50
    _WINDOW = 20
    _STREAM_LEN = 100
else:
    _N_REF = 500
    _WINDOW = 100
    _STREAM_LEN = 800

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# %% [markdown]
# ## Synthetic data with a known drift
#
# We create a simple two-feature stream where the first half is drawn from
# `N(0, 1)` and the second half shifts to `N(3, 1)`. This makes it easy to
# verify whether detectors fire at the right moment.

# %%
N_REF = _N_REF
WINDOW = _WINDOW
STREAM_LEN = _STREAM_LEN
N_FEATURES = 2
SHIFT = 3.0

# Reference data (stable distribution)
X_ref = rng.normal(0, 1, size=(N_REF, N_FEATURES))

# Streaming data: first half is same distribution, second half is shifted
half = STREAM_LEN // 2
X_stream = np.vstack(
    [
        rng.normal(0, 1, size=(half, N_FEATURES)),
        rng.normal(SHIFT, 1, size=(half, N_FEATURES)),
    ]
)

print(f"Reference: {X_ref.shape},  Stream: {X_stream.shape}")
print(f"Drift occurs at observation {half}")

# %% [markdown]
# ## 1. A first example with Kolmogorov–Smirnov
#
# The Kolmogorov–Smirnov (KS) test compares the empirical CDFs of the
# reference and test windows, feature by feature. It is one of the simplest
# and most widely used univariate data-drift detectors.
#
# The workflow is:
#
# 1. **Fit** the detector on a reference dataset.
# 2. **Stream** new observations one at a time with `add_element()`.
# 3. **Check** for drift with `detected_change()`.

# %%
from capymoa.drift.detectors import KolmogorovSmirnov

ks = KolmogorovSmirnov(window_size=WINDOW)
ks.fit(X_ref)

detections = []
for i, x in enumerate(X_stream):
    ks.add_element(x)
    if ks.detected_change():
        detections.append(i)

print(f"KS detected drift at observations: {detections}")
print(f"(True drift is at observation {half})")

# %% [markdown]
# ## 2. Inspecting a `DataDriftResult`
#
# Instead of a simple boolean, you can call `compare()` on any test window
# to get a full `DataDriftResult`. This object contains:
#
# - `is_drift` – overall drift decision.
# - `statistic` – aggregated test statistic.
# - `p_value` – aggregated p-value (for statistical tests).
# - `feature_is_drift` – per-feature drift decisions.
# - `feature_statistics` – per-feature test statistics.
# - `feature_p_values` – per-feature p-values.

# %%
ks_fresh = KolmogorovSmirnov(window_size=WINDOW)
ks_fresh.fit(X_ref)

# Compare against the shifted portion of the stream
result = ks_fresh.compare(X_stream[half : half + WINDOW])

print(f"is_drift      : {result.is_drift}")
print(f"statistic     : {result.statistic:.4f}")
print(f"p_value       : {result.p_value:.2e}")
print(f"feature_drift : {result.feature_is_drift}")
print(f"feature_stats : {result.feature_statistics}")
print(f"feature_pvals : {result.feature_p_values}")

# %% [markdown]
# ## 3. Statistical test detectors
#
# CapyMOA includes three univariate statistical-test detectors that produce
# p-values and use Bonferroni correction across features:
#
# | Detector | Key idea |
# |---|---|
# | **KolmogorovSmirnov** | Max difference between empirical CDFs |
# | **AndersonDarling** | Weighted CDF comparison (more sensitive in tails) |
# | **CramerVonMises** | Integrated squared CDF difference |
#
# We compare them on the same synthetic stream.

# %%
from capymoa.drift.detectors import (
    AndersonDarling,
    CramerVonMises,
    KolmogorovSmirnov,
)

stat_detectors = {
    "KS": KolmogorovSmirnov(window_size=WINDOW),
    "Anderson-Darling": AndersonDarling(window_size=WINDOW),
    "Cramér-von Mises": CramerVonMises(window_size=WINDOW),
}

results = {}
for name, det in stat_detectors.items():
    det.fit(X_ref)
    dets = []
    for i, x in enumerate(X_stream):
        det.add_element(x)
        if det.detected_change():
            dets.append(i)
    results[name] = dets

for name, dets in results.items():
    print(
        f"{name:>20s}: first detection at {dets[0] if dets else 'N/A'} "
        f"(total {len(dets)} detections)"
    )

# %% [markdown]
# ## 4. Distance-based detectors
#
# Distance-based detectors measure how far the test window is from the
# reference and compare this distance to a fixed threshold (no p-value).
#
# | Detector | Key idea |
# |---|---|
# | **KLDivergence** | Information gain (asymmetric) |
# | **JensenShannon** | Symmetric, bounded divergence |
# | **PSI** | Popular in finance / credit scoring |
# | **Hellinger** | Bounded [0, 1], symmetric |
# | **Wasserstein** | Optimal-transport / earth mover's distance |
# | **EnergyDistance** | Compares distributions independently for each feature; no kernel choice required |
#
# Histogram-based detectors (KL, JS, PSI, Hellinger) use `num_bins` to
# discretise continuous features. Choose bins so that each bin has enough
# observations; a good rule of thumb is `num_bins ≈ sqrt(window_size)`.

# %%
from capymoa.drift.detectors import (
    PSI,
    EnergyDistance,
    Hellinger,
    JensenShannon,
    KLDivergence,
    Wasserstein,
)

n_bins = max(5, int(np.sqrt(WINDOW)))

dist_detectors = {
    "KL Divergence": KLDivergence(window_size=WINDOW, num_bins=n_bins, threshold=0.5),
    "Jensen-Shannon": JensenShannon(window_size=WINDOW, num_bins=n_bins, threshold=0.2),
    "PSI": PSI(window_size=WINDOW, num_bins=n_bins, threshold=1.0),
    "Hellinger": Hellinger(window_size=WINDOW, num_bins=n_bins, threshold=0.3),
    "Wasserstein": Wasserstein(window_size=WINDOW, threshold=0.5),
    "Energy Distance": EnergyDistance(window_size=WINDOW, threshold=0.5),
}

results_dist = {}
for name, det in dist_detectors.items():
    det.fit(X_ref)
    dets = []
    for i, x in enumerate(X_stream):
        det.add_element(x)
        if det.detected_change():
            dets.append(i)
    results_dist[name] = dets

for name, dets in results_dist.items():
    print(
        f"{name:>20s}: first detection at {dets[0] if dets else 'N/A'} "
        f"(total {len(dets)} detections)"
    )

# %% [markdown]
# ## 5. Multivariate detectors
#
# Some detectors operate on the full feature vector instead of testing each
# feature independently:
#
# | Detector | Key idea |
# |---|---|
# | **MMD** | Kernel-based two-sample test (permutation p-value) |
# | **D3** | Trains a classifier to tell reference from test |
#
# **BNDM** (Bayesian Pólya-tree test) is listed separately because it is
# applied per feature, like the statistical tests above, even though it
# does not produce a classical p-value. Unlike those tests, BNDM does not
# use Bonferroni correction: overall drift is any feature crossing its
# similarity threshold, uncorrected.

# %%
from capymoa.drift.detectors import D3, MMD

multi_detectors = {
    "MMD": MMD(window_size=WINDOW, n_permutations=50, sigma=1.0),
    "D3": D3(window_size=WINDOW, threshold=0.7, seed=42),
}

results_multi = {}
for name, det in multi_detectors.items():
    det.fit(X_ref)
    dets = []
    for i, x in enumerate(X_stream):
        det.add_element(x)
        if det.detected_change():
            dets.append(i)
    results_multi[name] = dets

for name, dets in results_multi.items():
    print(
        f"{name:>6s}: first detection at {dets[0] if dets else 'N/A'} "
        f"(total {len(dets)} detections)"
    )

# %% [markdown]
# ### Bayesian detector: BNDM
#
# BNDM uses a Pólya-tree two-sample test. Although it can handle
# multivariate data, it is applied **per feature**
# (`IS_UNIVARIATE = True`), like the statistical tests above. It does not
# produce p-values, so no Bonferroni correction is applied: overall drift
# is any feature crossing its similarity threshold.

# %%
from capymoa.drift.detectors import BNDM

bndm = BNDM(window_size=WINDOW, threshold=0.3, max_depth=3)
bndm.fit(X_ref)

bndm_dets = []
for i, x in enumerate(X_stream):
    bndm.add_element(x)
    if bndm.detected_change():
        bndm_dets.append(i)

results_bndm = {"BNDM": bndm_dets}
print(
    f"BNDM: first detection at {bndm_dets[0] if bndm_dets else 'N/A'} "
    f"(total {len(bndm_dets)} detections)"
)

# %% [markdown]
# ### Visualising detection points
#
# We can compare how quickly each family of detectors reacts to the drift.

# %%
all_results = {**results, **results_dist, **results_multi, **results_bndm}

fig, ax = plt.subplots(figsize=(12, 5))
names = list(all_results.keys())
for idx, (name, dets) in enumerate(all_results.items()):
    if dets:
        ax.scatter(dets, [idx] * len(dets), marker="|", s=100, linewidths=1.5)
    else:
        ax.scatter([], [])

ax.axvline(half, color="red", linestyle="--", label=f"True drift at {half}")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("Observation index")
ax.set_title("Detection points by detector")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Categorical features with Chi-Square
#
# The `ChiSquare` detector is designed for **categorical** (discrete) data.
# It builds a contingency table per feature and applies the chi-square test
# of independence.

# %%
from capymoa.drift.detectors import ChiSquare

categories = ["a", "b", "c"]
cat_ref = rng.choice(categories, size=(N_REF, 2), p=[0.5, 0.3, 0.2])
cat_stream = np.vstack(
    [
        rng.choice(categories, size=(half, 2), p=[0.5, 0.3, 0.2]),
        rng.choice(categories, size=(half, 2), p=[0.1, 0.2, 0.7]),
    ]
)

chi2 = ChiSquare(window_size=WINDOW)
chi2.fit(cat_ref)

chi2_dets = []
for i, x in enumerate(cat_stream):
    chi2.add_element(x)
    if chi2.detected_change():
        chi2_dets.append(i)

print(
    f"Chi-Square: first detection at {chi2_dets[0] if chi2_dets else 'N/A'} "
    f"(total {len(chi2_dets)} detections)"
)

# %% [markdown]
# ## 7. Batch comparison with `compare()`
#
# If you already have a test window, you can compare it against the
# reference in one call. This skips the streaming loop and returns a
# `DataDriftResult` directly.

# %%
from capymoa.drift.detectors import MMD, KolmogorovSmirnov

X_no_drift = rng.normal(0, 1, size=(WINDOW, N_FEATURES))
X_drifted = rng.normal(SHIFT, 1, size=(WINDOW, N_FEATURES))

for label, DetCls, kwargs in [
    ("KS", KolmogorovSmirnov, {"window_size": WINDOW}),
    ("MMD", MMD, {"window_size": WINDOW, "n_permutations": 50, "sigma": 1.0}),
]:
    det = DetCls(**kwargs)
    det.fit(X_ref)

    r_same = det.compare(X_no_drift)
    r_drift = det.compare(X_drifted)

    print(f"\n{label}:")
    print(f"  Same distribution → drift={r_same.is_drift}, stat={r_same.statistic:.4f}")
    print(
        f"  Shifted           → drift={r_drift.is_drift}, stat={r_drift.statistic:.4f}"
    )

# %% [markdown]
# ## 8. Auto-fit mode
#
# If you do not have a separate reference dataset, you can let the detector
# collect the first `N` observations as its reference by setting
# `auto_fit_samples`. No explicit `fit()` call is needed; `REQUIRES_FIT`
# stays `True` because a reference is still collected through `add_element`.

# %%
from capymoa.drift.detectors import KolmogorovSmirnov

ks_auto = KolmogorovSmirnov(window_size=WINDOW, auto_fit_samples=N_REF)

# Stream everything – the first N_REF samples are used as reference
all_data = np.vstack([X_ref, X_stream])
auto_dets = []
for i, x in enumerate(all_data):
    ks_auto.add_element(x)
    if ks_auto.detected_change():
        auto_dets.append(i)

# Drift is at N_REF + half
true_drift = N_REF + half
print(f"Auto-fit KS: first detection at {auto_dets[0] if auto_dets else 'N/A'}")
print(f"(True drift at observation {true_drift})")

# %% [markdown]
# ## 9. Feature names from pandas DataFrames
#
# When you pass a `DataFrame` to `fit()`, the detector extracts column names
# automatically. These appear in the `feature_names` attribute.

# %%
df_ref = pd.DataFrame(X_ref, columns=["temperature", "humidity"])

ks_df = KolmogorovSmirnov(window_size=WINDOW)
ks_df.fit(df_ref)

print(f"Feature names: {ks_df.feature_names}")

result = ks_df.compare(X_stream[half : half + WINDOW])
print(f"Per-feature drift: {result.feature_is_drift}")

# %% [markdown]
# ## Summary
#
# | Category | Detectors | Data type |
# |---|---|---|
# | Statistical tests | KolmogorovSmirnov, AndersonDarling, CramerVonMises | Numeric (univariate per feature) |
# | Distance-based | KLDivergence, JensenShannon, PSI, Hellinger, Wasserstein, EnergyDistance | Numeric (univariate per feature) |
# | Multivariate | MMD, D3 | Numeric (full feature vector) |
# | Bayesian | BNDM | Numeric (univariate per feature) |
# | Categorical | ChiSquare | Categorical |
#
# All detectors share the same API. They have `REQUIRES_FIT = True`:
# provide a reference with `fit(X_ref)`, or set `auto_fit_samples` so
# `add_element` collects it.
#
# 1. `fit(X_ref)` – set the reference distribution (not needed with `auto_fit_samples`).
# 2. `add_element(x)` – stream a single observation.
# 3. `detected_change()` – check if drift was detected.
# 4. `compare(X_test)` – batch comparison returning a `DataDriftResult` (needs `fit()`).
# 5. `reset()` – clear the sliding window (keeps reference).
#
