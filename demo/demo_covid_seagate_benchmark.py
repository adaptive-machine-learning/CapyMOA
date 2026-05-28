# SPDX-License-Identifier: BSD-3-Clause
"""
demo_covid_seagate_benchmark.py
--------------------------------
Benchmark all 10 UOL classifiers on the 8 COVID policy datasets
(C1–C8) plus Seagate Binary and Seagate Multi.

PURPOSE
-------
Evaluate which COVID/Seagate datasets yield meaningful learning signals
for UOL algorithms under dynamic feature spaces (TDS / CDS / EDS).
Datasets where all algorithms perform near random baseline (~1/n_classes)
can be considered uninformative and excluded from future benchmarks.

EVALUATION PROTOCOL
-------------------
- Prequential (test-then-train), no separate test set
- 5 repeated runs with different random seeds (shuffled each time)
- Three feature-space paradigms: TDS, CDS, EDS
- Final metric: mean ± std of windowed prequential accuracy

OUTPUT
------
  demo_covid_seagate_output/
    summary.csv              — final accuracy table (dataset × algorithm)
    heatmap_<paradigm>.png   — accuracy heatmap per paradigm
    trace_<dataset>_<algo>_<paradigm>.png  — per-run accuracy curves

USAGE
-----
  # From repo root:
  python demo/demo_covid_seagate_benchmark.py

  # Run only a subset for quick testing:
  python demo/demo_covid_seagate_benchmark.py --fast
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections import deque

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# ── Imports ───────────────────────────────────────────────────────────────────

try:
    import openmoa.datasets as datasets
    from openmoa.stream.stream_wrapper import (
        OpenFeatureStream, TrapezoidalStream,
        CapriciousStream, EvolvableStream, ShuffledStream,
    )

    from openmoa.classifier._fesl_classifier  import FESLClassifier
    from openmoa.classifier._oasf_classifier  import OASFClassifier
    from openmoa.classifier._rsol_classifier  import RSOLClassifier
    from openmoa.classifier._fobos_classifier import FOBOSClassifier
    from openmoa.classifier._ftrl_classifier  import FTRLClassifier
    from openmoa.classifier._orf3v_classifier import ORF3VClassifier
    from openmoa.classifier._ovfm_classifier  import OVFMClassifier
    from openmoa.classifier._oslmf_classifier import OSLMFClassifier
    from openmoa.classifier._old3s_classifier import OLD3SClassifier
    from openmoa.classifier._owss_classifier  import OWSSClassifier
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_covid_seagate_output")

# All COVID/Seagate datasets with their class counts for baseline reference
DATASET_REGISTRY = [
    ("Seagate_Binary", "SeagateBinary", 2),
    ("Seagate_Multi",  "SeagateMulti",  11),
    ("C1_School_Closing",           "C1SchoolClosing",           4),
    ("C2_Workplace_Closing",        "C2WorkplaceClosing",        4),
    ("C3_Cancel_Public_Events",     "C3CancelPublicEvents",      2),
    ("C4_Restrictions_Gatherings",  "C4RestrictionsGatherings",  5),
    ("C5_Close_Public_Transport",   "C5ClosePublicTransport",    2),
    ("C6_Stay_Home_Requirements",   "C6StayHomeRequirements",    3),
    ("C7_Internal_Movement",        "C7InternalMovement",        3),
    ("C8_International_Travel",     "C8InternationalTravel",     4),
]

# Three feature-evolution paradigms (matching the OpenMOA benchmark)
PARADIGMS = {
    "TDS": dict(evolution_pattern="tds", tds_mode="ordered"),
    "CDS": dict(evolution_pattern="cds", missing_ratio=0.4),
    "EDS": dict(evolution_pattern="eds", n_segments=3, overlap_ratio=0.2),
}

# COVID datasets are time-ordered — do NOT shuffle
# Seagate datasets are static — shuffle to remove label-sorting bias
NO_SHUFFLE = {
    "C1_School_Closing", "C2_Workplace_Closing", "C3_Cancel_Public_Events",
    "C4_Restrictions_Gatherings", "C5_Close_Public_Transport",
    "C6_Stay_Home_Requirements", "C7_Internal_Movement", "C8_International_Travel",
}

# ── Auto-tuning ───────────────────────────────────────────────────────────────

# Search grids: one key learning-rate-like parameter per algorithm.
# Other hyperparameters are fixed at robust defaults.
_SEARCH_GRIDS: dict[str, dict] = {
    "FESL":  {"alpha":        [0.01, 0.1, 1.0, 5.0, 10.0]},
    "OASF":  {"lambda_param": [0.001, 0.01, 0.1, 1.0]},
    "RSOL":  {"lambda_param": [0.001, 0.01, 0.1, 1.0]},
    "FOBOS": {"alpha":        [0.01, 0.1, 1.0, 5.0, 10.0]},
    "FTRL":  {"alpha":        [0.01, 0.1, 0.5, 1.0, 5.0]},
    "ORF3V": {"alpha":        [0.01, 0.1, 0.5, 1.0]},
    "OVFM":  {"learning_rate": [0.001, 0.01, 0.05, 0.1]},
    "OSLMF": {"learning_rate": [0.001, 0.01, 0.05, 0.1]},
    "OLD3S": {"learning_rate": [0.0001, 0.001, 0.01]},
    "OWSS":  {"learning_rate": [0.001, 0.01, 0.05, 0.1]},
}

# Maximum burn-in instances used for tuning.
# Adaptive: actual burn-in = min(BURN_IN_MAX, BURN_IN_RATIO * n_total)
BURN_IN_MAX   = 300
BURN_IN_RATIO = 0.20   # use at most 20% of the dataset for tuning


def _make_classifier_with_param(name: str, schema, d: int, seed: int,
                                 param_name: str, param_value: float):
    """Instantiate a classifier with one specific hyperparameter overridden."""
    kw = dict(schema=schema, random_seed=seed)
    if name == "FESL":
        return FESLClassifier(**kw, lambda_=0.1, window_size=100,
                              **{param_name: param_value})
    if name == "OASF":
        return OASFClassifier(**kw, mu=1.0, L=100,
                              **{param_name: param_value})
    if name == "RSOL":
        return RSOLClassifier(**kw, mu=1.0, L=100,
                              **{param_name: param_value})
    if name == "FOBOS":
        return FOBOSClassifier(**kw, lambda_=0.001, regularization="l1",
                               **{param_name: param_value})
    if name == "FTRL":
        return FTRLClassifier(**kw, beta=1.0, l1=0.1, l2=0.0,
                              **{param_name: param_value})
    if name == "ORF3V":
        return ORF3VClassifier(**kw, n_stumps=10, d_max=d,
                               **{param_name: param_value})
    if name == "OVFM":
        return OVFMClassifier(**kw, window_size=200, batch_size=50, l2_lambda=0.01,
                              **{param_name: param_value})
    if name == "OSLMF":
        return OSLMFClassifier(**kw, window_size=200, buffer_size=200, batch_size=50,
                               **{param_name: param_value})
    if name == "OLD3S":
        return OLD3SClassifier(**kw, latent_dim=20, hidden_dim=128, num_hbp_layers=3,
                               **{param_name: param_value})
    if name == "OWSS":
        return OWSSClassifier(**kw, window_size=100, hidden_dim=32, rec_weight=0.1,
                              **{param_name: param_value})
    raise ValueError(f"Unknown classifier: {name}")


def auto_tune(algo_name: str, stream_factory, schema, d: int,
              seed: int, n_total: int) -> dict:
    """Search the best value for the primary hyperparameter via burn-in.

    Creates the stream **once** and calls restart() between candidates to
    avoid Windows file-locking errors from repeatedly re-opening the same
    gzipped dataset file.

    Parameters
    ----------
    algo_name     : classifier name string
    stream_factory: callable() → fresh wrapped stream
    schema        : stream schema
    d             : number of features
    seed          : random seed (used for classifier init)
    n_total       : total stream length (for adaptive burn-in sizing)

    Returns
    -------
    dict  e.g. {"alpha": 0.1}  — best hyperparameter found
    """
    grid = _SEARCH_GRIDS[algo_name]
    param_name   = next(iter(grid))
    param_values = grid[param_name]

    burn_in = max(10, min(BURN_IN_MAX, int(n_total * BURN_IN_RATIO)))

    best_val = param_values[0]
    best_acc = -1.0

    # Create the stream once; restart() between candidates avoids repeated
    # gzip extraction that triggers Windows [WinError 32] file-lock conflicts.
    stream = stream_factory()

    for val in param_values:
        try:
            stream.restart()
        except Exception:
            # Fallback: recreate if restart() is unsupported
            stream = stream_factory()

        learner = _make_classifier_with_param(
            algo_name, schema, d, seed, param_name, val
        )
        correct = total = 0
        while stream.has_more_instances() and total < burn_in:
            inst = stream.next_instance()
            if learner.predict(inst) == inst.y_index:
                correct += 1
            learner.train(inst)
            total += 1

        acc = correct / total if total > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            best_val = val

    return {param_name: best_val}


# ── Classifier factory ────────────────────────────────────────────────────────

def make_classifier(name: str, schema, d: int, seed: int, tuned: dict | None = None):
    """Instantiate a UOL classifier, optionally with tuned hyperparameters."""
    override = tuned or {}
    kw = dict(schema=schema, random_seed=seed)
    if name == "FESL":
        params = {"alpha": 0.1, **override}
        return FESLClassifier(**kw, lambda_=0.1, window_size=100, **params)
    if name == "OASF":
        params = {"lambda_param": 0.01, **override}
        return OASFClassifier(**kw, mu=1.0, L=100, **params)
    if name == "RSOL":
        params = {"lambda_param": 0.01, **override}
        return RSOLClassifier(**kw, mu=1.0, L=100, **params)
    if name == "FOBOS":
        params = {"alpha": 1.0, **override}
        return FOBOSClassifier(**kw, lambda_=0.001, regularization="l1", **params)
    if name == "FTRL":
        params = {"alpha": 0.1, **override}
        return FTRLClassifier(**kw, beta=1.0, l1=0.1, l2=0.0, **params)
    if name == "ORF3V":
        params = {"alpha": 0.1, **override}   # override wins over default
        return ORF3VClassifier(**kw, n_stumps=10, d_max=d, **params)
    if name == "OVFM":
        params = {"learning_rate": 0.01, **override}
        return OVFMClassifier(**kw, window_size=200, batch_size=50, l2_lambda=0.01, **params)
    if name == "OSLMF":
        params = {"learning_rate": 0.01, **override}
        return OSLMFClassifier(**kw, window_size=200, buffer_size=200, batch_size=50, **params)
    if name == "OLD3S":
        params = {"learning_rate": 0.001, **override}
        return OLD3SClassifier(**kw, latent_dim=20, hidden_dim=128, num_hbp_layers=3, **params)
    if name == "OWSS":
        params = {"learning_rate": 0.01, **override}
        return OWSSClassifier(**kw, window_size=100, hidden_dim=32, rec_weight=0.1, **params)
    raise ValueError(f"Unknown classifier: {name}")


CLASSIFIER_NAMES = [
    "FESL", "OASF", "RSOL", "FOBOS", "FTRL", "ORF3V",
]

# FESL requires O(d²) mapping matrix — skip on very high-d datasets
FESL_MAX_D = 5_000

# Algorithms that only support binary classification (n_classes == 2)
BINARY_ONLY_ALGOS = {"FESL", "OASF", "RSOL"}

# Stream type required by each algorithm:
#   "open"        → OpenFeatureStream  (physically variable-length x, feature_indices attached)
#   "trapezoidal" → TrapezoidalStream  (fixed-length x with NaN for inactive features)
#
# FESL/OASF/RSOL/FOBOS/FTRL use SparseInputMixin which handles both formats,
# but OpenFeatureStream is preferred (more faithful to UOL setting).
# OVFM/OSLMF expect a fixed-length NaN-padded x — must use TrapezoidalStream.
# ORF3V/OLD3S rely on feature_indices — must use OpenFeatureStream.
# OWSS reads instance.x directly without NaN handling — must use OpenFeatureStream.
ALGO_STREAM_TYPE: dict[str, str] = {
    "FESL":  "open",
    "OASF":  "open",
    "RSOL":  "open",
    "FOBOS": "open",
    "FTRL":  "open",
    "ORF3V": "open",
}

# ── Stream helpers ────────────────────────────────────────────────────────────

def get_stream_length(stream, default: int = 789) -> int:
    for attr in ("get_num_instances", "n_instances", "total_instances", "_length", "__len__"):
        if hasattr(stream, attr):
            val = getattr(stream, attr)
            return val() if callable(val) else int(val)
    return default


def make_stream(dataset_cls, d_name: str, paradigm_kw: dict,
                n_total: int, seed: int, stream_type: str = "open"):
    """Build base stream (optionally shuffled) → OpenFeatureStream or TrapezoidalStream.

    Parameters
    ----------
    stream_type : "open" | "trapezoidal"
        "open"        → OpenFeatureStream: physically variable-length x with feature_indices.
        "trapezoidal" → TrapezoidalStream: fixed-length x with NaN for inactive features.
    """
    base = dataset_cls()
    if d_name not in NO_SHUFFLE:
        base = ShuffledStream(base, random_seed=seed)

    if stream_type == "trapezoidal":
        ep = paradigm_kw.get("evolution_pattern", "tds")
        if ep == "tds":
            return TrapezoidalStream(
                base_stream=base,
                total_instances=n_total,
                random_seed=seed,
                evolution_mode=paradigm_kw.get("tds_mode", "ordered"),
            )
        elif ep == "cds":
            return CapriciousStream(
                base_stream=base,
                total_instances=n_total,
                random_seed=seed,
                missing_ratio=paradigm_kw.get("missing_ratio", 0.4),
            )
        else:  # eds
            return EvolvableStream(
                base_stream=base,
                total_instances=n_total,
                random_seed=seed,
                n_segments=paradigm_kw.get("n_segments", 3),
                overlap_ratio=paradigm_kw.get("overlap_ratio", 0.2),
            )

    # Default: OpenFeatureStream
    return OpenFeatureStream(
        base_stream=base,
        total_instances=n_total,
        random_seed=seed,
        **paradigm_kw,
    )

# ── Prequential evaluation ────────────────────────────────────────────────────

def prequential_eval(stream, learner, n_instances: int,
                     window_size: int) -> tuple[list, list, list]:
    window        = deque(maxlen=window_size)
    total_correct = 0
    steps, preq_accs, cum_accs = [], [], []
    log_every = max(1, n_instances // 50)

    for t in range(1, n_instances + 1):
        if not stream.has_more_instances():
            break
        inst = stream.next_instance()

        pred          = learner.predict(inst)
        correct       = int(pred == inst.y_index)
        total_correct += correct
        window.append(correct)
        learner.train(inst)

        if t % log_every == 0 or t == n_instances:
            steps.append(t)
            preq_accs.append(sum(window) / len(window))
            cum_accs.append(total_correct / t)

    return steps, preq_accs, cum_accs

# ── Single run ────────────────────────────────────────────────────────────────

def run_one(dataset_cls, d_name: str, algo_name: str,
            paradigm_name: str, paradigm_kw: dict,
            seed: int, window_size: int,
            do_tune: bool = True, n_classes: int = 2) -> dict | None:
    """Run one (dataset, algorithm, paradigm, seed) combination.

    Returns a dict with final prequential/cumulative accuracy,
    or None if the combination is skipped or crashes.
    """
    try:
        # Probe stream length
        probe = dataset_cls()
        n_total = get_stream_length(probe)
        schema  = probe.get_schema()
        d       = schema.get_num_attributes()
        del probe

        # Skip binary-only algorithms on multi-class datasets
        if algo_name in BINARY_ONLY_ALGOS and n_classes > 2:
            return None

        # Skip FESL on high-d datasets (OOM risk)
        if algo_name == "FESL" and d > FESL_MAX_D:
            return None

        stream_type = ALGO_STREAM_TYPE[algo_name]

        # Auto-tune primary hyperparameter via burn-in search
        tuned = None
        if do_tune:
            stream_factory = lambda: make_stream(
                dataset_cls, d_name, paradigm_kw, n_total, seed,
                stream_type=stream_type,
            )
            tuned = auto_tune(algo_name, stream_factory, schema, d, seed, n_total)

        stream  = make_stream(dataset_cls, d_name, paradigm_kw, n_total, seed,
                              stream_type=stream_type)
        learner = make_classifier(algo_name, schema, d, seed, tuned=tuned)

        steps, preq, cum = prequential_eval(stream, learner, n_total, window_size)

        if not steps:
            return None

        return {
            "dataset":    d_name,
            "algo":       algo_name,
            "paradigm":   paradigm_name,
            "seed":       seed,
            "n":          n_total,
            "d":          d,
            "tuned":      tuned,
            "final_preq": preq[-1],
            "final_cum":  cum[-1],
            "steps":      steps,
            "preq":       preq,
            "cum":        cum,
        }
    except Exception as exc:
        warnings.warn(f"[{d_name}|{algo_name}|{paradigm_name}|seed={seed}] "
                      f"failed: {exc}", RuntimeWarning)
        return None

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trace(runs: list[dict], d_name: str, algo_name: str,
               paradigm_name: str, n_classes: int, output_dir: str):
    """Plot mean ± std prequential accuracy across seeds."""
    if not runs:
        return

    # Align on common step grid
    min_len = min(len(r["steps"]) for r in runs)
    steps   = runs[0]["steps"][:min_len]
    preq_mat = np.array([r["preq"][:min_len] for r in runs])

    mean = preq_mat.mean(axis=0)
    std  = preq_mat.std(axis=0)
    baseline = 1.0 / n_classes

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, mean, lw=2, color="steelblue", label="Prequential acc (mean)")
    ax.fill_between(steps, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                    alpha=0.2, color="steelblue", label="±1 std")
    ax.axhline(baseline, ls="--", color="tomato", lw=1.2,
               label=f"Random baseline ({baseline:.2f})")
    ax.set_title(f"{algo_name} | {d_name} | {paradigm_name}", fontweight="bold")
    ax.set_xlabel("Instances"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4)

    fname = os.path.join(output_dir,
                         f"trace_{d_name}_{algo_name}_{paradigm_name}.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_heatmap(summary_df: pd.DataFrame, paradigm_name: str, output_dir: str):
    """Accuracy heatmap: rows = datasets, columns = algorithms."""
    pivot = summary_df[summary_df["paradigm"] == paradigm_name].pivot_table(
        index="dataset", columns="algo", values="final_preq_mean"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.1),
                                    max(4, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Prequential Accuracy"})
    ax.set_title(f"Accuracy Heatmap — {paradigm_name} paradigm",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Algorithm"); ax.set_ylabel("Dataset")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    fname = os.path.join(output_dir, f"heatmap_{paradigm_name}.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_baseline_comparison(summary_df: pd.DataFrame,
                              registry: list, output_dir: str):
    """Bar chart: mean accuracy vs random baseline, averaged over paradigms."""
    baseline_map = {name: 1.0 / n_cls for name, _, n_cls in registry}
    agg = (summary_df.groupby("dataset")["final_preq_mean"]
           .mean().reset_index()
           .rename(columns={"final_preq_mean": "acc_mean"}))
    agg["baseline"] = agg["dataset"].map(baseline_map)
    agg["gain"]     = agg["acc_mean"] - agg["baseline"]
    agg = agg.sort_values("gain", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(agg))
    ax.bar(x - 0.2, agg["acc_mean"],  width=0.4, color="steelblue",
           label="Mean accuracy (all algos & paradigms)")
    ax.bar(x + 0.2, agg["baseline"], width=0.4, color="tomato",
           alpha=0.7, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["dataset"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean accuracy vs random baseline across all UOL algorithms & paradigms",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)

    fname = os.path.join(output_dir, "baseline_comparison.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COVID/Seagate UOL benchmark")
    parser.add_argument("--fast", action="store_true",
                        help="Quick mode: 2 seeds, 1 paradigm (EDS only), "
                             "skip PyTorch classifiers")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of repeated runs (default: 5)")
    parser.add_argument("--paradigms", nargs="+", default=["TDS", "CDS", "EDS"],
                        choices=["TDS", "CDS", "EDS"],
                        help="Paradigms to evaluate (default: all three)")
    parser.add_argument("--algos", nargs="+", default=CLASSIFIER_NAMES,
                        choices=CLASSIFIER_NAMES,
                        help="Algorithms to evaluate (default: all 10)")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--no-tune", action="store_true",
                        help="Disable automatic hyperparameter tuning (use defaults)")
    args = parser.parse_args()

    do_tune = not args.no_tune

    if args.fast:
        n_seeds        = 2
        paradigm_names = ["EDS"]
        algo_names     = ["FESL", "FOBOS", "FTRL", "ORF3V"]
        window_size    = 100
    else:
        n_seeds        = args.seeds
        paradigm_names = args.paradigms
        algo_names     = args.algos
        window_size    = 200

    os.makedirs(args.output, exist_ok=True)

    # ── Discover available datasets ───────────────────────────────────────────
    registry = []
    for d_name, cls_name, n_classes in DATASET_REGISTRY:
        cls = getattr(datasets, cls_name, None)
        if cls is None:
            print(f"  ⚠  {cls_name} not found in openmoa.datasets — skipping")
            continue
        registry.append((d_name, cls, n_classes))

    print(f"\nOpenMOA COVID/Seagate Benchmark")
    print(f"  Datasets  : {[r[0] for r in registry]}")
    print(f"  Algorithms: {algo_names}")
    print(f"  Paradigms : {paradigm_names}")
    print(f"  Seeds     : {n_seeds}")
    print(f"  Auto-tune : {'yes' if do_tune else 'no'}")
    print(f"  Output    : {args.output}")
    print("=" * 70)

    # ── Main loop ─────────────────────────────────────────────────────────────
    all_rows   = []   # aggregated results (one row per dataset×algo×paradigm×seed)
    trace_bank = {}   # {(d_name, algo, paradigm): [run_dict, ...]}

    total = len(registry) * len(algo_names) * len(paradigm_names) * n_seeds
    done  = 0
    t0    = time.time()

    for d_name, dataset_cls, n_classes in registry:
        for paradigm_name in paradigm_names:
            paradigm_kw = PARADIGMS[paradigm_name]
            for algo_name in algo_names:
                key  = (d_name, algo_name, paradigm_name)
                runs = []
                print(f"\n  [{d_name}] {algo_name} / {paradigm_name} — seeds: ",
                      end="", flush=True)

                for seed in range(1, n_seeds + 1):
                    result = run_one(
                        dataset_cls, d_name, algo_name,
                        paradigm_name, paradigm_kw,
                        seed, window_size,
                        do_tune=do_tune,
                        n_classes=n_classes,
                    )
                    done += 1
                    if result is None:
                        print("x", end="", flush=True)
                        continue
                    runs.append(result)
                    tuned_str = ""
                    if result.get("tuned"):
                        k, v = next(iter(result["tuned"].items()))
                        tuned_str = f"[{k}={v}]"
                    all_rows.append({
                        "dataset":    d_name,
                        "algo":       algo_name,
                        "paradigm":   paradigm_name,
                        "seed":       seed,
                        "n":          result["n"],
                        "d":          result["d"],
                        "n_classes":  n_classes,
                        "final_preq": result["final_preq"],
                        "final_cum":  result["final_cum"],
                        "baseline":   1.0 / n_classes,
                        "tuned_param": str(result.get("tuned", "")),
                    })
                    print(f"{seed}{tuned_str}", end="", flush=True)

                trace_bank[key] = runs
                if runs:
                    mean_acc = np.mean([r["final_preq"] for r in runs])
                    baseline = 1.0 / n_classes
                    gain_str = f"{mean_acc - baseline:+.3f}"
                    print(f"  → preq={mean_acc:.3f} (baseline={baseline:.2f}, "
                          f"gain={gain_str})")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All runs complete in {elapsed:.1f}s")

    if not all_rows:
        print("No results to save.")
        return

    # ── Aggregate & save ──────────────────────────────────────────────────────
    raw_df = pd.DataFrame(all_rows)

    summary_df = (
        raw_df.groupby(["dataset", "algo", "paradigm", "n_classes", "baseline"])
        .agg(
            final_preq_mean=("final_preq", "mean"),
            final_preq_std=("final_preq",  "std"),
            final_cum_mean=("final_cum",   "mean"),
            n_runs=("seed", "count"),
        )
        .reset_index()
    )
    summary_df["gain"] = summary_df["final_preq_mean"] - summary_df["baseline"]

    summary_path = os.path.join(args.output, "summary.csv")
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"  Saved: {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")

    # 1. Accuracy heatmaps (one per paradigm)
    for paradigm_name in paradigm_names:
        plot_heatmap(summary_df, paradigm_name, args.output)

    # 2. Baseline comparison bar chart
    plot_baseline_comparison(summary_df, registry, args.output)

    # 3. Trace plots for every combination
    for (d_name, algo_name, paradigm_name), runs in trace_bank.items():
        n_classes = next(nc for r_name, _, nc in registry if r_name == d_name)
        plot_trace(runs, d_name, algo_name, paradigm_name, n_classes, args.output)

    # ── Dataset quality report ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATASET QUALITY REPORT")
    print("(mean accuracy vs random baseline, averaged over all algos & paradigms)")
    print("-" * 70)

    qual = (
        summary_df.groupby(["dataset", "baseline"])
        .agg(acc_mean=("final_preq_mean", "mean"),
             acc_std=("final_preq_mean",  "std"))
        .reset_index()
    )
    qual["gain"] = qual["acc_mean"] - qual["baseline"]
    qual = qual.sort_values("gain", ascending=False)

    RECOMMEND_THRESHOLD = 0.05   # must beat baseline by at least 5 pp

    keep, drop = [], []
    for _, row in qual.iterrows():
        verdict = "KEEP ✓" if row["gain"] >= RECOMMEND_THRESHOLD else "WEAK  ?"
        tag = keep if row["gain"] >= RECOMMEND_THRESHOLD else drop
        tag.append(row["dataset"])
        print(f"  {row['dataset']:<35} acc={row['acc_mean']:.3f} "
              f"baseline={row['baseline']:.3f}  gain={row['gain']:+.3f}  {verdict}")

    print("-" * 70)
    print(f"  Recommended to KEEP ({len(keep)}): {keep}")
    print(f"  Potentially weak  ({len(drop)}): {drop}")
    print("  (Threshold: gain ≥ +0.05 over random baseline)")
    print("=" * 70)

    qual_path = os.path.join(args.output, "dataset_quality.csv")
    qual.to_csv(qual_path, index=False, float_format="%.4f")
    print(f"  Saved: {qual_path}")


if __name__ == "__main__":
    main()
