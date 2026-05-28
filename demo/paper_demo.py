# SPDX-License-Identifier: BSD-3-Clause
"""
OpenMOA Paper Demo — UOL Classifiers on Evolving Feature Streams
=================================================================
This script benchmarks the four core UOL (Utilitarian Online Learning)
classifiers provided by OpenMOA across all six feature-evolution patterns.

KEY PROPERTY DEMONSTRATED
--------------------------
All UOL classifiers are pure Python.  The entire script runs without
a JVM — no Java, no MOA jar, no jpype initialisation.

    pip install openmoa          # pulls numpy, scipy, scikit-learn
    python demo/paper_demo.py    # no JVM setup required

Classifiers compared
--------------------
  FESL   Feature-Evolving Sparse Learning
  FOBOS  Forward-Backward Splitting
  FTRL   Follow-The-Regularized-Leader
  ORF3V  Online Random Forest for Varied-Vocabulary streams

Feature-evolution patterns
--------------------------
  PIR  Pyramid       features increase then symmetrically decrease
  INC  Incremental   features increase monotonically
  DEC  Decremental   features decrease monotonically
  TDS  Trapezoidal   plateau of peak features between rise/fall
  CDS  Capricious    features appear/disappear randomly (with NaN)
  EDS  Evolvable     overlapping feature segments (with NaN)

Evaluation protocol
-------------------
  Prequential (test-then-train), windowed accuracy (mean ± std over
  N_REPEATS independent shuffled runs).

Output
------
  paper_demo_output/
    openmoa_uol_comparison.png   — 2×3 paper figure
    results_summary.csv          — final accuracy per (pattern, algorithm)
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections import deque

import numpy as np
import pandas as pd

# ── matplotlib: non-interactive backend for reproducible server/CI runs ───────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── path setup (run from repo root *or* from demo/ folder) ───────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Pure-Python UOL imports — NO Java dependency ──────────────────────────────
from openmoa.classifier import (
    FESLClassifier,
    FOBOSClassifier,
    FTRLClassifier,
    ORF3VClassifier,
)
from openmoa.stream.stream_wrapper import OpenFeatureStream, ShuffledStream
import openmoa.datasets

warnings.filterwarnings("ignore")

# =============================================================================
# Experiment configuration
# =============================================================================

OUTPUT_DIR  = os.path.join(_HERE, "paper_demo_output")
DATASET_CLS = openmoa.datasets.Spambase   # 4601 instances, 57 features, binary
DATASET_NAME = "Spambase"

N_REPEATS   = 5      # independent shuffled runs  -> mean +/- std
WINDOW_SIZE = 500    # sliding-window width for prequential accuracy
LOG_EVERY   = 50     # record accuracy every N instances

# Feature-evolution patterns and their OpenFeatureStream keyword arguments
PATTERNS: dict[str, dict] = {
    "PIR": dict(evolution_pattern="pyramid"),
    "INC": dict(evolution_pattern="incremental"),
    "DEC": dict(evolution_pattern="decremental"),
    "TDS": dict(evolution_pattern="tds", tds_mode="ordered"),
    "CDS": dict(evolution_pattern="cds", missing_ratio=0.4),
    "EDS": dict(evolution_pattern="eds", n_segments=3, overlap_ratio=0.3),
}

PATTERN_LABELS = {
    "PIR": "Pyramid (PIR)",
    "INC": "Incremental (INC)",
    "DEC": "Decremental (DEC)",
    "TDS": "Trapezoidal (TDS)",
    "CDS": "Capricious (CDS)",
    "EDS": "Evolvable (EDS)",
}

# One factory function per classifier (schema + random_seed → learner instance)
CLASSIFIERS: dict[str, object] = {
    "FESL":  lambda schema, seed: FESLClassifier(
        schema, alpha=0.1, lambda_=0.1, window_size=100, random_seed=seed
    ),
    "FOBOS": lambda schema, seed: FOBOSClassifier(
        schema, alpha=1.0, lambda_=0.001, random_seed=seed
    ),
    "FTRL":  lambda schema, seed: FTRLClassifier(
        schema, alpha=0.1, beta=1.0, l1=1.0, l2=1.0, random_seed=seed
    ),
    "ORF3V": lambda schema, seed: ORF3VClassifier(
        schema, n_stumps=10, alpha=0.1, random_seed=seed
    ),
}

# Colours used consistently across all subplots
CLF_COLORS = {
    "FESL":  "#1f77b4",
    "FOBOS": "#ff7f0e",
    "FTRL":  "#2ca02c",
    "ORF3V": "#d62728",
}

# =============================================================================
# Evaluation
# =============================================================================

def _stream_length(dataset_cls) -> int:
    """Return the number of instances in the dataset (or a safe default)."""
    ds = dataset_cls()
    if hasattr(ds, "get_num_instances"):
        n = ds.get_num_instances()
        if n and n > 0:
            return n
    return 5000


def prequential_run(stream, learner) -> tuple[np.ndarray, np.ndarray]:
    """
    Test-then-train loop on *stream* with *learner*.

    Returns
    -------
    steps : ndarray
        Instance indices at which accuracy was recorded.
    accs  : ndarray
        Corresponding windowed accuracy values.
    """
    window        = deque(maxlen=WINDOW_SIZE)
    window_correct = 0
    steps, accs    = [], []
    step           = 0

    while stream.has_more_instances():
        instance = stream.next_instance()
        if instance is None:
            break

        # ── test ──────────────────────────────────────────────────────────────
        pred    = learner.predict(instance)
        correct = int(pred == instance.y_index) if pred is not None else 0

        # Sliding-window accuracy bookkeeping
        if len(window) == WINDOW_SIZE:
            window_correct -= window[0]   # element about to be evicted
        window.append(correct)
        window_correct += correct

        # ── train ─────────────────────────────────────────────────────────────
        learner.train(instance)

        step += 1
        if step % LOG_EVERY == 0 or not stream.has_more_instances():
            steps.append(step)
            accs.append(window_correct / len(window))

    return np.array(steps), np.array(accs)


def run_experiment(
    dataset_cls,
    pattern_kwargs: dict,
    n_repeats: int = N_REPEATS,
    seed_base: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Repeat the prequential evaluation *n_repeats* times for every classifier
    under a single feature-evolution pattern.

    Returns
    -------
    dict mapping classifier name → (grid, mean_acc, std_acc)
    """
    n_total    = _stream_length(dataset_cls)
    base_schema = dataset_cls().get_schema()

    # Accumulate per-run accuracy traces for each classifier
    all_traces: dict[str, list] = {name: [] for name in CLASSIFIERS}

    for seed in range(seed_base, seed_base + n_repeats):
        for name, clf_factory in CLASSIFIERS.items():
            # Fresh shuffled stream for each (seed × classifier) combination
            base   = ShuffledStream(dataset_cls(), random_seed=seed)
            stream = OpenFeatureStream(
                base,
                d_min=2,
                total_instances=n_total,
                random_seed=seed,
                **pattern_kwargs,
            )
            learner = clf_factory(base_schema, seed)

            steps, accs = prequential_run(stream, learner)
            all_traces[name].append((steps, accs))

    # Interpolate all runs onto a shared grid and compute mean ± std
    results: dict[str, tuple] = {}
    for name, traces in all_traces.items():
        max_step = max(t[0][-1] for t in traces if len(t[0]) > 0)
        grid     = np.linspace(LOG_EVERY, max_step, 200)

        interp = np.array([np.interp(grid, s, a) for s, a in traces if len(s) > 0])
        results[name] = (grid, interp.mean(axis=0), interp.std(axis=0))

    return results


# =============================================================================
# Visualisation
# =============================================================================

def _add_accuracy_curves(ax, results, pattern_name):
    """Draw accuracy curves (mean ± std band) for one subplot."""
    for name, (grid, mean, std) in results.items():
        color = CLF_COLORS[name]
        ax.plot(grid, mean, label=name, color=color, linewidth=1.8)
        ax.fill_between(
            grid,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=color, alpha=0.15,
        )

    ax.set_ylim(0.35, 1.02)
    ax.set_title(PATTERN_LABELS.get(pattern_name, pattern_name), fontsize=11, pad=4)
    ax.set_xlabel("Instances processed", fontsize=8)
    ax.set_ylabel("Prequential accuracy", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)


def generate_paper_figure(
    all_results: dict[str, dict],
    output_path: str,
):
    """
    2 × 3 grid of subplots — one per feature-evolution pattern.
    A shared legend is placed at the bottom of the figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    axes_flat  = axes.flatten()

    for ax, (pname, results) in zip(axes_flat, all_results.items()):
        _add_accuracy_curves(ax, results, pname)

    # Shared legend (bottom centre)
    legend_handles = [
        mlines.Line2D([], [], color=CLF_COLORS[name], linewidth=2, label=name)
        for name in CLASSIFIERS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=4,
        fontsize=11, frameon=True,
        title="Algorithm", title_fontsize=11,
    )

    fig.suptitle(
        f"OpenMOA UOL Classifiers -- {DATASET_NAME} dataset, 6 feature-evolution patterns\n"
        f"Prequential accuracy  |  window = {WINDOW_SIZE}  |  "
        f"mean +/- std over {N_REPEATS} seeds  |  pure Python (no Java)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure  -> {output_path}")


def save_summary_csv(all_results: dict, output_path: str):
    """Save final-accuracy table (pattern × algorithm) as CSV."""
    rows = []
    for pname, results in all_results.items():
        for name, (grid, mean, std) in results.items():
            rows.append({
                "pattern":       pname,
                "algorithm":     name,
                "final_acc_mean": round(float(mean[-1]), 4),
                "final_acc_std":  round(float(std[-1]),  4),
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Summary -> {output_path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 64)
    print("  OpenMOA Paper Demo: UOL on Evolving Feature Streams")
    print("  Pure Python -- no Java / JVM required")
    print("=" * 64)
    print(f"  Dataset    : {DATASET_NAME}")
    print(f"  Patterns   : {', '.join(PATTERNS)}")
    print(f"  Classifiers: {', '.join(CLASSIFIERS)}")
    print(f"  Repeats    : {N_REPEATS}  |  Window: {WINDOW_SIZE}")
    print()

    all_results: dict[str, dict] = {}
    t_global = time.time()

    for pname, pkwargs in PATTERNS.items():
        t0 = time.time()
        print(f"  [{pname}] running ...", end=" ", flush=True)

        all_results[pname] = run_experiment(
            dataset_cls=DATASET_CLS,
            pattern_kwargs=pkwargs,
        )
        print(f"done  ({time.time() - t0:.1f}s)")

    print(f"\n  Total elapsed: {time.time() - t_global:.1f}s")
    print()

    # ── Accuracy summary ──────────────────────────────────────────────────────
    print(f"  {'Pattern':<6}  {'Algorithm':<8}  {'Acc (mean+/-std)':>17}")
    print("  " + "-" * 37)
    for pname, results in all_results.items():
        for name, (_, mean, std) in results.items():
            print(f"  {pname:<6}  {name:<8}  {mean[-1]:.4f} +/- {std[-1]:.4f}")
    print()

    # ── Outputs ───────────────────────────────────────────────────────────────
    generate_paper_figure(
        all_results,
        os.path.join(OUTPUT_DIR, "openmoa_uol_comparison.png"),
    )
    save_summary_csv(
        all_results,
        os.path.join(OUTPUT_DIR, "results_summary.csv"),
    )

    print("\n  Demo completed successfully.")


if __name__ == "__main__":
    main()
