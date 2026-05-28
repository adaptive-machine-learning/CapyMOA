# SPDX-License-Identifier: BSD-3-Clause
"""
Micro-benchmarks comparing old vs new implementation for each optimisation.
Run directly: python tests/benchmark_optimizations.py
"""
import time
import numpy as np
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def timeit(fn, repeats=200):
    fn()                          # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1000   # ms per call


def header(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def row(label, old_ms, new_ms):
    ratio = old_ms / new_ms if new_ms > 0 else float('inf')
    print(f"  {label:<30}  old={old_ms:7.3f}ms  new={new_ms:7.3f}ms  "
          f"speedup={ratio:.1f}x")


# ─────────────────────────────────────────────────────────────────────────────
# P1 — DensityPeaks loop vs vectorised
# ─────────────────────────────────────────────────────────────────────────────

def _density_peaks_loop(dist_matrix, p_arr=0.02):
    n = len(dist_matrix)
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]
    d_cut = max(np.sort(upper_tri)[int(len(upper_tri) * p_arr)], 1e-6)
    rho = np.sum(np.exp(-(dist_matrix / d_cut) ** 2), axis=1) - 1
    delta = np.zeros(n)
    nearest_higher = np.full(n, -1, dtype=int)
    sorted_indices = np.argsort(-rho)
    for i, idx in enumerate(sorted_indices):
        if i == 0:
            delta[idx] = np.max(dist_matrix[idx])
        else:
            higher = sorted_indices[:i]
            dists = dist_matrix[idx, higher]
            j = np.argmin(dists)
            delta[idx] = dists[j]
            nearest_higher[idx] = higher[j]
    return rho, delta, nearest_higher


def _density_peaks_vec(dist_matrix, p_arr=0.02):
    n = len(dist_matrix)
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]
    d_cut = max(np.sort(upper_tri)[int(len(upper_tri) * p_arr)], 1e-6)
    rho = np.sum(np.exp(-(dist_matrix / d_cut) ** 2), axis=1) - 1
    sorted_indices = np.argsort(-rho)
    rank = np.empty(n, dtype=np.intp)
    rank[sorted_indices] = np.arange(n)
    rank_mask = rank[np.newaxis, :] < rank[:, np.newaxis]
    dist_masked = np.where(rank_mask, dist_matrix, np.inf)
    delta = dist_masked.min(axis=1)
    nearest_higher = np.argmin(dist_masked, axis=1).astype(np.intp)
    top = sorted_indices[0]
    delta[top] = dist_matrix[top].max()
    nearest_higher[top] = -1
    return rho, delta, nearest_higher


def bench_p1():
    header("P1 — DensityPeaks: Python loop  vs  NumPy vectorised")
    rng = np.random.RandomState(0)
    for n in [50, 100, 200]:
        X = rng.randn(n, 8)
        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=-1))
        old = timeit(lambda: _density_peaks_loop(D))
        new = timeit(lambda: _density_peaks_vec(D))
        row(f"n={n} points", old, new)


# ─────────────────────────────────────────────────────────────────────────────
# P2 — statsmodels ECDF  vs  np.searchsorted
# ─────────────────────────────────────────────────────────────────────────────

def bench_p2():
    header("P2 — ECDF: statsmodels  vs  np.searchsorted")
    try:
        from statsmodels.distributions.empirical_distribution import ECDF
        HAS_STATSMODELS = True
    except ImportError:
        HAS_STATSMODELS = False
        print("  statsmodels not available — showing searchsorted baseline only")

    rng = np.random.RandomState(1)
    window = rng.randn(200)
    x_obs_small = rng.randn(10)
    x_obs_batch = rng.randn(100)

    for label, x_obs in [("single-point (10 obs)", x_obs_small),
                          ("batch    (100 obs)", x_obs_batch)]:
        sorted_w = np.sort(window)
        n_w = len(sorted_w)

        if HAS_STATSMODELS:
            def old_fn():
                ecdf = ECDF(window)
                H = n_w / (n_w + 1)
                return np.clip(H * ecdf(x_obs), 1e-10, 1 - 1e-10)
            old = timeit(old_fn, repeats=500)
        else:
            old = float('nan')

        def new_fn():
            return np.clip(
                np.searchsorted(sorted_w, x_obs, side='right') / (n_w + 1),
                1e-10, 1 - 1e-10
            )
        new = timeit(new_fn, repeats=500)

        if HAS_STATSMODELS:
            row(label, old, new)
        else:
            print(f"  {label:<30}  new={new:7.3f}ms  (old=N/A)")


# ─────────────────────────────────────────────────────────────────────────────
# P3 — np.array(instance.x)  vs  np.asarray(instance.x)
# ─────────────────────────────────────────────────────────────────────────────

def bench_p3():
    header("P3 — np.array  vs  np.asarray  (array already float64)")
    rng = np.random.RandomState(2)
    arr = rng.rand(200).astype(np.float64)   # already float64 ndarray

    old = timeit(lambda: np.array(arr, dtype=float), repeats=10000)
    new = timeit(lambda: np.asarray(arr, dtype=float), repeats=10000)
    row("d=200 float64", old, new)

    arr32 = rng.rand(200).astype(np.float32)
    old32 = timeit(lambda: np.array(arr32, dtype=float), repeats=10000)
    new32 = timeit(lambda: np.asarray(arr32, dtype=float), repeats=10000)
    row("d=200 float32→float64 (copy needed)", old32, new32)


# ─────────────────────────────────────────────────────────────────────────────
# P4 — OLD3S HBP: Python loop  vs  vectorised
# ─────────────────────────────────────────────────────────────────────────────

def bench_p4():
    header("P4 — OLD3S HBP weights: Python loop  vs  vectorised")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("  torch not available — skipping P4")
        return

    device = torch.device('cpu')
    num_layers = 3

    def make_state():
        w = torch.ones(num_layers, device=device) / num_layers
        losses = [torch.tensor(0.5 + 0.1 * i, requires_grad=False) for i in range(num_layers)]
        return w, losses

    beta = 0.99
    decay = torch.tensor(beta, device=device)

    def old_fn():
        w, losses = make_state()
        for i, l in enumerate(losses):
            w[i] *= torch.pow(decay, l)
        w /= w.sum()
        return w

    def new_fn():
        w, losses = make_state()
        losses_t = torch.stack([l.detach() for l in losses])
        w.mul_(decay.pow(losses_t))
        w.div_(w.sum())
        return w

    old = timeit(old_fn, repeats=5000)
    new = timeit(new_fn, repeats=5000)
    row(f"num_layers={num_layers}", old, new)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: full train() loop timing
# ─────────────────────────────────────────────────────────────────────────────

def bench_e2e():
    header("End-to-end — 200 train() calls on real classifier")
    from openmoa.stream import NumpyStream
    from openmoa.classifier import OSLMFClassifier, OVFMClassifier

    rng = np.random.RandomState(0)
    d, n = 8, 600
    X = rng.rand(n, d).astype(np.float32)
    y = rng.randint(0, 2, size=n)

    def make_stream():
        return NumpyStream(X, y, dataset_name="bench",
                           feature_names=[f"f{i}" for i in range(d)])

    for Clf, kwargs, label in [
        (OSLMFClassifier, dict(batch_size=50), "OSLMF (batch=50)"),
        (OVFMClassifier,  {},                  "OVFM"),
    ]:
        stream = make_stream()
        clf = Clf(schema=stream.get_schema(), random_seed=1, **kwargs)
        instances = []
        while stream.has_more_instances():
            instances.append(stream.next_instance())

        t0 = time.perf_counter()
        for inst in instances:
            clf.train(inst)
        elapsed = (time.perf_counter() - t0) * 1000
        per_inst = elapsed / len(instances)
        print(f"  {label:<30}  total={elapsed:.1f}ms  per-instance={per_inst:.3f}ms  "
              f"({len(instances)} instances)")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bench_p1()
    bench_p2()
    bench_p3()
    bench_p4()
    bench_e2e()
    print()
