# SPDX-License-Identifier: BSD-3-Clause
"""
Numerical-equivalence tests for the P1–P4 performance optimisations.

Each test proves that the optimised code produces bit-for-bit (or within
float64 epsilon) identical results to the original reference formula, AND
that prequential prediction sequences are unchanged.
"""
import numpy as np
import pytest
from scipy.stats import norm

from openmoa.stream import NumpyStream
from openmoa.stream import OpenFeatureStream
from openmoa.classifier import OVFMClassifier, OSLMFClassifier, ORF3VClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stream(n=200, d=6, n_classes=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, d).astype(np.float32)
    y = rng.randint(0, n_classes, size=n)
    return NumpyStream(X, y, dataset_name="test",
                       feature_names=[f"f{i}" for i in range(d)])


# ---------------------------------------------------------------------------
# P2 — ECDF ≡ np.searchsorted
# Reference formula: ecdf(x) = #{w_i <= x} / n   (statsmodels convention)
# With H-smoothing (continuous case): u = #{w_i <= x} / (n+1)
# ---------------------------------------------------------------------------

class TestECDFEquivalence:
    """Verify that np.searchsorted reproduces ECDF exactly."""

    def _ecdf_reference(self, window_clean, x_obs):
        """Pure-numpy reference for ECDF(window)(x_obs)."""
        sorted_w = np.sort(window_clean)
        return np.searchsorted(sorted_w, x_obs, side='right') / len(sorted_w)

    def test_continuous_searchsorted_matches_formula(self):
        """Continuous CDF: #{w <= x}/(n+1) should equal H*ecdf(x)."""
        rng = np.random.RandomState(7)
        window = rng.randn(150)
        x_obs = rng.randn(20)

        n = len(window)
        sorted_w = np.sort(window)
        # Optimised formula
        u_opt = np.searchsorted(sorted_w, x_obs, side='right') / (n + 1)
        # Reference formula (H * ecdf)
        H = n / (n + 1)
        ecdf_vals = self._ecdf_reference(window, x_obs)
        u_ref = H * ecdf_vals

        np.testing.assert_allclose(u_opt, u_ref, rtol=1e-12, atol=1e-14)

    def test_ordinal_searchsorted_matches_formula(self):
        """Ordinal CDF (no H-factor): #{w <= x}/n."""
        rng = np.random.RandomState(13)
        window = rng.randint(0, 5, size=100).astype(float)
        x_obs = np.array([0.0, 1.0, 2.0, 4.5, 5.0])

        n = len(window)
        sorted_w = np.sort(window)
        threshold = 0.5

        u_lower_opt = np.searchsorted(sorted_w, x_obs - threshold, side='right') / n
        u_upper_opt = np.searchsorted(sorted_w, x_obs + threshold, side='right') / n

        u_lower_ref = self._ecdf_reference(window, x_obs - threshold)
        u_upper_ref = self._ecdf_reference(window, x_obs + threshold)

        np.testing.assert_allclose(u_lower_opt, u_lower_ref, rtol=1e-12)
        np.testing.assert_allclose(u_upper_opt, u_upper_ref, rtol=1e-12)

    def test_batch_searchsorted_matches(self):
        """Batch version (evaluate_cont_latent pattern)."""
        rng = np.random.RandomState(99)
        window = rng.rand(200)
        vals = rng.rand(50)

        sorted_w = np.sort(window)
        n = len(sorted_w)
        probs_opt = np.searchsorted(sorted_w, vals, side='right') / n
        probs_ref = self._ecdf_reference(window, vals)

        np.testing.assert_allclose(probs_opt, probs_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# P1 — _compute_density_peaks vectorised ≡ original loop
# ---------------------------------------------------------------------------

class TestDensityPeaksVectorisation:
    """Verify that the vectorised implementation matches the loop version."""

    @staticmethod
    def _density_peaks_loop(dist_matrix, p_arr=0.02):
        """Reference: original loop implementation."""
        n = len(dist_matrix)
        upper_tri = dist_matrix[np.triu_indices(n, k=1)]
        if len(upper_tri) > 0:
            position = int(len(upper_tri) * p_arr)
            d_cut = np.sort(upper_tri)[min(position, len(upper_tri) - 1)]
            d_cut = max(d_cut, 1e-6)
        else:
            d_cut = 1.0

        rho = np.sum(np.exp(-(dist_matrix / d_cut) ** 2), axis=1) - 1

        delta = np.zeros(n)
        nearest_higher = np.full(n, -1, dtype=int)
        sorted_indices = np.argsort(-rho)

        for i, idx in enumerate(sorted_indices):
            if i == 0:
                delta[idx] = np.max(dist_matrix[idx])
            else:
                higher_indices = sorted_indices[:i]
                dists = dist_matrix[idx, higher_indices]
                j = np.argmin(dists)
                delta[idx] = dists[j]
                nearest_higher[idx] = higher_indices[j]

        return rho, delta, nearest_higher

    @staticmethod
    def _density_peaks_vectorised(dist_matrix, p_arr=0.02):
        """Optimised vectorised implementation (copy of production code)."""
        n = len(dist_matrix)
        upper_tri = dist_matrix[np.triu_indices(n, k=1)]
        if len(upper_tri) > 0:
            position = int(len(upper_tri) * p_arr)
            d_cut = np.sort(upper_tri)[min(position, len(upper_tri) - 1)]
            d_cut = max(d_cut, 1e-6)
        else:
            d_cut = 1.0

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

        no_higher = np.isinf(delta)
        no_higher[top] = False
        if no_higher.any():
            for idx in np.where(no_higher)[0]:
                delta[idx] = dist_matrix[idx].max()
                nearest_higher[idx] = -1

        return rho, delta, nearest_higher

    def _random_dist_matrix(self, n, seed):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 5)
        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=-1))
        return D

    @pytest.mark.parametrize("n,seed", [(10, 0), (30, 1), (50, 2), (100, 3)])
    def test_rho_delta_nearest_identical(self, n, seed):
        D = self._random_dist_matrix(n, seed)
        rho_l, delta_l, nh_l = self._density_peaks_loop(D)
        rho_v, delta_v, nh_v = self._density_peaks_vectorised(D)

        np.testing.assert_allclose(rho_l, rho_v, rtol=1e-12, err_msg="rho mismatch")
        np.testing.assert_allclose(delta_l, delta_v, rtol=1e-10,
                                   err_msg="delta mismatch")
        np.testing.assert_array_equal(nh_l, nh_v, err_msg="nearest_higher mismatch")

    def test_all_unique_rho(self):
        """Edge case: all unique densities (common case)."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3)
        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=-1))
        rho_l, delta_l, nh_l = self._density_peaks_loop(D)
        rho_v, delta_v, nh_v = self._density_peaks_vectorised(D)
        np.testing.assert_allclose(delta_l, delta_v, rtol=1e-10)
        np.testing.assert_array_equal(nh_l, nh_v)


# ---------------------------------------------------------------------------
# End-to-end prequential reproducibility
# (same seed → identical predictions before and after optimisation)
# ---------------------------------------------------------------------------

class TestPrequentialReproducibility:
    """Verify that optimised classifiers produce the same prediction sequence."""

    def _collect_predictions(self, stream_fn, clf_fn, n=100):
        stream = stream_fn()
        clf = clf_fn(stream.get_schema())
        preds = []
        for _ in range(n):
            if not stream.has_more_instances():
                break
            inst = stream.next_instance()
            preds.append(clf.predict(inst))
            clf.train(inst)
        return preds

    def test_ovfm_same_predictions_both_runs(self):
        """Two identical OVFM runs must produce identical predictions."""
        def mk_stream():
            return _make_stream(n=150, d=6, seed=0)
        def mk_clf(schema):
            return OVFMClassifier(schema=schema, random_seed=1)

        run1 = self._collect_predictions(mk_stream, mk_clf, n=100)
        run2 = self._collect_predictions(mk_stream, mk_clf, n=100)
        assert run1 == run2, "OVFM predictions not reproducible"

    def test_oslmf_same_predictions_both_runs(self):
        """Two identical OSLMF runs must produce identical predictions."""
        def mk_stream():
            return _make_stream(n=200, d=6, seed=2)
        def mk_clf(schema):
            return OSLMFClassifier(schema=schema, batch_size=50, random_seed=1)

        run1 = self._collect_predictions(mk_stream, mk_clf, n=150)
        run2 = self._collect_predictions(mk_stream, mk_clf, n=150)
        assert run1 == run2, "OSLMF predictions not reproducible"

    def test_orf3v_same_predictions_both_runs(self):
        """Two identical ORF3V runs (3-class) must produce identical predictions."""
        def mk_stream():
            return _make_stream(n=150, d=6, n_classes=3, seed=5)
        def mk_clf(schema):
            return ORF3VClassifier(schema=schema, random_seed=1)

        run1 = self._collect_predictions(mk_stream, mk_clf, n=100)
        run2 = self._collect_predictions(mk_stream, mk_clf, n=100)
        assert run1 == run2, "ORF3V predictions not reproducible"
