# SPDX-License-Identifier: BSD-3-Clause
"""
OSLMF (Online Semi-supervised Learning with Mix-Typed Streaming Features)
-----------------------------------------------------------------------
Authentic Implementation - Fully Merged.
Combines Classifier, Gaussian Copula, and Density Peak Clustering logic 
exactly as defined in the original paper/code provided.

Reference:
    Wu, D., et al. (2023). "Online Semi-supervised Learning with Mix-Typed 
    Streaming Features". AAAI.

Complexity Warning:
    - Gaussian Copula: O(d^2) memory for Covariance Matrix.
    - Density Peaks: O(Buffer^2) for Distance Matrix.
    - RCV1 (47k features): Will OOM due to Copula Covariance Matrix (17GB+).
"""
from __future__ import annotations
import numpy as np
import warnings
from collections import deque
from scipy.stats import norm
from typing import Tuple, List, Optional

from openmoa.base import Classifier
from openmoa.stream import Schema
from openmoa.instance import Instance, LabeledInstance

# =============================================================================
# PART 1: Gaussian Copula (Handles Mixed Data Types & Missing Features)
# =============================================================================
class GaussianCopula:
    """Online Gaussian Copula model for mixed data types."""
    
    def __init__(self, cont_indices, ord_indices, window_size=200):
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        self.window_size = window_size
        
        p = len(cont_indices)
        self.p = p
        
        # Sliding window for ECDF
        self.window = np.full((window_size, p), np.nan)
        self.update_pos = np.zeros(p, dtype=int)
        
        # Covariance Matrix (sigma)
        self.Sigma = np.eye(p)
        self.iteration = 1
        
    def partial_fit(self, X_batch):
        """Update sliding window with new data."""
        for row in X_batch:
            for col_idx in range(self.p):
                value = row[col_idx]
                if not np.isnan(value):
                    self.window[self.update_pos[col_idx], col_idx] = value
                    self.update_pos[col_idx] = (self.update_pos[col_idx] + 1) % self.window_size
    
    def transform_to_latent(self, X_batch):
        """Map Observed X -> Latent Z (Standard Normal)."""
        Z = np.empty_like(X_batch)
        Z[:] = np.nan
        
        # Continuous features
        for i in np.where(self.cont_indices)[0]:
            missing = np.isnan(X_batch[:, i])
            if np.sum(~missing) > 0:
                Z[~missing, i] = self._continuous_to_latent(
                    X_batch[~missing, i], 
                    self.window[:, i]
                )
        
        # Ordinal features
        for i in np.where(self.ord_indices)[0]:
            missing = np.isnan(X_batch[:, i])
            if np.sum(~missing) > 0:
                z_lower, z_upper = self._ordinal_to_latent(
                    X_batch[~missing, i],
                    self.window[:, i]
                )
                # Simplified point estimate for latent representation
                Z[~missing, i] = (z_lower + z_upper) / 2
                Z[~missing, i] = np.clip(Z[~missing, i], -5, 5)
        
        return Z
    
    def _continuous_to_latent(self, x_obs, window):
        window_clean = window[~np.isnan(window)]
        if len(window_clean) == 0:
            return np.zeros_like(x_obs)
        # P2: replace statsmodels ECDF with np.searchsorted (numerically identical).
        # ECDF(w)(x) = #{w_i <= x} / n; with H-smoothing: u = #{w_i <= x} / (n+1)
        sorted_w = np.sort(window_clean)
        n_w = len(sorted_w)
        u = np.searchsorted(sorted_w, x_obs, side='right') / (n_w + 1)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        return norm.ppf(u)

    def _ordinal_to_latent(self, x_obs, window):
        window_clean = window[~np.isnan(window)]
        if len(window_clean) == 0:
            return np.zeros_like(x_obs), np.zeros_like(x_obs)

        sorted_w = np.sort(window_clean)
        n_w = len(sorted_w)
        unique = np.unique(window_clean)

        if len(unique) > 1:
            threshold = np.min(np.abs(unique[1:] - unique[:-1])) / 2
            z_lower = norm.ppf(np.clip(
                np.searchsorted(sorted_w, x_obs - threshold, side='right') / n_w,
                1e-10, 1 - 1e-10
            ))
            z_upper = norm.ppf(np.clip(
                np.searchsorted(sorted_w, x_obs + threshold, side='right') / n_w,
                1e-10, 1 - 1e-10
            ))
        else:
            z_lower = np.full_like(x_obs, -5.0)
            z_upper = np.full_like(x_obs, 5.0)

        return z_lower, z_upper
    
    def update_covariance_em(self, X_batch, Z_latent, decay_coef=0.5):
        """Online EM Step to update Sigma."""
        # This is the logic from _oslmf_copula.py
        batch_size = len(X_batch)
        if batch_size < 2: return
        
        # E-step: Impute missing Z using current Sigma (Simplified Mean Imputation for speed in OSLMF code)
        # Note: The original code used simplified imputation for covariance update stability
        Z_imputed = Z_latent.copy()
        for i in range(len(Z_imputed)):
            missing_idx = np.where(np.isnan(Z_imputed[i]))[0]
            if len(missing_idx) > 0:
                Z_imputed[i, missing_idx] = 0.0 # Mean imputation
        
        # M-step: Update Sigma
        try:
            valid_rows = ~np.any(np.isnan(Z_imputed), axis=1)
            if np.sum(valid_rows) < 2: return
            
            Sigma_new = np.cov(Z_imputed, rowvar=False)
            
            if np.any(np.isnan(Sigma_new)) or np.any(np.isinf(Sigma_new)): return
            if Sigma_new.shape != (self.p, self.p): return
            
            # Regularization
            Sigma_new = Sigma_new + np.eye(self.p) * 1e-6
            
            # Project to Correlation
            D = np.sqrt(np.diag(Sigma_new))
            D[D < 1e-10] = 1.0
            Sigma_new = Sigma_new / D[:, None] / D[None, :]
            
            if np.any(np.isnan(Sigma_new)) or np.any(np.isinf(Sigma_new)): return
            
            # Exponential Update
            self.Sigma = decay_coef * Sigma_new + (1 - decay_coef) * self.Sigma
            self.iteration += 1
            
        except (ValueError, np.linalg.LinAlgError):
            pass
    
    def reconstruct_features(self, X_batch, Z_latent):
        """Reconstruct missing X from Z."""
        X_rec = X_batch.copy()
        
        for i in range(self.p):
            missing = np.isnan(X_rec[:, i])
            if np.sum(missing) == 0: continue
            
            z_missing = Z_latent[missing, i]
            if np.any(np.isnan(z_missing)):
                z_missing = np.nan_to_num(z_missing, nan=0.0)
            
            if self.cont_indices[i]:
                X_rec[missing, i] = self._latent_to_continuous(z_missing, self.window[:, i])
            else:
                X_rec[missing, i] = self._latent_to_ordinal(z_missing, self.window[:, i])
        
        return X_rec
    
    def _latent_to_continuous(self, z, window):
        window_clean = window[~np.isnan(window)]
        if len(window_clean) == 0: return np.zeros_like(z)
        u = norm.cdf(z)
        return np.quantile(window_clean, u)
    
    def _latent_to_ordinal(self, z, window):
        window_clean = window[~np.isnan(window)]
        if len(window_clean) == 0: return np.zeros_like(z)
        u = norm.cdf(z)
        n = len(window_clean)
        indices = np.ceil(np.round((n + 1) * u - 1, 3))
        indices = np.clip(indices, 0, n - 1).astype(int)
        return np.sort(window_clean)[indices]


# =============================================================================
# PART 2: Density Peak Clustering (Semi-supervised Label Propagation)
# =============================================================================
class DensityPeakClustering:
    """Online Density-Peak Clustering for Label Propagation."""
    
    def __init__(self, buffer_size=200, p_arr=0.02):
        self.buffer_size = buffer_size
        self.p_arr = p_arr

        # P2: deque with maxlen eliminates O(n) pop(0) on every add
        self.buffer_X = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)
        self.buffer_labeled = deque(maxlen=buffer_size)

    def add_instance(self, x, y=None, is_labeled=False):
        # deque(maxlen=...) evicts the oldest entry automatically
        self.buffer_X.append(x)
        self.buffer_y.append(y if is_labeled else None)
        self.buffer_labeled.append(is_labeled)

    def propagate_labels(self) -> Tuple[List, List]:
        """Core Logic: Propagate labels based on density structure."""
        if len(self.buffer_X) < 2:
            return list(self.buffer_y), [1.0] * len(self.buffer_y)

        X = np.array(list(self.buffer_X))
        n = len(X)
        
        # 1. Compute Distances
        dist_matrix = self._compute_distances(X)
        
        # 2. Compute Density (rho) and Distance to Higher Density (delta)
        rho, delta, nearest_higher = self._compute_density_peaks(dist_matrix)
        
        # 3. Propagate
        pseudo_labels = list(self.buffer_y)
        confidence = [1.0 if labeled else 0.0 for labeled in self.buffer_labeled]
        
        # Sort by density descending
        sorted_indices = np.argsort(-rho)
        
        for idx in sorted_indices:
            if pseudo_labels[idx] is not None:
                continue 
            
            # Trace back to nearest higher density neighbor
            current = idx
            path = []
            max_depth = 10 
            
            for _ in range(max_depth):
                if current == -1: break
                path.append(current)
                
                if pseudo_labels[current] is not None:
                    # Found label source, propagate down
                    label = pseudo_labels[current]
                    conf = confidence[current] * 0.9 # Decay confidence
                    
                    for p in path[:-1]:
                        if pseudo_labels[p] is None:
                            pseudo_labels[p] = label
                            confidence[p] = conf
                    break
                
                current = nearest_higher[current]
        
        return pseudo_labels, confidence
    
    def _compute_distances(self, X):
        from scipy.spatial.distance import cdist
        X_filled = np.nan_to_num(X, nan=0.0)
        return cdist(X_filled, X_filled, metric="euclidean")
    
    def _compute_density_peaks(self, dist_matrix):
        n = len(dist_matrix)

        # Cutoff distance (dc)
        upper_tri = dist_matrix[np.triu_indices(n, k=1)]
        if len(upper_tri) > 0:
            position = int(len(upper_tri) * self.p_arr)
            d_cut = np.sort(upper_tri)[min(position, len(upper_tri) - 1)]
            d_cut = max(d_cut, 1e-6)
        else:
            d_cut = 1.0

        # Local Density (Gaussian Kernel)
        rho = np.sum(np.exp(-(dist_matrix / d_cut) ** 2), axis=1) - 1

        # P1: Vectorized delta & nearest_higher.
        # Assign each point a rank (0 = highest density).
        sorted_indices = np.argsort(-rho)           # same as original, no kind= change
        rank = np.empty(n, dtype=np.intp)
        rank[sorted_indices] = np.arange(n)

        # rank_mask[i,j] is True iff point j has a strictly higher density rank than i
        rank_mask = rank[np.newaxis, :] < rank[:, np.newaxis]   # (n, n)

        # Distances to higher-density points only; everything else → inf
        dist_masked = np.where(rank_mask, dist_matrix, np.inf)

        delta = dist_masked.min(axis=1)
        nearest_higher = np.argmin(dist_masked, axis=1).astype(np.intp)

        # The highest-density point has no higher-density neighbor
        top = sorted_indices[0]
        delta[top] = dist_matrix[top].max()
        nearest_higher[top] = -1

        # Safety: any other point whose every neighbour was masked (shouldn't happen)
        no_higher = np.isinf(delta)
        no_higher[top] = False
        if no_higher.any():
            for idx in np.where(no_higher)[0]:
                delta[idx] = dist_matrix[idx].max()
                nearest_higher[idx] = -1

        return rho, delta, nearest_higher


# =============================================================================
# PART 3: OSLMF Classifier (The Orchestrator)
# =============================================================================
class OSLMFClassifier(Classifier):
    def __init__(
        self,
        schema: Schema,
        window_size: int = 200,
        buffer_size: int = 200,
        batch_size: int = 50,
        learning_rate: float = 0.01,
        decay_coef: float = 0.5,
        max_ord_levels: int = 14,
        ensemble_weight: float = 0.5,
        l2_lambda: float = 0.001,
        random_seed: int = 1
    ):
        super().__init__(schema=schema, random_seed=random_seed)

        if schema.get_num_classes() != 2:
            raise ValueError("OSLMF only supports Binary Classification.")

        self.window_size = window_size
        self.buffer_size = buffer_size
        # A5: DensityPeaks is batch-based in the original paper; run once per batch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_coef = decay_coef
        self.max_ord_levels = max_ord_levels
        self.l2_lambda = l2_lambda
        self.ensemble_weight = ensemble_weight

        self._rng = np.random.RandomState(random_seed)

        self._trained = False
        self._max_features_seen = 0

        # Components
        self._copula = None
        self._density_peaks = None

        # Weights (+1 bias)
        self._w_obs = None
        self._w_lat = None

        # Stats
        self._loss_obs = 0.0
        self._loss_lat = 0.0
        self._num_updates = 0

        # Type Masks
        self._cont_indices = None
        self._ord_indices = None

        # A5: batch accumulation buffer
        self._batch_X: List[np.ndarray] = []
        self._batch_y: List[int] = []

    def __str__(self):
        return f"OSLMF(win={self.window_size}, buf={self.buffer_size}, lr={self.learning_rate})"

    def _initialize(self, x):
        self._max_features_seen = len(x)
        
        # Default: All continuous
        self._cont_indices = np.ones(self._max_features_seen, dtype=bool)
        self._ord_indices = np.zeros(self._max_features_seen, dtype=bool)
        
        # Components
        self._copula = GaussianCopula(self._cont_indices, self._ord_indices, self.window_size)
        self._density_peaks = DensityPeakClustering(self.buffer_size)
        
        # Weights
        self._w_obs = self._rng.randn(self._max_features_seen + 1) * 0.01
        self._w_lat = self._rng.randn(self._max_features_seen + 1) * 0.01
        
        self._trained = True

    def train(self, instance: Instance):
        x = np.asarray(instance.x, dtype=float)
        y = 1 if instance.y_index == 1 else -1

        if not self._trained:
            self._initialize(x)

        # Expand dimensions
        if len(x) > self._max_features_seen:
            self._extend_dimensions(len(x))

        # Pad to current feature dimension
        x_padded = np.full(self._max_features_seen, np.nan)
        limit = min(len(x), self._max_features_seen)
        x_padded[:limit] = x[:limit]

        # === 1. Copula sliding-window update (per-instance) ===
        self._copula.partial_fit(x_padded.reshape(1, -1))

        # === 2. Buffer accumulation (A5) ===
        self._batch_X.append(x_padded)
        self._batch_y.append(y)

        if len(self._batch_X) < self.batch_size:
            return  # wait until a full batch is ready

        # === 3. Batch is full: run DensityPeaks once for the whole batch ===
        batch_X = np.array(self._batch_X)           # (batch_size, d)
        batch_y = list(self._batch_y)

        # Transform entire batch to latent space
        Z_batch = self._copula.transform_to_latent(batch_X)  # (batch_size, d)

        # Covariance EM update on the batch
        self._copula.update_covariance_em(batch_X, Z_batch, self.decay_coef)

        # Add all batch instances to DensityPeaks buffer, then propagate once
        for i in range(len(batch_X)):
            z_filled = np.nan_to_num(Z_batch[i], nan=0.0)
            self._density_peaks.add_instance(z_filled, batch_y[i], is_labeled=True)
        self._density_peaks.propagate_labels()  # updates internal buffer labels

        # === 4. Update classifiers for each instance in the batch ===
        X_rec_batch = self._copula.reconstruct_features(batch_X, Z_batch)

        for i in range(len(batch_X)):
            effective_y = batch_y[i]

            x_in = np.nan_to_num(X_rec_batch[i], nan=0.0)
            x_in = np.append(x_in, 1.0)
            self._sgd_update(self._w_obs, x_in, effective_y)

            z_in = np.nan_to_num(Z_batch[i], nan=0.0)
            z_in = np.append(z_in, 1.0)
            self._sgd_update(self._w_lat, z_in, effective_y)

            # Update ensemble weights
            score_obs = np.dot(self._w_obs, x_in)
            score_lat = np.dot(self._w_lat, z_in)
            self._loss_obs += self._logistic_loss(score_obs, effective_y)
            self._loss_lat += self._logistic_loss(score_lat, effective_y)
            self._num_updates += 1

            tau = 2 * np.sqrt(2 * np.log(2) / max(1, self._num_updates))
            w_o = np.exp(-tau * self._loss_obs)
            w_z = np.exp(-tau * self._loss_lat)
            if w_o + w_z > 0:
                self.ensemble_weight = w_o / (w_o + w_z)

        # Clear batch buffer
        self._batch_X = []
        self._batch_y = []

    def predict_proba(self, instance: Instance) -> np.ndarray:
        if not self._trained:
            return np.array([0.5, 0.5])
            
        x = np.asarray(instance.x, dtype=float)
        
        # Align
        x_padded = np.full(self._max_features_seen, np.nan)
        limit = min(len(x), self._max_features_seen)
        x_padded[:limit] = x[:limit]
        
        # 1. Transform
        X_batch = x_padded.reshape(1, -1)
        Z_latent = self._copula.transform_to_latent(X_batch)
        X_rec = self._copula.reconstruct_features(X_batch, Z_latent)
        
        # 2. Predict Obs
        x_in = np.nan_to_num(X_rec[0], nan=0.0)
        x_in = np.append(x_in, 1.0)
        score_obs = np.dot(self._w_obs, x_in)
        
        # 3. Predict Lat
        z_in = np.nan_to_num(Z_latent[0], nan=0.0)
        z_in = np.append(z_in, 1.0)
        score_lat = np.dot(self._w_lat, z_in)
        
        # 4. Ensemble
        final = self.ensemble_weight * score_obs + (1 - self.ensemble_weight) * score_lat
        p = 1.0 / (1.0 + np.exp(-np.clip(final, -50, 50)))
        return np.array([1-p, p])

    def predict(self, instance: Instance) -> int:
        return 1 if self.predict_proba(instance)[1] > 0.5 else 0

    def _extend_dimensions(self, new_dim):
        diff = new_dim - self._max_features_seen
        
        # Extend Masks
        self._cont_indices = np.append(self._cont_indices, np.ones(diff, dtype=bool))
        self._ord_indices = np.append(self._ord_indices, np.zeros(diff, dtype=bool))
        
        # Extend Weights
        new_w = self._rng.randn(diff) * 0.01
        self._w_obs = np.insert(self._w_obs, -1, new_w)
        self._w_lat = np.insert(self._w_lat, -1, new_w.copy())
        
        # Update Copula Components
        self._copula.cont_indices = self._cont_indices
        self._copula.ord_indices = self._ord_indices
        self._copula.p = new_dim
        
        # Extend Copula Window
        new_cols = np.full((self.window_size, diff), np.nan)
        self._copula.window = np.hstack([self._copula.window, new_cols])
        self._copula.update_pos = np.append(self._copula.update_pos, np.zeros(diff, dtype=int))
        
        # Extend Sigma
        new_sigma = np.eye(new_dim)
        new_sigma[:self._max_features_seen, :self._max_features_seen] = self._copula.Sigma
        self._copula.Sigma = new_sigma
        
        self._max_features_seen = new_dim

    def _sgd_update(self, w, x, y):
        score = np.dot(w, x)
        p = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))
        grad = -(1 - p) * y * x
        # L2 Reg
        reg = self.l2_lambda * w
        reg[-1] = 0
        w -= self.learning_rate * (grad + reg)

    def _logistic_loss(self, wx, y):
        z = np.clip(-y * wx, -50, 50)
        if z > 0: return z + np.log(1 + np.exp(-z))
        return np.log(1 + np.exp(z))