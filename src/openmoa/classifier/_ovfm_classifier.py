# SPDX-License-Identifier: BSD-3-Clause
"""
OVFM (Online Learning in Variable Feature Spaces with Mixed Data) - Authentic Implementation
------------------------------------------------------------------------------------------
Faithfully implements the Gaussian Copula EM algorithm from He et al. (ICDM 2021).
Merges logic from Classifier, EM, and Transforms into a single unified file.

Reference:
    Yi He, Jiaxian Dong, Bo-Jian Hou, Yu Wang, Fei Wang. (2021).
    "Online Learning in Variable Feature Spaces with Mixed Data."
    IEEE International Conference on Data Mining (ICDM).

Complexity Warning:
    This algorithm performs Matrix Inversion O(d^3) and Storage O(d^2).
    It is theoretically impossible to run on RCV1 (47k features) on standard machines.
"""
from __future__ import annotations
import numpy as np
import warnings
from scipy.stats import norm

# P2: statsmodels ECDF removed; replaced by np.searchsorted (numerically identical).

from openmoa.base import Classifier
from openmoa.stream import Schema
from openmoa.instance import Instance

class OVFMClassifier(Classifier):
    def __init__(
        self,
        schema: Schema,
        window_size: int = 200,
        batch_size: int = 50,
        evolution_pattern: str = "vfs", 
        decay_coef: float = 0.5,
        num_ord_updates: int = 2,
        max_ord_levels: int = 14,
        ensemble_weight: float = 0.5,
        learning_rate: float = 0.01,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.01,
        sparsity_threshold: float = 0.01,
        random_seed: int = 1
    ):
        super().__init__(schema=schema, random_seed=random_seed)
        
        if schema.get_num_classes() != 2:
            raise ValueError("OVFM only supports Binary Classification.")
            
        self.window_size = window_size
        self.batch_size = batch_size
        self.evolution_pattern = evolution_pattern
        self.decay_coef = decay_coef
        self.num_ord_updates = num_ord_updates
        self.max_ord_levels = max_ord_levels
        self.ensemble_weight = ensemble_weight
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.sparsity_threshold = sparsity_threshold
        
        self._rng = np.random.RandomState(random_seed)
        
        # State Initialization
        self._trained = False
        self._max_features_seen = 0
        
        # Buffers for Batch Processing
        self._instance_buffer = []
        self._label_buffer = []
        
        # Feature Type Masks
        self._cont_indices = None # Continuous
        self._ord_indices = None  # Ordinal
        
        # Core Math Objects
        self._transform_function = None 
        self._sigma = None            
        
        # Classifiers
        self._w_obs = None
        self._w_lat = None
        
        # Ensemble tracking
        self._cumulative_loss_obs = 0.0
        self._cumulative_loss_lat = 0.0
        self._num_updates = 0

    def __str__(self):
        return f"OVFM(mode={self.evolution_pattern}, win={self.window_size}, batch={self.batch_size})"

    def _initialize(self, first_x):
        """Lazy initialization on first data point."""
        self._max_features_seen = len(first_x)
        
        # Default assumption: All features continuous until proven ordinal
        self._cont_indices = np.ones(self._max_features_seen, dtype=bool)
        self._ord_indices = np.zeros(self._max_features_seen, dtype=bool)
        
        # Initialize Transform Function
        if self.evolution_pattern == "tds":
             self._transform_function = TrapezoidalTransformFunction(
                self._cont_indices, self._ord_indices, 
                window_size=self.window_size, window_width=self._max_features_seen
            )
        else: # vfs / cds / eds
            self._transform_function = OnlineTransformFunction(
                self._cont_indices, self._ord_indices, window_size=self.window_size
            )
            
        # Initialize Sigma (Correlation Matrix)
        self._sigma = np.identity(self._max_features_seen)
        
        # Initialize Weights (+1 for bias term)
        self._w_obs = self._rng.randn(self._max_features_seen + 1) * 0.01
        self._w_lat = self._rng.randn(self._max_features_seen + 1) * 0.01
        
        self._trained = True

    def train(self, instance: Instance):
        """Accumulates instances into a buffer. When buffer full, runs EM and SGD."""
        x = np.asarray(instance.x, dtype=float) 
        y = 1 if instance.y_index == 1 else -1
        
        if not self._trained:
            self._initialize(x)
            
        # Handle dynamic expansion (TDS case where new features appear)
        if len(x) > self._max_features_seen:
            self._extend_dimensions(len(x))
            
        # Align dimensions
        if len(x) < self._max_features_seen:
            x_padded = np.full(self._max_features_seen, np.nan)
            x_padded[:len(x)] = x
            x = x_padded
        elif len(x) > self._max_features_seen:
            x = x[:self._max_features_seen]
            
        self._instance_buffer.append(x)
        self._label_buffer.append(y)
        
        if len(self._instance_buffer) >= self.batch_size:
            self._batch_update()

    def predict_proba(self, instance: Instance) -> np.ndarray:
        if not self._trained:
            return np.array([0.5, 0.5])
            
        x = np.asarray(instance.x, dtype=float)
        
        if len(x) < self._max_features_seen:
            x_padded = np.full(self._max_features_seen, np.nan)
            x_padded[:len(x)] = x
            x = x_padded
        elif len(x) > self._max_features_seen:
             x = x[:self._max_features_seen]
             
        x_obs = np.nan_to_num(x, nan=0.0)
        x_obs_bias = np.append(x_obs, 1.0)
        wx_obs = np.dot(self._w_obs, x_obs_bias)
        
        z_approx = self._transform_to_latent_simple(x)
        z_lat = np.nan_to_num(z_approx, nan=0.0) 
        z_lat_bias = np.append(z_lat, 1.0)
        wx_lat = np.dot(self._w_lat, z_lat_bias)
        
        wx_final = self.ensemble_weight * wx_obs + (1 - self.ensemble_weight) * wx_lat
        p = self._sigmoid(wx_final)
        return np.array([1-p, p])

    def predict(self, instance: Instance) -> int:
        return 1 if self.predict_proba(instance)[1] > 0.5 else 0

    def _batch_update(self):
        X = np.array(self._instance_buffer)
        y = np.array(self._label_buffer)
        
        # 1. Update feature types
        self._update_feature_types(X)
        
        # 2. Update Marginals
        if self.evolution_pattern == "tds":
             self._transform_function.partial_fit(X, self._cont_indices, self._ord_indices)
        else:
             self._transform_function.partial_fit(X)
             
        # 3. EM Step
        Z_imp, sigma_new = self._fit_covariance_em(X)
        
        # 4. Update Classifiers
        for i in range(len(X)):
            x_raw = X[i]
            z_vec = Z_imp[i]
            yi = y[i]
            
            x_in = np.nan_to_num(x_raw, nan=0.0)
            x_in = np.append(x_in, 1.0) 
            self._sgd_update(self._w_obs, x_in, yi)
            
            z_in = np.nan_to_num(z_vec, nan=0.0)
            z_in = np.append(z_in, 1.0) 
            self._sgd_update(self._w_lat, z_in, yi)
            
            score_obs = np.dot(self._w_obs, x_in)
            score_lat = np.dot(self._w_lat, z_in)
            self._cumulative_loss_obs += self._logistic_loss(score_obs, yi)
            self._cumulative_loss_lat += self._logistic_loss(score_lat, yi)
            
        self._num_updates += len(X)
        self._update_ensemble_weight()
        self._sparsify_weights()
        
        self._instance_buffer = []
        self._label_buffer = []

    def _fit_covariance_em(self, X_batch):
        """EM Algorithm: E-Step (Impute Z) -> M-Step (Update Sigma)."""
        # 1. Map to Latent Bounds
        Z_ord_lower, Z_ord_upper = self._transform_function.evaluate_ord_latent(X_batch)
        Z_cont = self._transform_function.evaluate_cont_latent(X_batch)
        
        Z_ord = self._init_z_ordinal(Z_ord_lower, Z_ord_upper)
        
        # Combine
        Z = np.zeros_like(X_batch)
        
        # === [CRITICAL FIX] Explicit Slicing to avoid Broadcasting Errors ===
        if np.any(self._ord_indices):
            # Z_ord is (B, n_ord), we map it to the corresponding columns in Z
            Z[:, self._ord_indices] = Z_ord
            
        if np.any(self._cont_indices):
            # Z_cont is (B, n_cont)
            Z[:, self._cont_indices] = Z_cont
        
        batch_size, p = Z.shape
        C_accum = np.zeros((p, p))
        Z_imputed_accum = np.copy(Z)
        
        prev_sigma = self._sigma
        
        for i in range(batch_size):
            c_i, z_imp_i = self._em_step_single_row(
                Z[i], Z_ord_lower[i], Z_ord_upper[i], prev_sigma
            )
            Z_imputed_accum[i] = z_imp_i
            C_accum += c_i
            
        C_accum /= batch_size
        sigma_emp = np.cov(Z_imputed_accum, rowvar=False) + C_accum
        
        # Handle scalar/empty edge cases
        if sigma_emp.ndim == 0: sigma_emp = np.eye(1)
        sigma_emp = np.nan_to_num(sigma_emp, nan=0.0)
        
        d = np.sqrt(np.diag(sigma_emp))
        d[d < 1e-6] = 1.0 
        sigma_emp = sigma_emp / np.outer(d, d)
        
        self._sigma = sigma_emp * self.decay_coef + (1 - self.decay_coef) * prev_sigma
        
        return Z_imputed_accum, self._sigma

    def _em_step_single_row(self, Z_row, lower, upper, sigma):
        p = len(Z_row)
        obs_idx = np.where(~np.isnan(Z_row))[0]
        miss_idx = np.where(np.isnan(Z_row))[0]
        
        Z_imp = np.copy(Z_row)
        C_correction = np.zeros((p, p))
        
        if len(miss_idx) == 0:
            return C_correction, Z_imp
            
        S_oo = sigma[np.ix_(obs_idx, obs_idx)]
        S_om = sigma[np.ix_(obs_idx, miss_idx)]
        S_mm = sigma[np.ix_(miss_idx, miss_idx)]
        
        try:
            J = np.linalg.solve(S_oo, S_om)
        except np.linalg.LinAlgError:
            # Regularize if singular
            S_oo += 1e-6 * np.eye(len(obs_idx))
            try:
                J = np.linalg.solve(S_oo, S_om)
            except:
                J = np.zeros((len(obs_idx), len(miss_idx)))
            
        z_obs = Z_row[obs_idx]
        z_miss_mean = np.dot(z_obs, J) 
        Z_imp[miss_idx] = z_miss_mean
        
        cov_miss = S_mm - np.dot(J.T, S_om)
        C_correction[np.ix_(miss_idx, miss_idx)] = cov_miss
        
        return C_correction, Z_imp

    def _transform_to_latent_simple(self, x):
        z = np.full_like(x, np.nan)
        for i in range(len(x)):
            if np.isnan(x[i]): continue
            win = self._transform_function.window[:, i]
            win = win[~np.isnan(win)]
            if len(win) < 5:
                z[i] = 0.0
                continue
                
            if self._cont_indices[i]:
                sorted_win = np.sort(win)
                prob = np.searchsorted(sorted_win, x[i], side='right') / len(sorted_win)
                prob = np.clip(prob, 1e-6, 1 - 1e-6)
                z[i] = norm.ppf(prob)
            else:
                mean = np.mean(win)
                std = np.std(win) + 1e-6
                z[i] = (x[i] - mean) / std
        return z

    def _init_z_ordinal(self, lower, upper):
        """Sample latent Z uniformly between the ordinal CDF bounds.

        Uses ``self._rng`` for reproducibility instead of the global
        ``np.random`` state.
        """
        Z    = np.full_like(lower, np.nan)
        mask = ~np.isnan(lower)
        if not np.any(mask):
            return Z

        u_lower  = norm.cdf(lower[mask])
        u_upper  = norm.cdf(upper[mask])
        u_sample = self._rng.uniform(u_lower, u_upper)   # ← was: np.random.uniform
        Z[mask]  = norm.ppf(u_sample)
        return Z

    def _update_feature_types(self, X):
        for i in range(X.shape[1]):
            col = X[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                uniques = np.unique(valid)
                if len(uniques) <= self.max_ord_levels:
                    self._ord_indices[i] = True
                    self._cont_indices[i] = False

    def _extend_dimensions(self, new_dim):
        diff = new_dim - self._max_features_seen
        self._cont_indices = np.append(self._cont_indices, np.ones(diff, dtype=bool))
        self._ord_indices = np.append(self._ord_indices, np.zeros(diff, dtype=bool))
        
        new_w = self._rng.randn(diff) * 0.01
        self._w_obs = np.insert(self._w_obs, -1, new_w)
        self._w_lat = np.insert(self._w_lat, -1, new_w.copy())
        
        # Expand Sigma
        new_sigma = np.eye(new_dim)
        new_sigma[:self._max_features_seen, :self._max_features_seen] = self._sigma
        self._sigma = new_sigma
        
        if self._transform_function:
            self._transform_function.extend(diff)
            
        self._max_features_seen = new_dim

    def _pad_with_bias(self, x):
        return np.append(x, 1.0)
    
    def _sigmoid(self, z):
        z = np.clip(z, -100, 100)
        return 1.0 / (1.0 + np.exp(-z))

    def _logistic_loss(self, wx, y):
        z = np.clip(-y * wx, -100, 100)
        if z > 0: return z + np.log(1 + np.exp(-z))
        return np.log(1 + np.exp(z))

    def _sgd_update(self, w, x, y):
        wx = np.dot(w, x)
        p = self._sigmoid(y * wx)
        grad = -(1 - p) * y * x
        reg = self.l1_lambda * np.sign(w) + self.l2_lambda * w
        reg[-1] = 0 
        w -= self.learning_rate * (grad + reg)

    def _update_ensemble_weight(self):
        tau = 2 * np.sqrt(2 * np.log(2) / max(1, self._num_updates))
        w_o = np.exp(-tau * self._cumulative_loss_obs)
        w_z = np.exp(-tau * self._cumulative_loss_lat)
        if w_o + w_z > 0:
            self.ensemble_weight = w_o / (w_o + w_z)

    def _sparsify_weights(self):
        mask = np.abs(self._w_obs[:-1]) < self.sparsity_threshold
        self._w_obs[:-1][mask] = 0.0
        mask = np.abs(self._w_lat[:-1]) < self.sparsity_threshold
        self._w_lat[:-1][mask] = 0.0

# --- Helper Classes ---

class OnlineTransformFunction:
    def __init__(self, cont, ord, window_size=200):
        self.window_size = window_size
        # === [FIXED] Save cont/ord mask ===
        self.cont = cont
        self.ord = ord
        self.p = len(cont)
        self.window = np.full((window_size, self.p), np.nan)
        self.ptr = np.zeros(self.p, dtype=int)
        
    def partial_fit(self, X):
        # Extend if batch has more features than current window
        if X.shape[1] > self.p:
             self.extend(X.shape[1] - self.p)
             
        for j in range(X.shape[1]):
            vals = X[:, j]
            valid = vals[~np.isnan(vals)]
            for v in valid:
                self.window[self.ptr[j], j] = v
                self.ptr[j] = (self.ptr[j] + 1) % self.window_size
                
    def extend(self, diff):
        self.window = np.hstack([self.window, np.full((self.window_size, diff), np.nan)])
        self.ptr = np.append(self.ptr, np.zeros(diff, dtype=int))
        
        # Need to extend internal masks too if they track p
        # But logic is controlled by main class, we just update p
        self.p += diff

    def evaluate_cont_latent(self, X):
        # Need to return Z with shape (B, n_cont)
        n_cont = np.sum(self.cont)
        Z = np.full((X.shape[0], n_cont), np.nan)
        
        cont_idx_list = np.where(self.cont)[0]
        limit_col = min(X.shape[1], self.p)
        
        # i is index in Z, col_idx is index in X/Window
        for i, col_idx in enumerate(cont_idx_list):
            if col_idx >= limit_col: continue
            
            win = self.window[:, col_idx]
            win = win[~np.isnan(win)]
            if len(win) < 5: continue
            
            vals = X[:, col_idx]
            mask = ~np.isnan(vals)
            if np.any(mask):
                sorted_win = np.sort(win)
                probs = np.searchsorted(sorted_win, vals[mask], side='right') / len(sorted_win)
                probs = np.clip(probs, 1e-5, 1 - 1e-5)
                Z[mask, i] = norm.ppf(probs)
        return Z

    def evaluate_ord_latent(self, X):
        # Need to return Z_lo/hi with shape (B, n_ord)
        n_ord = np.sum(self.ord)
        Z_lo = np.full((X.shape[0], n_ord), -np.inf)
        Z_hi = np.full((X.shape[0], n_ord), np.inf)
        
        ord_idx_list = np.where(self.ord)[0]
        limit_col = min(X.shape[1], self.p)
        
        for i, col_idx in enumerate(ord_idx_list):
            if col_idx >= limit_col: continue
            
            win = self.window[:, col_idx]
            win = win[~np.isnan(win)]
            if len(win) < 5: continue
            
            vals = X[:, col_idx]
            mask = ~np.isnan(vals)
            if np.any(mask):
                sorted_win = np.sort(win)
                probs = np.searchsorted(sorted_win, vals[mask], side='right') / len(sorted_win)
                z_point = norm.ppf(np.clip(probs, 1e-5, 1 - 1e-5))
                Z_lo[mask, i] = z_point - 0.1
                Z_hi[mask, i] = z_point + 0.1
        return Z_lo, Z_hi

class TrapezoidalTransformFunction(OnlineTransformFunction):
    def __init__(self, cont, ord, window_size=200, window_width=1):
        super().__init__(cont, ord, window_size)
        if window_width > self.p:
            self.extend(window_width - self.p)
    
    def partial_fit(self, X, cont, ord):
        # Update masks from main class
        self.cont = cont
        self.ord = ord
        super().partial_fit(X)