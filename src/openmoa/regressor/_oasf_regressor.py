# SPDX-License-Identifier: BSD-3-Clause
"""openmoa/regressor/_oasf_regressor.py"""
from __future__ import annotations
from typing import Optional
import numpy as np

from openmoa.base import Regressor
from openmoa.stream import Schema
from openmoa.instance import Instance


class OASFRegressor(Regressor):
    """Online Active Sparse Feature learning (OASF) Regressor.
    
    Handles online regression with dynamically evolving feature spaces.
    Uses ℓ1,2-norm regularization for sparsity and passive-aggressive updates.
    
    Reference:
    
    Chen, Z., He, Y., Wu, D., et al. (2024).
    ℓ1,2-Norm and CUR Decomposition based Sparse Online Active Learning 
    for Data Streams with Streaming Features.
    IEEE International Conference on Big Data.
    
    Example:
    
    >>> from openmoa.datasets import Fried
    >>> from openmoa.regressor import OASFRegressor
    >>> from openmoa.stream import EvolvingFeatureStream
    >>> from openmoa.evaluation import prequential_evaluation
    >>> 
    >>> base_stream = Fried()
    >>> evolving_stream = EvolvingFeatureStream(base_stream, d_min=5, d_max=10)
    >>> learner = OASFRegressor(
    ...     schema=evolving_stream.get_schema(),
    ...     lambda_param=0.01,
    ...     mu=10,
    ...     L=100
    ... )
    >>> results = prequential_evaluation(evolving_stream, learner, max_instances=10000)
    """

    def __init__(
        self,
        schema: Schema,
        lambda_param: float = 0.01,
        mu: float = 10.0,
        eta: float = 1.0,
        L: int = 100,
        d_max: int = 1000,
        random_seed: int = 1,
    ):
        """Initialize OASF Regressor.
        
        :param schema: The schema of the stream
        :param lambda_param: Regularization parameter for ℓ1,2-norm sparsity
        :param mu: Penalty parameter for PA update
        :param eta: Regularization parameter for CUR decomposition
        :param L: Sliding window size
        :param d_max: Maximum expected feature dimension
        :param random_seed: Random seed for reproducibility
        """
        super().__init__(schema=schema, random_seed=random_seed)
        
        np.random.seed(random_seed)
        
        self.lambda_param = lambda_param
        self.mu = mu
        self.eta = eta
        self.L = L
        self.d_max = d_max
        
        self.W = np.zeros((d_max, L))
        self.X_cur = np.zeros((d_max, L))
        self.current_dim = 0
        self.t = 0
        self.selected_columns = []

    def __str__(self):
        return (f"OASFRegressor(lambda={self.lambda_param}, mu={self.mu}, "
                f"L={self.L}, d_max={self.d_max})")

    def train(self, instance: Instance):
        """Train on a single instance."""
        self.t += 1
        
        x_full = np.array(instance.x)
        d_current = len(x_full)
        y = instance.y_value
        
        if self.current_dim >= d_current:
            w_new = self._update_decremental(x_full, y, d_current)
        else:
            w_new = self._update_incremental(x_full, y, d_current)
        
        self.W = np.roll(self.W, -1, axis=1)
        self.W[:, -1] = 0
        self.W[:d_current, -1] = w_new
        
        self._apply_l12_sparsity()
        
        self.X_cur = np.roll(self.X_cur, -1, axis=1)
        self.X_cur[:, -1] = 0
        self.X_cur[:d_current, -1] = x_full
        
        if self.t % self.L == 0:
            self._update_cur()
        
        self.current_dim = d_current

    def predict(self, instance: Instance) -> float:
        """Predict target value."""
        x_full = np.array(instance.x)
        d_current = len(x_full)
        
        if self.current_dim >= d_current:
            w_pred = self.W[:d_current, -1]
            pred = np.dot(w_pred, x_full)
        else:
            w_padded = np.concatenate([
                self.W[:self.current_dim, -1],
                np.zeros(d_current - self.current_dim)
            ])
            pred = np.dot(w_padded, x_full)
        
        return pred

    def _update_decremental(self, xt: np.ndarray, yt: float, d_current: int) -> np.ndarray:
        """Update for decremental case (adapted for regression)."""
        w_s = self.W[:d_current, -1]
        
        pred = np.dot(w_s, xt)
        loss = max(0, abs(yt - pred) - 0.1)  # ε-insensitive loss
        
        if loss > 0:
            sign_error = np.sign(yt - pred)
            gamma = loss / (np.linalg.norm(xt)**2 + 1/(2*self.mu))
            w_new = w_s + gamma * sign_error * xt
        else:
            w_new = w_s
        
        return w_new

    def _update_incremental(self, xt: np.ndarray, yt: float, d_current: int) -> np.ndarray:
        """Update for incremental case (adapted for regression)."""
        x_s = xt[:self.current_dim]
        x_n = xt[self.current_dim:]
        
        w_padded = np.concatenate([
            self.W[:self.current_dim, -1],
            np.zeros(d_current - self.current_dim)
        ])
        
        pred = np.dot(w_padded, xt)
        loss = max(0, abs(yt - pred) - 0.1)
        
        if loss > 0:
            sign_error = np.sign(yt - pred)
            gamma = loss / (np.linalg.norm(xt)**2 + 1/(2*self.mu))
            w_s_new = self.W[:self.current_dim, -1] + gamma * sign_error * x_s
            w_n_new = gamma * sign_error * x_n
            w_new = np.concatenate([w_s_new, w_n_new])
        else:
            w_new = np.concatenate([
                self.W[:self.current_dim, -1], 
                np.zeros(len(x_n))
            ])
        
        return w_new

    def _apply_l12_sparsity(self):
        """Apply ℓ1,2-norm regularization."""
        for i in range(self.d_max):
            row = self.W[i, :]
            row_norm = np.linalg.norm(row)
            
            if row_norm <= self.lambda_param:
                self.W[i, :] = 0
            else:
                self.W[i, :] = (1 - self.lambda_param / row_norm) * row

    def _update_cur(self):
        """Update CUR decomposition (simplified)."""
        column_norms = np.sqrt(np.sum(self.W ** 2, axis=0))
        important_cols = np.argsort(column_norms)[::-1]
        self.selected_columns = important_cols[:min(self.L // 2, len(important_cols))]