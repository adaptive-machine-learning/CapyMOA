# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import numpy as np

from openmoa.base import Classifier, SparseInputMixin
from openmoa.stream import Schema
from openmoa.instance import Instance


class RSOLClassifier(SparseInputMixin, Classifier):
    """RSOL – Robust Sparse Online Learning.

    Key design decisions
    --------------------
    * **Ring buffer** for the sliding-window matrix W avoids costly
      ``np.roll`` copies (O(d·L) per step).
    * **Vectorised** L₁,₂ sparsity shrinkage – no Python loops.
    * **Auto-expanding** W to handle arbitrarily high-dimensional streams.
    * **Wrapper-aware** via :class:`~openmoa.base.SparseInputMixin`.

    Reference
    ---------
    Chen, Z., et al. (2024). Robust Sparse Online Learning … SDM.

    Only binary classification is supported.
    """

    def __init__(
        self,
        schema: Schema,
        lambda_param: float = 50.0,
        mu: float = 1.0,
        L: int = 1000,
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)

        if schema.get_num_classes() != 2:
            raise ValueError("RSOLClassifier only supports binary classification.")

        self.lambda_param = lambda_param
        self.mu           = mu
        self.L            = L

        # Instance-level RNG
        self._rng = np.random.RandomState(random_seed)

        initial_dim = max(schema.get_num_attributes(), 1)
        self.W = np.zeros((initial_dim, self.L))

        # Ring-buffer write pointer
        self._ptr = 0

        self.current_dim = 0
        self.t = 0

    def __str__(self):
        return f"RSOLClassifier(lambda={self.lambda_param}, mu={self.mu}, L={self.L})"

    # ------------------------------------------------------------------
    # Dynamic dimension
    # ------------------------------------------------------------------

    def _ensure_dimension(self, target_dim: int):
        if target_dim <= self.W.shape[0]:
            return
        new_rows = max(target_dim, int(self.W.shape[0] * 1.5))
        new_W = np.zeros((new_rows, self.L))
        new_W[:self.W.shape[0], :] = self.W
        self.W = new_W

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instance: Instance):
        self.t += 1

        indices, values = self._get_sparse_x(instance)
        d_current = int(np.max(indices) + 1) if len(indices) > 0 else self.current_dim
        self._ensure_dimension(d_current)

        y = 1 if instance.y_index == 1 else -1

        # Read previous weight from ring buffer
        prev_col = (self._ptr - 1) % self.L
        w_s = self.W[:d_current, prev_col].copy()

        xt = np.zeros(d_current)
        safe_idx = indices[indices < d_current]
        xt[safe_idx] = values[indices < d_current]

        loss  = max(0.0, 1.0 - y * np.dot(w_s, xt))
        denom = np.dot(xt, xt) + 1.0 / (2.0 * self.mu)
        gamma = loss / denom if denom > 0 else 0.0

        w_new = w_s + gamma * y * xt

        # Write into ring buffer
        self.W[:, self._ptr] = 0.0
        self.W[:d_current, self._ptr] = w_new

        self._apply_l12_sparsity(d_current)

        self._ptr = (self._ptr + 1) % self.L
        self.current_dim = d_current

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instance: Instance) -> int:
        return 1 if self.predict_proba(instance)[1] > 0.5 else 0

    def predict_proba(self, instance: Instance) -> np.ndarray:
        indices, values = self._get_sparse_x(instance)
        if len(indices) == 0:
            return np.array([0.5, 0.5])

        latest_col = (self._ptr - 1) % self.L
        w_pred     = self.W[:, latest_col]

        valid      = indices < w_pred.shape[0]
        margin     = np.dot(w_pred[indices[valid]], values[valid]) if np.any(valid) else 0.0

        prob = 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))
        return np.array([1.0 - prob, prob])

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def _apply_l12_sparsity(self, active_rows: int):
        W_sub     = self.W[:active_rows, :]
        row_norms = np.linalg.norm(W_sub, axis=1)

        zero_mask   = row_norms <= self.lambda_param
        shrink_mask = ~zero_mask

        self.W[:active_rows][zero_mask] = 0.0

        if np.any(shrink_mask):
            scales = 1.0 - self.lambda_param / row_norms[shrink_mask]
            self.W[:active_rows][shrink_mask] *= scales[:, np.newaxis]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_sparsity(self) -> float:
        latest_col = (self._ptr - 1) % self.L
        w_latest   = self.W[:self.current_dim, latest_col]
        if w_latest.size == 0:
            return 1.0
        return float(np.sum(np.abs(w_latest) < 1e-10)) / w_latest.size
