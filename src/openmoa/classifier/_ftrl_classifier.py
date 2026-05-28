# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import numpy as np

from openmoa.base import Classifier, SparseInputMixin
from openmoa.stream import Schema
from openmoa.instance import Instance


class FTRLClassifier(SparseInputMixin, Classifier):
    """FTRL-Proximal classifier (Follow the Regularized Leader).

    Supports binary (logistic) and multi-class (softmax) tasks.
    Per-coordinate adaptive learning rates via accumulated squared gradients.
    Weight arrays auto-expand to handle dynamic feature indices from wrapper
    streams (e.g. :class:`OpenFeatureStream` incremental mode).

    Reference
    ---------
    McMahan, H. B. (2011). Follow-the-Regularized-Leader and Mirror Descent:
    Equivalence Theorems and L1 Regularization. AISTATS.
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 0.1,
        beta:  float = 1.0,
        l1:    float = 1.0,
        l2:    float = 1.0,
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)

        if schema.is_regression():
            raise ValueError("FTRLClassifier does not support regression.")

        self.alpha = alpha
        self.beta  = beta
        self.l1    = l1
        self.l2    = l2

        self.n_classes = schema.get_num_classes()
        if self.n_classes == 2:
            self.task_type = "binary"
            self.n_outputs = 1
        else:
            self.task_type = "multiclass"
            self.n_outputs = self.n_classes

        self.n_features = schema.get_num_attributes()

        # Instance-level RNG
        self._rng = np.random.RandomState(random_seed)

        # FTRL accumulators: z (gradient accumulator), n (squared-grad sum), w (weights)
        self.z = np.zeros((self.n_features, self.n_outputs), dtype=np.float64)
        self.n = np.zeros((self.n_features, self.n_outputs), dtype=np.float64)
        self.w = np.zeros((self.n_features, self.n_outputs), dtype=np.float64)

    def __str__(self):
        return (f"FTRLClassifier(task={self.task_type}, alpha={self.alpha}, "
                f"beta={self.beta}, l1={self.l1}, l2={self.l2})")

    # ------------------------------------------------------------------
    # Dynamic dimension (A1 fix)
    # ------------------------------------------------------------------

    def _ensure_dimension(self, target_dim: int):
        """Expand z, n, w if feature indices from a dynamic stream exceed current size."""
        if target_dim <= self.n_features:
            return
        new_dim = max(target_dim, int(self.n_features * 1.5))
        for attr in ("z", "n", "w"):
            old = getattr(self, attr)
            new = np.zeros((new_dim, self.n_outputs), dtype=np.float64)
            new[:self.n_features] = old
            setattr(self, attr, new)
        self.n_features = new_dim

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instance: Instance):
        indices, values = self._get_sparse_x(instance)
        if len(indices) == 0:
            return

        max_idx = int(np.max(indices))
        self._ensure_dimension(max_idx + 1)

        w_active = self.w[indices]
        linear_pred = np.dot(values, w_active)   # shape (n_outputs,)

        if self.task_type == "binary":
            pred        = 1.0 / (1.0 + np.exp(-np.clip(linear_pred[0], -50, 50)))
            diff        = pred - instance.y_index
            grad        = np.outer(values, np.array([diff]))

        else:  # multiclass
            shift       = linear_pred - np.max(linear_pred)
            exp_s       = np.exp(shift)
            probs       = exp_s / np.sum(exp_s)
            diff        = probs.copy()
            diff[instance.y_index] -= 1.0
            grad        = np.outer(values, diff)

        # FTRL core update (active features only)
        n_active = self.n[indices]
        n_new    = n_active + grad ** 2
        sigma    = (np.sqrt(n_new) - np.sqrt(n_active)) / self.alpha
        z_new    = self.z[indices] + grad - sigma * w_active

        self.n[indices] = n_new
        self.z[indices] = z_new

        # Proximal step: zero-out features below L1 threshold
        sign_z  = np.sign(z_new)
        abs_z   = np.abs(z_new)
        denom   = (self.beta + np.sqrt(n_new)) / self.alpha + self.l2
        active_mask = abs_z > self.l1

        new_w = np.zeros_like(w_active)
        if np.any(active_mask):
            new_w[active_mask] = (
                -(z_new[active_mask] - sign_z[active_mask] * self.l1)
                / denom[active_mask]
            )
        self.w[indices] = new_w

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instance: Instance) -> int:
        indices, values = self._get_sparse_x(instance)
        if len(indices) == 0:
            return 0
        valid = indices < self.n_features
        xi, xv = indices[valid], values[valid]
        if len(xi) == 0:
            return 0

        linear_pred = np.dot(xv, self.w[xi])
        if self.task_type == "binary":
            p = 1.0 / (1.0 + np.exp(-np.clip(linear_pred[0], -50, 50)))
            return 1 if p > 0.5 else 0
        return int(np.argmax(linear_pred))

    def predict_proba(self, instance: Instance) -> np.ndarray:
        indices, values = self._get_sparse_x(instance)
        valid = indices < self.n_features
        xi, xv = indices[valid], values[valid]

        linear_pred = np.dot(xv, self.w[xi]) if len(xi) > 0 else np.zeros(self.n_outputs)

        if self.task_type == "binary":
            p = 1.0 / (1.0 + np.exp(-np.clip(linear_pred[0], -50, 50)))
            return np.array([1.0 - p, p])
        else:
            shift = linear_pred - np.max(linear_pred)
            exp_s = np.exp(shift)
            return exp_s / np.sum(exp_s)

    def get_sparsity(self) -> float:
        if self.w.size == 0:
            return 1.0
        return float(np.sum(np.abs(self.w) < 1e-10)) / self.w.size
