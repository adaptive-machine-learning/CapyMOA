# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import numpy as np
from typing import Literal

from openmoa.base import Classifier, SparseInputMixin
from openmoa.stream import Schema
from openmoa.instance import Instance


class FOBOSClassifier(SparseInputMixin, Classifier):
    """Forward-Backward Splitting (FOBOS) for online classification.

    Supports binary (logistic) and multi-class (softmax) tasks.
    Weight matrix auto-expands to accommodate dynamic feature indices from
    wrapper streams (e.g. :class:`OpenFeatureStream` incremental mode).

    Reference
    ---------
    Duchi, J., & Singer, Y. (2009). Efficient Online and Batch Learning
    Using Forward Backward Splitting. JMLR.
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 1.0,
        lambda_: float = 0.001,
        regularization: Literal["l1", "l2", "l1_l2", "none"] = "l1",
        step_schedule: Literal["sqrt", "linear"] = "sqrt",
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)

        self.alpha          = alpha
        self.lambda_        = lambda_
        self.regularization = regularization
        self.step_schedule  = step_schedule

        self.n_classes = schema.get_num_classes()
        if self.n_classes == 2:
            self.task_type = "binary"
            self.n_outputs = 1
        else:
            self.task_type = "multiclass"
            self.n_outputs = self.n_classes

        self.n_features = schema.get_num_attributes()

        # Instance-level RNG (no global seed pollution)
        self._rng = np.random.RandomState(random_seed)
        scale = 1.0 / np.sqrt(self.n_features) if self.n_features > 0 else 0.01
        self.W = self._rng.randn(self.n_features, self.n_outputs) * scale

        self.t = 0

    def __str__(self):
        return (f"FOBOSClassifier(task={self.task_type}, alpha={self.alpha}, "
                f"lambda={self.lambda_}, reg={self.regularization})")

    # ------------------------------------------------------------------
    # Dynamic dimension (A1 fix)
    # ------------------------------------------------------------------

    def _ensure_dimension(self, target_dim: int):
        """Expand W if feature indices from a dynamic stream exceed current size."""
        if target_dim <= self.n_features:
            return
        new_dim = max(target_dim, int(self.n_features * 1.5))
        new_W   = np.zeros((new_dim, self.n_outputs))
        new_W[:self.n_features] = self.W
        self.W          = new_W
        self.n_features = new_dim

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, instance: Instance):
        self.t += 1
        x_indices, x_values = self._get_sparse_x(instance)

        if len(x_indices) == 0:
            return

        # Expand if needed (handles OpenFeatureStream incremental mode)
        max_idx = int(np.max(x_indices))
        self._ensure_dimension(max_idx + 1)

        eta = self._get_learning_rate()

        if self.task_type == "binary":
            score    = np.dot(self.W[x_indices, 0], x_values)
            pred     = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))
            grad_s   = pred - instance.y_index
            self.W[x_indices, 0] -= eta * grad_s * x_values

        else:  # multiclass
            scores      = self.W[x_indices].T @ x_values
            scores     -= np.max(scores)
            exp_s       = np.exp(scores)
            probs       = exp_s / np.sum(exp_s)
            err         = probs.copy()
            err[instance.y_index] -= 1.0
            self.W[x_indices] -= eta * np.outer(x_values, err)

        if self.lambda_ > 0:
            self._apply_proximal_operator(eta)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, instance: Instance) -> int:
        x_indices, x_values = self._get_sparse_x(instance)
        if len(x_indices) == 0:
            return 0
        # Clamp indices to current W size
        valid = x_indices < self.n_features
        xi, xv = x_indices[valid], x_values[valid]

        if self.task_type == "binary":
            return 1 if np.dot(self.W[xi, 0], xv) > 0 else 0
        else:
            return int(np.argmax(self.W[xi].T @ xv))

    def predict_proba(self, instance: Instance) -> np.ndarray:
        x_indices, x_values = self._get_sparse_x(instance)
        valid = x_indices < self.n_features
        xi, xv = x_indices[valid], x_values[valid]

        if self.task_type == "binary":
            score = np.dot(self.W[xi, 0], xv) if len(xi) > 0 else 0.0
            p     = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))
            return np.array([1.0 - p, p])
        else:
            scores = self.W[xi].T @ xv if len(xi) > 0 else np.zeros(self.n_outputs)
            scores -= np.max(scores)
            exp_s   = np.exp(scores)
            return exp_s / np.sum(exp_s)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_learning_rate(self) -> float:
        if self.step_schedule == "sqrt":
            return self.alpha / np.sqrt(self.t)
        elif self.step_schedule == "linear":
            return self.alpha / self.t
        return self.alpha

    def _apply_proximal_operator(self, eta: float):
        threshold = eta * self.lambda_

        if self.regularization == "l1":
            self.W = np.sign(self.W) * np.maximum(0.0, np.abs(self.W) - threshold)

        elif self.regularization == "l2":
            # Clamp to avoid sign flip when threshold >= 1
            self.W *= max(0.0, 1.0 - threshold)

        elif self.regularization == "l1_l2":
            row_norms = np.linalg.norm(self.W, axis=1, keepdims=True)
            safe_norms = np.where(row_norms == 0, 1.0, row_norms)
            shrinkage  = np.maximum(0.0, 1.0 - threshold / safe_norms)
            self.W    *= shrinkage
