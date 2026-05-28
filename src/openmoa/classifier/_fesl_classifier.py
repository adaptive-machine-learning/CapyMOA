# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import numpy as np
from openmoa.base import Classifier, SparseInputMixin
from openmoa.stream import Schema
from openmoa.instance import Instance


class FESLClassifier(SparseInputMixin, Classifier):
    """FESL (Feature Evolvable Streaming Learning) – Hou et al., NeurIPS 2017.

    Logic
    -----
    1. Detects feature-space shift via Jaccard distance between consecutive
       active index sets.
    2. Maintains two linear models: ``w_curr`` (current space) and ``w_old``
       (previous space), stored as sparse dicts keyed by global feature ID.
    3. During the overlap buffer period, learns a linear mapping
       ``M : S_new → S_old`` via Ridge Regression.
    4. Final prediction is a weighted ensemble:
       ``y = μ_curr · f_curr(x) + μ_old · f_old(M · x)``.

    .. warning::
        Computing ``M`` requires a dense ``(D_new × D_old)`` matrix.  For
        UCI datasets (d < 5 000) this is fine.  For RCV1 (d ≈ 47 000) a
        ``MemoryError`` is expected and handled gracefully (mapping
        disabled, only ``w_curr`` is used).

    Only binary classification is supported.
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 0.1,
        lambda_: float = 0.1,
        window_size: int = 100,
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)

        if schema.get_num_classes() != 2:
            raise ValueError("FESLClassifier only supports binary classification.")

        self.alpha = alpha
        self.lambda_ = lambda_
        self.window_size = window_size

        # Instance-level RNG (no global seed pollution)
        self._rng = np.random.RandomState(random_seed)

        # Sparse weight dicts keyed by global feature ID
        self.w_curr: dict = {}
        self.w_old:  dict = {}

        # Mapping matrix struct (set after buffer fills)
        self.M_struct = None

        # Drift detection state
        self.current_indices_set: set = set()

        # Buffer of sparse dicts accumulated during overlap
        self.overlap_buffer: list = []

        # Ensemble weights
        self.mu_curr = 0.5
        self.mu_old  = 0.5

        self.t = 0

    def __str__(self):
        return f"FESLClassifier(alpha={self.alpha}, lambda={self.lambda_}, win={self.window_size})"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, instance: Instance):
        self.t += 1
        indices, values = self._get_sparse_x(instance)
        y = 1 if instance.y_index == 1 else -1

        # 1. Detect feature-space shift via Jaccard similarity
        new_indices_set = set(indices.tolist())

        if self.current_indices_set and new_indices_set != self.current_indices_set:
            inter = len(new_indices_set & self.current_indices_set)
            union = len(new_indices_set | self.current_indices_set)
            jaccard = inter / union if union > 0 else 0.0
            if jaccard < 0.8:
                self._transition_to_new_stage()
                self.current_indices_set = new_indices_set

        if not self.current_indices_set:
            self.current_indices_set = new_indices_set

        # 2. Buffer data during overlap for mapping
        if self.w_old and len(self.overlap_buffer) < self.window_size:
            self.overlap_buffer.append(dict(zip(indices.tolist(), values.tolist())))
            if len(self.overlap_buffer) == self.window_size:
                self._learn_mapping()

        # 3. Update current model
        pred_curr = self._predict_linear(self.w_curr, indices, values)
        self._update_weights(self.w_curr, indices, values, pred_curr, y)

        # 4. Update ensemble weights using numerically stable log-loss
        if self.M_struct is not None:
            pred_old = self._predict_via_mapping(indices, values)

            loss_curr = float(np.logaddexp(0, -y * pred_curr))
            loss_old  = float(np.logaddexp(0, -y * pred_old))

            eta = 0.1
            self.mu_curr *= np.exp(-eta * loss_curr)
            self.mu_old  *= np.exp(-eta * loss_old)

            total = self.mu_curr + self.mu_old
            if total > 1e-10:
                self.mu_curr /= total
                self.mu_old  /= total

    def predict(self, instance: Instance) -> int:
        return 1 if self.predict_proba(instance)[1] > 0.5 else 0

    def predict_proba(self, instance: Instance) -> np.ndarray:
        indices, values = self._get_sparse_x(instance)

        logit_curr = self._predict_linear(self.w_curr, indices, values)

        if self.M_struct is not None:
            logit_old   = self._predict_via_mapping(indices, values)
            final_logit = self.mu_curr * logit_curr + self.mu_old * logit_old
        else:
            final_logit = logit_curr

        prob = 1.0 / (1.0 + np.exp(-np.clip(final_logit, -50, 50)))
        return np.array([1.0 - prob, prob])

    # ------------------------------------------------------------------
    # Drift handling
    # ------------------------------------------------------------------

    def _transition_to_new_stage(self):
        """Promote current model to old; reset mapping state."""
        self.w_old = self.w_curr.copy()
        # Intentionally preserve w_curr: overlapping features inherit weights
        # (acts as transfer learning across feature spaces).
        self.M_struct = None
        self.overlap_buffer = []
        self.mu_curr = 0.5
        self.mu_old  = 0.5

    # ------------------------------------------------------------------
    # Mapping matrix
    # ------------------------------------------------------------------

    def _learn_mapping(self):
        """Learn M via Ridge Regression: X_old = X_new @ M.

        Solves: M = (X_new^T X_new + λI)^{-1} X_new^T X_old
        """
        if not self.overlap_buffer:
            return

        old_feat_ids = sorted(self.w_old.keys())
        if not old_feat_ids:
            return

        new_feat_ids = sorted({fid for x_dict in self.overlap_buffer for fid in x_dict})
        if not new_feat_ids:
            return

        B     = len(self.overlap_buffer)
        D_old = len(old_feat_ids)
        D_new = len(new_feat_ids)

        old_map = {fid: i for i, fid in enumerate(old_feat_ids)}
        new_map = {fid: i for i, fid in enumerate(new_feat_ids)}

        X_new = np.zeros((B, D_new))
        X_old = np.zeros((B, D_old))

        for i, x_dict in enumerate(self.overlap_buffer):
            for fid, val in x_dict.items():
                if fid in new_map:
                    X_new[i, new_map[fid]] = val
                if fid in old_map:
                    X_old[i, old_map[fid]] = val

        try:
            XtX = X_new.T @ X_new
            XtX[np.arange(D_new), np.arange(D_new)] += self.lambda_
            M_dense = np.linalg.solve(XtX, X_new.T @ X_old)

            self.M_struct = {
                "matrix":  M_dense,      # shape (D_new, D_old)
                "new_map": new_map,       # global ID → row index
                "old_ids": old_feat_ids,  # col index → global ID
            }

        except np.linalg.LinAlgError:
            self.M_struct = None
        except MemoryError:
            print(
                f"FESL: OOM during mapping (dims {D_new}×{D_old}). "
                "Feature-space transfer disabled."
            )
            self.M_struct = None

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_via_mapping(self, indices, values) -> float:
        """Compute logit via the old model after projecting through M.

        Vectorised: builds x_vec once, projects with a single dot, then
        computes w_old · x_rec with a pre-built weight array.
        """
        M       = self.M_struct["matrix"]
        new_map = self.M_struct["new_map"]
        old_ids = self.M_struct["old_ids"]

        # Build sparse input vector aligned to new_map
        x_vec = np.zeros(M.shape[0])
        for idx, val in zip(indices, values):
            col = new_map.get(int(idx))
            if col is not None:
                x_vec[col] = val

        # Project to old feature space: shape (D_old,)
        x_rec = x_vec @ M

        # Vectorised dot with w_old (pre-build weight array once per call)
        w_old_vec = np.fromiter(
            (self.w_old.get(gid, 0.0) for gid in old_ids),
            dtype=float, count=len(old_ids)
        )
        return float(w_old_vec @ x_rec)

    def _predict_linear(self, w_dict: dict, indices, values) -> float:
        return sum(w_dict.get(int(idx), 0.0) * val for idx, val in zip(indices, values))

    def _update_weights(self, w_dict: dict, indices, values, pred: float, y: int):
        p = 1.0 / (1.0 + np.exp(-np.clip(pred, -50, 50)))
        grad_scalar = p - (1 if y == 1 else 0)
        for idx, val in zip(indices, values):
            key = int(idx)
            w_dict[key] = w_dict.get(key, 0.0) - self.alpha * grad_scalar * val
