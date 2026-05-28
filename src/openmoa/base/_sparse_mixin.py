# SPDX-License-Identifier: BSD-3-Clause
"""Mixin providing sparse feature extraction for UOL classifiers.

All UOL classifiers that handle OpenFeatureStream (varying feature spaces)
or NaN-padded streams (TrapezoidalStream, CapriciousStream, EvolvableStream)
inherit this mixin to avoid duplicating the same extraction logic.
"""
from __future__ import annotations
import numpy as np


class SparseInputMixin:
    """Mixin that provides ``_get_sparse_x`` for sparse/dynamic feature handling.

    Handles three input formats in priority order:

    1. **OpenFeatureStream** – instance carries ``feature_indices`` (global IDs)
       and a physically smaller ``x`` array.
    2. **Native sparse** – instance exposes ``x_index`` / ``x_value`` (e.g. LibSVM).
    3. **Dense / NaN-padded** – standard numpy array where 0s and NaNs are
       treated as absent features (covers TrapezoidalStream, CapriciousStream,
       EvolvableStream).
    """

    def _get_sparse_x(self, instance) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, values)`` for *instance*.

        :returns: A pair of 1-D numpy arrays ``(indices, values)`` where
            ``indices`` holds global feature IDs and ``values`` holds the
            corresponding feature values.  Both arrays have the same length.
        """
        # Case 1: OpenFeatureStream – global IDs attached directly
        if hasattr(instance, "feature_indices"):
            return np.asarray(instance.feature_indices), np.asarray(instance.x, dtype=float)

        # Case 2: native sparse instance (e.g. from ARFF / LibSVM loader)
        if hasattr(instance, "x_index") and hasattr(instance, "x_value"):
            return instance.x_index, instance.x_value

        # Case 3: dense array, possibly NaN-padded by wrapper streams
        x = np.asarray(instance.x, dtype=float)
        valid_mask = (x != 0) & (~np.isnan(x))
        indices = np.where(valid_mask)[0]
        return indices, x[indices]
