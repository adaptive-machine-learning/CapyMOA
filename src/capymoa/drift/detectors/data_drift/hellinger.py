"""Hellinger distance for data drift."""

import sys
from typing import Any, Dict, Literal, Optional

import numpy as np

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult

_SQRT2 = np.sqrt(2.0)


class Hellinger(BaseDataDriftDetector):
    """Hellinger Distance

    Measures the similarity between two probability distributions by
    comparing binned proportions. The Hellinger distance is symmetric
    and bounded in ``[0, 1]``, where 0 means the distributions are
    identical and 1 means they have no overlap.

    Related to the Bhattacharyya coefficient:
    ``H = sqrt(1 - BC)`` where ``BC = sum(sqrt(p * q))``.

    Drift is declared when the distance exceeds ``threshold``.
    Applied per feature, with Bonferroni correction by default.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import Hellinger
    >>> rng = np.random.default_rng(42)
    >>> detector = Hellinger(window_size=50, num_bins=20, threshold=0.1)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Hellinger, Ernst. "Neue Begründung der Theorie quadratischer Formen von
    unendlichvielen Veränderlichen." Journal für die reine und angewandte
    Mathematik 136 (1909): 210-271.

    """

    IS_UNIVARIATE = True

    def __init__(
        self,
        window_size: int,
        num_bins: int = 10,
        threshold: float = 0.1,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: Optional[int] = None,
    ):
        """Create a Hellinger data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param num_bins: Number of equal-width bins for proportion
            estimation.
        :param threshold: Distance above which drift is declared
            (per feature). Must be in ``(0, 1]``.
        :param correction: Multiple-testing correction across features.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *num_bins* < 1 or *threshold* not in
            ``(0, 1]``.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        super().__init__(
            window_size, alpha=0.05, correction=correction,
            auto_fit_samples=auto_fit_samples,
        )
        self._num_bins = num_bins
        self._threshold = threshold

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X

    def _test(self, x_ref: np.ndarray, x_test: np.ndarray) -> DataDriftResult:
        combined = np.concatenate([x_ref, x_test])
        edges = np.linspace(combined.min(), combined.max(), self._num_bins + 1)
        ref_counts, _ = np.histogram(x_ref, bins=edges)
        test_counts, _ = np.histogram(x_test, bins=edges)

        ref_pct = ref_counts.astype(float) / max(ref_counts.sum(), 1)
        test_pct = test_counts.astype(float) / max(test_counts.sum(), 1)

        # Replace zeros to avoid sqrt issues in edge cases
        ref_pct[ref_pct == 0] = sys.float_info.min
        test_pct[test_pct == 0] = sys.float_info.min

        dist = float(
            np.sqrt(np.sum((np.sqrt(ref_pct) - np.sqrt(test_pct)) ** 2)) / _SQRT2
        )
        return DataDriftResult(
            is_drift=dist > self._threshold,
            statistic=dist,
            distance=dist,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "num_bins": self._num_bins,
            "threshold": self._threshold,
            "correction": self._correction,
        }
