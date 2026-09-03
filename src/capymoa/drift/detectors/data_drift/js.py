"""Jensen-Shannon distance for data drift."""

import sys
from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


def _bin_probabilities(
    x_ref: np.ndarray, x_test: np.ndarray, num_bins: int
) -> tuple:
    """Compute bin probabilities for reference and test samples.

    :returns: ``(ref_probs, test_probs)`` each of shape ``(num_bins,)``.
    """
    combined = np.concatenate([x_ref, x_test])
    edges = np.linspace(combined.min(), combined.max(), num_bins + 1)
    ref_counts, _ = np.histogram(x_ref, bins=edges)
    test_counts, _ = np.histogram(x_test, bins=edges)
    ref_counts = ref_counts.astype(float)
    test_counts = test_counts.astype(float)
    ref_counts[ref_counts == 0] = sys.float_info.min
    test_counts[test_counts == 0] = sys.float_info.min
    ref_probs = ref_counts / ref_counts.sum()
    test_probs = test_counts / test_counts.sum()
    return ref_probs, test_probs


class JensenShannon(BaseDataDriftDetector):
    """Jensen-Shannon Distance

    Computes the Jensen-Shannon distance (the square root of the
    Jensen-Shannon divergence) between reference and test distributions.
    Unlike KL divergence, JS distance is symmetric and bounded in
    ``[0, 1]``.

    Drift is declared when the distance exceeds ``threshold``.
    Applied per feature, with Bonferroni correction by default.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import JensenShannon
    >>> rng = np.random.default_rng(42)
    >>> detector = JensenShannon(window_size=50, num_bins=20, threshold=0.1)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Lin, Jianhua. "Divergence measures based on the Shannon entropy."
    IEEE Transactions on Information Theory 37.1 (1991): 145-151.

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
        """Create a Jensen-Shannon data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param num_bins: Number of histogram bins for probability
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
        ref_probs, test_probs = _bin_probabilities(x_ref, x_test, self._num_bins)
        js_dist = float(jensenshannon(ref_probs, test_probs))
        return DataDriftResult(
            is_drift=js_dist > self._threshold,
            statistic=js_dist,
            distance=js_dist,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "num_bins": self._num_bins,
            "threshold": self._threshold,
            "correction": self._correction,
        }
