"""Kullback-Leibler divergence for data drift."""

import sys
from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy.special import rel_entr

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


def _bin_probabilities(
    x_ref: np.ndarray, x_test: np.ndarray, num_bins: int
) -> tuple:
    """Compute bin probabilities for reference and test samples.

    Bin edges span the combined range of both samples so that every
    observation falls inside a bin.

    :returns: ``(ref_probs, test_probs)`` each of shape ``(num_bins,)``.
    """
    combined = np.concatenate([x_ref, x_test])
    edges = np.linspace(combined.min(), combined.max(), num_bins + 1)
    ref_counts, _ = np.histogram(x_ref, bins=edges)
    test_counts, _ = np.histogram(x_test, bins=edges)
    # Replace zeros with the smallest representable float to avoid
    # log(0) and division by zero.
    ref_counts = ref_counts.astype(float)
    test_counts = test_counts.astype(float)
    ref_counts[ref_counts == 0] = sys.float_info.min
    test_counts[test_counts == 0] = sys.float_info.min
    ref_probs = ref_counts / ref_counts.sum()
    test_probs = test_counts / test_counts.sum()
    return ref_probs, test_probs


class KLDivergence(BaseDataDriftDetector):
    """Kullback-Leibler Divergence

    Computes KL(test || reference), i.e. the information lost when the
    reference distribution is used to approximate the test distribution.
    Drift is declared when the divergence exceeds ``threshold``.

    KL divergence is *not* symmetric: ``KL(P || Q) != KL(Q || P)``.
    For a symmetric alternative see :class:`JensenShannon`.

    Applied per feature, with Bonferroni correction by default.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import KLDivergence
    >>> rng = np.random.default_rng(42)
    >>> detector = KLDivergence(window_size=50, num_bins=20, threshold=0.1)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Kullback, Solomon, and Richard A. Leibler. "On information and sufficiency."
    The Annals of Mathematical Statistics 22.1 (1951): 79-86.

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
        """Create a KL divergence data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param num_bins: Number of histogram bins for probability
            estimation.
        :param threshold: Divergence above which drift is declared
            (per feature, before correction).
        :param correction: Multiple-testing correction across features.
            Ignored for the divergence comparison itself (which uses
            ``threshold``), but used to label per-feature results.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *num_bins* < 1 or *threshold* <= 0.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
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
        kl = float(np.sum(rel_entr(test_probs, ref_probs)))
        return DataDriftResult(
            is_drift=kl > self._threshold,
            statistic=kl,
            distance=kl,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "num_bins": self._num_bins,
            "threshold": self._threshold,
            "correction": self._correction,
        }
