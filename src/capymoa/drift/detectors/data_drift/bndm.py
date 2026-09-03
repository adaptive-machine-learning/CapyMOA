"""Bayesian Nonparametric Detection Method (BNDM) for data drift."""

from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy import stats
from scipy.special import betaln

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


def _normalize(data: np.ndarray) -> np.ndarray:
    """Normalize by subtracting the mean and dividing by the IQR."""
    centered = data - np.mean(data)
    iqr = stats.iqr(data)
    if iqr != 0:
        centered = centered / iqr
    return centered


def _polya_tree_test(
    sample_one: np.ndarray,
    sample_two: np.ndarray,
    const: float,
    max_depth: int,
) -> float:
    """Run the Pólya tree two-sample test.

    :returns: Log-odds ratio; large positive values indicate the two
        samples come from the same distribution.
    """
    norm = stats.norm(loc=0, scale=1)

    def _recurse(level: int, partition: str) -> float:
        if level > max_depth:
            return 0.0

        p_left = partition + "0"
        p_right = partition + "1"

        n1_l, n2_l = _interval_count(sample_one, sample_two, p_left, norm)
        n1_r, n2_r = _interval_count(sample_one, sample_two, p_right, norm)

        if (n1_l + n1_r) == 0 or (n2_l + n2_r) == 0:
            return 0.0

        n_l = n1_l + n2_l
        n_r = n1_r + n2_r
        alpha = const * (level + 1) ** 2

        num = -betaln(alpha, alpha) + betaln(alpha + n_l, alpha + n_r)
        den = (
            -2.0 * betaln(alpha, alpha)
            + betaln(alpha + n1_l, alpha + n1_r)
            + betaln(alpha + n2_l, alpha + n2_r)
        )
        contribution = num - den
        return (
            contribution
            + _recurse(level + 1, p_left)
            + _recurse(level + 1, p_right)
        )

    return _recurse(0, "")


def _interval_count(
    s1: np.ndarray, s2: np.ndarray, partition: str, dist: stats.rv_continuous
) -> tuple:
    """Count samples falling in the partition's interval."""
    idx = int(partition, 2)
    level = len(partition)
    q_lo = idx / (2**level)
    q_hi = (idx + 1) / (2**level)
    lo, hi = dist.ppf([q_lo, q_hi])
    n1 = int(np.sum((s1 > lo) & (s1 <= hi)))
    n2 = int(np.sum((s2 > lo) & (s2 <= hi)))
    return n1, n2


class BNDM(BaseDataDriftDetector):
    """Bayesian Nonparametric Detection Method

    Uses a Pólya tree two-sample test to determine whether reference
    and test data come from the same distribution. The data is
    partitioned recursively using normal-distribution percentiles,
    and the partitions are compared using the Beta function.

    Because the Pólya tree test is univariate, it is applied per
    feature with Bonferroni correction by default.

    .. note::
        Data is normalized internally (mean-centered, scaled by IQR).
        Best suited for sudden changes; may struggle with subtle drift.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import BNDM
    >>> rng = np.random.default_rng(42)
    >>> detector = BNDM(window_size=100, threshold=0.5)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(100, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Xuan, J., Lu, J., and Zhang, G. "Bayesian nonparametric unsupervised
    concept drift detection for data stream mining." ACM Transactions on
    Intelligent Systems and Technology (2020).

    """

    IS_UNIVARIATE = True

    def __init__(
        self,
        window_size: int,
        const: float = 1.0,
        threshold: float = 0.5,
        max_depth: int = 3,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: Optional[int] = None,
    ):
        """Create a BNDM data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param const: Constant that scales the Pólya tree concentration
            parameters. Larger values give more weight to the prior.
        :param threshold: Similarity below which drift is declared
            (per feature). Must be in ``(0, 1)``.
        :param max_depth: Maximum depth of the Pólya tree recursion.
        :param correction: Multiple-testing correction across features.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *threshold* not in ``(0, 1)`` or
            *max_depth* < 1.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        super().__init__(
            window_size, alpha=0.05, correction=correction,
            auto_fit_samples=auto_fit_samples,
        )
        self._const = const
        self._threshold = threshold
        self._max_depth = max_depth

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X

    def _test(self, x_ref: np.ndarray, x_test: np.ndarray) -> DataDriftResult:
        combined = np.concatenate([x_ref, x_test])
        normalized = _normalize(combined)
        ref_norm = normalized[: len(x_ref)]
        test_norm = normalized[len(x_ref) :]

        log_odds = _polya_tree_test(
            ref_norm, test_norm, self._const, self._max_depth,
        )
        similarity = 1.0 / (1.0 + np.exp(-log_odds))

        return DataDriftResult(
            is_drift=similarity < self._threshold,
            statistic=similarity,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "const": self._const,
            "threshold": self._threshold,
            "max_depth": self._max_depth,
            "correction": self._correction,
        }
