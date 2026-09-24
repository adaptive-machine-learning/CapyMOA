"""Kullback-Leibler divergence for data drift."""

from typing import Any, Literal

import numpy as np
from scipy.special import rel_entr

from .base import BaseDataDriftDetector, DataDriftResult, _bin_probabilities


class KLDivergence(BaseDataDriftDetector):
    """Kullback-Leibler Divergence

    Computes KL(test || reference), i.e. the information lost when the
    reference distribution is used to approximate the test distribution.
    Drift is declared when the divergence exceeds ``threshold``.

    KL divergence is *not* symmetric: ``KL(P || Q) != KL(Q || P)``.
    For a symmetric alternative see :class:`JensenShannon`.

    Applied independently to each feature using the configured threshold.
    Overall drift is reported when any feature flags drift. This detector
    does not return p-values, and the ``correction`` parameter has no
    effect on its drift decisions. No Bonferroni correction is applied.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import KLDivergence
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
        auto_fit_samples: int | None = None,
    ):
        """Create a KL divergence data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param num_bins: Number of histogram bins for probability
            estimation.
        :param threshold: Divergence above which drift is declared
            (per feature).
        :param correction: Accepted for interface consistency with other
            detectors, but has no effect: this detector does not produce
            p-values, so no multiple-testing correction is applied to its
            drift decisions.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *num_bins* < 1 or *threshold* <= 0.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        super().__init__(
            window_size,
            alpha=0.05,
            correction=correction,
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

    def get_params(self) -> dict[str, Any]:
        return {
            "window_size": self._window_size,
            "num_bins": self._num_bins,
            "threshold": self._threshold,
            "correction": self._correction,
            "auto_fit_samples": self._auto_fit_samples,
        }
