"""Kolmogorov-Smirnov two-sample test for data drift."""

from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy.stats import ks_2samp

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


class KolmogorovSmirnov(BaseDataDriftDetector):
    """Kolmogorov-Smirnov Two-Sample Test

    Univariate statistical test that compares the empirical CDFs of the
    reference and test samples. Large differences in the CDF indicate
    that the two samples come from different distributions.

    Applied per feature, with Bonferroni correction by default.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import KolmogorovSmirnov
    >>> rng = np.random.default_rng(42)
    >>> detector = KolmogorovSmirnov(window_size=50)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Massey Jr, F. J. "The Kolmogorov-Smirnov test for goodness of fit."
    Journal of the American Statistical Association 46.253 (1951): 68-78.

    """

    IS_UNIVARIATE = True

    def __init__(
        self,
        window_size: int,
        alpha: float = 0.05,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: Optional[int] = None,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        method: Literal["auto", "exact", "approx", "asymp"] = "auto",
    ):
        """Create a Kolmogorov-Smirnov data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param alpha: Significance level for drift decisions.
        :param correction: Multiple-testing correction across features.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :param alternative: Alternative hypothesis for the KS test.
            ``"two-sided"`` (default), ``"less"``, or ``"greater"``.
        :param method: Method for computing the p-value. See
            :func:`scipy.stats.ks_2samp` for details.
        """
        super().__init__(window_size, alpha, correction, auto_fit_samples)
        self._alternative = alternative
        self._method = method

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X

    def _test(self, x_ref: np.ndarray, x_test: np.ndarray) -> DataDriftResult:
        stat, p = ks_2samp(
            x_ref,
            x_test,
            alternative=self._alternative,
            method=self._method,
        )
        return DataDriftResult(is_drift=False, statistic=stat, p_value=p)

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "alpha": self._alpha,
            "correction": self._correction,
            "alternative": self._alternative,
            "method": self._method,
        }
