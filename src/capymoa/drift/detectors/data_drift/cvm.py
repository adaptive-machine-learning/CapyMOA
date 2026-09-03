"""Cramér-von Mises two-sample test for data drift."""

from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy.stats import cramervonmises_2samp

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


class CramerVonMises(BaseDataDriftDetector):
    """Cramér-von Mises Two-Sample Test

    Measures the integral of the squared difference between the
    empirical CDFs of two samples. While KS captures the *maximum*
    CDF gap, CVM integrates the *total* squared gap, making it more
    sensitive to subtle, spread-out differences.

    Applied per feature, with Bonferroni correction by default.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import CramerVonMises
    >>> rng = np.random.default_rng(42)
    >>> detector = CramerVonMises(window_size=50)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Cramér, Harald. "On the composition of elementary errors."
    Scandinavian Actuarial Journal 1928.1 (1928): 13-74.

    """

    IS_UNIVARIATE = True

    def __init__(
        self,
        window_size: int,
        alpha: float = 0.05,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: Optional[int] = None,
        method: Literal["auto", "asymptotic", "exact"] = "auto",
    ):
        """Create a Cramér-von Mises data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param alpha: Significance level for drift decisions.
        :param correction: Multiple-testing correction across features.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :param method: Method for computing the p-value. See
            :func:`scipy.stats.cramervonmises_2samp` for details.
        """
        super().__init__(window_size, alpha, correction, auto_fit_samples)
        self._method = method

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X

    def _test(self, x_ref: np.ndarray, x_test: np.ndarray) -> DataDriftResult:
        result = cramervonmises_2samp(x_ref, x_test, method=self._method)
        return DataDriftResult(
            is_drift=False, statistic=result.statistic, p_value=result.pvalue
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "alpha": self._alpha,
            "correction": self._correction,
            "method": self._method,
        }
