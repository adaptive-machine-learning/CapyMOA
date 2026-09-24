"""Anderson-Darling k-sample test for data drift."""

import warnings
from typing import Any, Literal

import numpy as np
from scipy.stats import anderson_ksamp

from .base import BaseDataDriftDetector, DataDriftResult

# scipy.stats.anderson_ksamp floors p-values at this value. With Bonferroni
# correction, once alpha / n_features falls at or below this floor, no
# p-value can ever be small enough to declare drift.
_PVALUE_FLOOR = 0.001


class AndersonDarling(BaseDataDriftDetector):
    """Anderson-Darling K-Sample Test

    Univariate statistical test that compares two samples by measuring
    the integrated squared difference between their empirical CDFs,
    weighted to give more sensitivity in the tails than the
    Kolmogorov-Smirnov test.

    Applied per feature, with Bonferroni correction by default.

    .. note::
        p-values are bounded between 0.001 and 0.25 by
        :func:`scipy.stats.anderson_ksamp`. With ``correction="bonferroni"``,
        :meth:`fit` raises :class:`ValueError` if ``alpha / n_features``
        falls at or below the 0.001 floor, since drift could then never be
        declared regardless of the data.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import AndersonDarling
    >>> rng = np.random.default_rng(42)
    >>> detector = AndersonDarling(window_size=50)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(0.8, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Scholz, Fritz W., and Michael A. Stephens. "K-sample Anderson-Darling tests."
    Journal of the American Statistical Association 82.399 (1987): 918-924.

    """

    IS_UNIVARIATE = True

    def __init__(
        self,
        window_size: int,
        alpha: float = 0.05,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: int | None = None,
    ):
        """Create an Anderson-Darling data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param alpha: Significance level for drift decisions.
        :param correction: Multiple-testing correction across features.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        """
        super().__init__(window_size, alpha, correction, auto_fit_samples)

    def _fit(self, X: np.ndarray) -> None:
        n_features = X.shape[1]
        if self._correction == "bonferroni":
            effective_alpha = self._alpha / n_features
            if effective_alpha <= _PVALUE_FLOOR:
                raise ValueError(
                    f"alpha / n_features = {effective_alpha} is at or below "
                    f"scipy.stats.anderson_ksamp's p-value floor "
                    f"({_PVALUE_FLOOR}); drift could never be detected with "
                    "this configuration. Use correction='none', increase "
                    "alpha, or reduce the number of features."
                )
        self._X_ref = X

    def _test(self, x_ref: np.ndarray, x_test: np.ndarray) -> DataDriftResult:
        with warnings.catch_warnings():
            # SciPy warns when the p-value is floored at 0.001; this
            # is expected behaviour, not a coding error.
            warnings.filterwarnings("ignore", message="p-value floored")
            warnings.filterwarnings("ignore", message="p-value capped")
            try:
                # `variant` replaces the older `midrank` boolean parameter
                # in newer SciPy; older SciPy only accepts `midrank`.
                result = anderson_ksamp([x_ref, x_test], variant="midrank")
            except TypeError:
                result = anderson_ksamp([x_ref, x_test], midrank=True)
        return DataDriftResult(
            is_drift=False,
            statistic=result.statistic,
            p_value=result.pvalue,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "window_size": self._window_size,
            "alpha": self._alpha,
            "correction": self._correction,
            "auto_fit_samples": self._auto_fit_samples,
        }
