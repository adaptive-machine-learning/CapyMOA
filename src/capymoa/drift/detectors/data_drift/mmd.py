"""Maximum Mean Discrepancy (MMD) for data drift."""

from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.spatial.distance import cdist

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Radial basis function (Gaussian) kernel.

    :param X: Array of shape ``(n, d)``.
    :param Y: Array of shape ``(m, d)``.
    :param sigma: Bandwidth parameter.
    :returns: Kernel matrix of shape ``(n, m)``.
    """
    return np.exp(-cdist(X, Y, "sqeuclidean") / (2.0 * sigma**2))


def _mmd2_from_matrices(
    K_XX: np.ndarray, K_YY: np.ndarray, K_XY: np.ndarray
) -> float:
    """Compute the unbiased MMD^2 estimate from pre-computed kernel matrices."""
    n = K_XX.shape[0]
    m = K_YY.shape[0]
    # Exclude the diagonal for the unbiased estimator
    k_xx = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
    k_yy = (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
    k_xy = K_XY.sum() / (n * m)
    return float(k_xx + k_yy - 2.0 * k_xy)


class MMD(BaseDataDriftDetector):
    """Maximum Mean Discrepancy

    MMD is a multivariate, kernel-based distance between two
    distributions. A permutation test is used to obtain a p-value.

    By default, the RBF kernel with ``sigma=1.0`` is used.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import MMD
    >>> rng = np.random.default_rng(42)
    >>> detector = MMD(window_size=50, n_permutations=100, sigma=1.0)
    >>> detector.fit(rng.normal(0, 1, size=(100, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Gretton, Arthur, et al. "A kernel two-sample test."
    The Journal of Machine Learning Research 13.1 (2012): 723-773.

    """

    IS_UNIVARIATE = False

    def __init__(
        self,
        window_size: int,
        alpha: float = 0.05,
        sigma: float = 1.0,
        kernel: Optional[Callable] = None,
        n_permutations: int = 100,
        auto_fit_samples: Optional[int] = None,
    ):
        """Create an MMD data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param alpha: Significance level for drift decisions.
        :param sigma: Bandwidth for the default RBF kernel. Ignored when
            a custom *kernel* is provided.
        :param kernel: Callable ``(X, Y) -> K`` that returns an
            ``(n, m)`` kernel matrix. If ``None``, an RBF kernel with
            the given *sigma* is used.
        :param n_permutations: Number of permutations for the
            significance test. Higher values give more accurate p-values
            at the cost of computation. Must be at least 1.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *n_permutations* < 1.
        """
        if n_permutations < 1:
            raise ValueError("n_permutations must be at least 1")
        super().__init__(
            window_size, alpha=alpha, correction="none",
            auto_fit_samples=auto_fit_samples,
        )
        self._sigma = sigma
        if kernel is not None:
            self._kernel = kernel
        else:
            self._kernel = lambda X, Y: rbf_kernel(X, Y, sigma=sigma)
        self._n_permutations = n_permutations
        # Pre-computed reference kernel sum (set in _fit)
        self._k_xx_mean: Optional[float] = None

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X
        K_XX = self._kernel(X, X)
        n = X.shape[0]
        self._k_xx_mean = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))

    def _test(self, X_ref: np.ndarray, X_test: np.ndarray) -> DataDriftResult:
        # Observed MMD
        K_XX = self._kernel(X_ref, X_ref)
        K_YY = self._kernel(X_test, X_test)
        K_XY = self._kernel(X_ref, X_test)
        observed = _mmd2_from_matrices(K_XX, K_YY, K_XY)

        # Permutation test: compute full kernel matrix once, then
        # re-index for each permutation to avoid redundant computation.
        combined = np.vstack([X_ref, X_test])
        n = X_ref.shape[0]
        K_full = self._kernel(combined, combined)

        rng = np.random.default_rng()
        count = 0
        total = len(combined)
        for _ in range(self._n_permutations):
            perm = rng.permutation(total)
            idx_a, idx_b = perm[:n], perm[n:]
            k_aa = K_full[np.ix_(idx_a, idx_a)]
            k_bb = K_full[np.ix_(idx_b, idx_b)]
            k_ab = K_full[np.ix_(idx_a, idx_b)]
            mmd2_perm = _mmd2_from_matrices(k_aa, k_bb, k_ab)
            if mmd2_perm >= observed:
                count += 1

        # +1 in numerator and denominator for conservative estimate
        p_value = (count + 1) / (self._n_permutations + 1)

        return DataDriftResult(
            is_drift=p_value < self._alpha,
            statistic=observed,
            p_value=p_value,
            distance=observed,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "alpha": self._alpha,
            "sigma": self._sigma,
            "n_permutations": self._n_permutations,
        }
