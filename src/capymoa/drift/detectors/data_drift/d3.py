"""Discriminative Drift Detector (D3) for data drift."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


class D3(BaseDataDriftDetector):
    """Discriminative Drift Detector (D3)

    Detects drift by training a classifier to distinguish reference
    samples from test samples. If the classifier achieves a high
    ROC-AUC, the two distributions must differ, indicating drift.

    This is a fundamentally different approach from statistical tests
    or distance measures: it frames drift detection as a binary
    classification problem.

    Stratified k-fold cross-validation is used so that every sample
    receives a prediction. Drift is declared when the ROC-AUC exceeds
    ``threshold``.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import D3
    >>> rng = np.random.default_rng(42)
    >>> detector = D3(window_size=50, threshold=0.7, seed=42)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Gözüak, Ö., Büyükçakir, A., Bonab, H., and Can, F. "Unsupervised
    concept drift detection with a discriminative classifier." Proceedings
    of the 28th ACM International Conference on Information and Knowledge
    Management (2019). ACM.

    """

    IS_UNIVARIATE = False

    def __init__(
        self,
        window_size: int,
        threshold: float = 0.7,
        n_splits: int = 2,
        seed: Optional[int] = None,
        auto_fit_samples: Optional[int] = None,
    ):
        """Create a D3 data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param threshold: ROC-AUC above which drift is declared. Values
            near 0.5 mean the classifier cannot distinguish the windows
            (no drift); values near 1.0 mean clear separation (drift).
        :param n_splits: Number of cross-validation folds.
        :param seed: Random seed for the classifier and CV splits.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *threshold* not in ``(0.5, 1]`` or
            *n_splits* < 2.
        """
        if not 0.5 < threshold <= 1.0:
            raise ValueError("threshold must be in (0.5, 1]")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        super().__init__(
            window_size, alpha=0.05, correction="none",
            auto_fit_samples=auto_fit_samples,
        )
        self._threshold = threshold
        self._n_splits = n_splits
        self._seed = seed

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X

    def _test(self, X_ref: np.ndarray, X_test: np.ndarray) -> DataDriftResult:
        n_ref, n_test = len(X_ref), len(X_test)
        X = np.vstack([X_ref, X_test])
        labels = np.concatenate([np.zeros(n_ref), np.ones(n_test)])

        kfold = StratifiedKFold(
            n_splits=self._n_splits, shuffle=True, random_state=self._seed,
        )
        clf = LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=self._seed,
        )

        predictions = np.zeros(len(X))
        for train_idx, test_idx in kfold.split(X, labels):
            clf.fit(X[train_idx], labels[train_idx])
            predictions[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

        auc = float(roc_auc_score(labels, predictions))

        return DataDriftResult(
            is_drift=auc >= self._threshold,
            statistic=auc,
            distance=auc,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "threshold": self._threshold,
            "n_splits": self._n_splits,
            "seed": self._seed,
        }
