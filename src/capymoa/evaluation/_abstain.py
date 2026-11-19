"""Count abstentions, the predictions a learner could not make.

A classifier abstains when its confidence is at or below chance: every label has
the same probability (all-zero votes count too). A regressor abstains when it
returns ``NaN``.
"""

from typing import Any

import jpype
import numpy as np


def proba_abstains(proba, num_classes: int) -> bool:
    """Whether a classifier's confidence is equal to or below chance.

    The largest probability is never below ``1 / num_classes``, so this is true
    when all the probabilities are equal. Empty, zero and non finite votes also
    count. Votes are scaled to sum to 1 and padded with zeros to ``num_classes``.

    >>> from capymoa.evaluation._abstain import proba_abstains
    >>> proba_abstains([0.5, 0.5], 2), proba_abstains([0.6, 0.4], 2)
    (True, False)
    >>> proba_abstains([0, 0, 0], 3), proba_abstains([2.0, 2.0], 3)
    (True, False)
    """
    p = np.asarray(proba, dtype=np.float64).ravel()
    total = p.sum()
    if p.size == 0 or not np.isfinite(total) or total <= 0.0:
        return True
    return bool(p.max() / total <= 1.0 / num_classes + 1e-9)


def batch_proba_abstains(proba: np.ndarray) -> np.ndarray:
    """Which rows of a ``(batch, num_classes)`` probability array abstain."""
    proba = np.asarray(proba, dtype=np.float64)
    chance = 1.0 / proba.shape[1]
    total = proba.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        top = proba.max(axis=1) / total
    return ~np.isfinite(total) | (total <= 0.0) | (top <= chance + 1e-6)


class AbstentionHook:
    """A MOA evaluator that counts abstentions as the Java loop calls it.

    Pass :attr:`proxy` to the Java loop in place of ``evaluator``. Each call is
    passed on to ``evaluator``. This is the only Python code the Java loop runs.

    :param evaluator: The MOA evaluator to wrap.
    :param window_size: Count abstentions in windows of this many instances.
    :param num_classes: Number of labels for a classifier, or ``None`` for a
        regressor.
    :param nan_value: A regressor's replacement for ``NaN``. ``None`` keeps it.
    """

    def __init__(
        self,
        evaluator: Any,
        window_size: int,
        num_classes: int | None,
        nan_value: float | None = None,
    ):
        from moa.evaluation import LearningPerformanceEvaluator

        self.evaluator = evaluator
        self.window_size = window_size
        self.num_classes = num_classes
        self.nan_value = nan_value
        self.total = 0
        self.windows: list[int] = []
        self._seen = 0
        self._window = 0
        self.proxy = jpype.JProxy(LearningPerformanceEvaluator, inst=self)

    def addResult(self, example, votes_or_prediction):
        if hasattr(votes_or_prediction, "__len__"):
            if self.num_classes is None:
                votes_or_prediction = self._regression_votes(votes_or_prediction)
            elif proba_abstains(votes_or_prediction, self.num_classes):
                self._count()
        self._seen += 1
        if self._seen % self.window_size == 0:
            self.windows.append(self._window)
            self._window = 0
        return self.evaluator.addResult(example, votes_or_prediction)

    def _count(self) -> None:
        self.total += 1
        self._window += 1

    def _regression_votes(self, votes):
        if len(votes) == 0:
            self._count()
            if self.nan_value is None:
                return votes
            return jpype.JArray(jpype.JDouble)([self.nan_value])
        if votes[0] != votes[0]:  # NaN
            self._count()
            if self.nan_value is not None:
                votes[0] = self.nan_value  # In place, so the windowed one sees it
        return votes

    def window_counts(self, rows: int) -> list[int]:
        """The count of each of ``rows`` windows, with the last partial window."""
        counts = [*self.windows, self._window] if self._window else list(self.windows)
        return (counts + [0] * rows)[:rows]

    def __getattr__(self, name: str):
        return getattr(self.evaluator, name)
