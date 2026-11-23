"""Find abstentions in the Python loop, the predictions a learner could not make.

A classifier abstains when its confidence is at or below chance: every label has
the same probability (all-zero votes count too). A regressor abstains when it
returns ``NaN``. MOA counts them, the same as in the Java loop.
"""

import numpy as np


def proba_abstains(proba, num_classes: int) -> bool:
    """Whether a classifier's confidence is equal to or below chance.

    This is true when every label has the same probability, as in MOA. Empty,
    zero and non finite votes also count. Votes are padded with zeros to
    ``num_classes``.

    >>> from capymoa.evaluation._abstain import proba_abstains
    >>> proba_abstains([0.5, 0.5], 2), proba_abstains([0.6, 0.4], 2)
    (True, False)
    >>> proba_abstains([0, 0, 0], 3), proba_abstains([2.0, 2.0], 3)
    (True, False)
    """
    p = np.asarray(proba, dtype=np.float64).ravel()
    if p.size < num_classes:
        p = np.pad(p, (0, num_classes - p.size))
    total = p.sum()
    if p.size == 0 or not np.isfinite(total) or total <= 0.0:
        return True
    return bool(np.all(p == p[0]))


def batch_proba_abstains(proba: np.ndarray) -> np.ndarray:
    """Which rows of a ``(batch, num_classes)`` probability array abstain."""
    proba = np.asarray(proba, dtype=np.float64)
    total = proba.sum(axis=1)
    equal = (proba == proba[:, :1]).all(axis=1)
    return ~np.isfinite(total) | (total <= 0.0) | equal
