"""Automatic machine learning.

Automatic machine learning (AutoML) automates the selection and tuning of
learners. In data stream learning, AutoML must make these choices online,
continually re-evaluating and adapting to changing data characteristics.
"""

from ._autoclass import AutoClass
from ._bandit_classifier import BanditClassifier, EpsilonGreedy
from ._successive_halving_classifier import SuccessiveHalvingClassifier

__all__ = [
    "AutoClass",
    "BanditClassifier",
    "SuccessiveHalvingClassifier",
    "EpsilonGreedy",
]
