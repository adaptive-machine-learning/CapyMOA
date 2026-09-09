"""Regression.

Regression predicts continuous target values. In data stream learning,
regressors must be updated incrementally from a single pass over the data
and adapt their predictions as the underlying target function drifts.
"""

from ._adaptive_random_forest import AdaptiveRandomForestRegressor
from ._arffimtdd import ARFFIMTDD
from ._fading_target_mean import FadingTargetMean
from ._fimtdd import FIMTDD
from ._knn import KNNRegressor
from ._no_change import NoChange
from ._orto import ORTO
from ._passive_aggressive_regressor import PassiveAggressiveRegressor
from ._sgbr import StreamingGradientBoostedRegression
from ._sgd_regressor import SGDRegressor
from ._sgt import StochasticGradientTree
from ._shrubs_regressor import ShrubsRegressor
from ._soknl import SOKNL
from ._soknl_base_tree import SOKNLBT
from ._target_mean import TargetMean

__all__ = [
    "ARFFIMTDD",
    "FIMTDD",
    "ORTO",
    "SOKNL",
    "SOKNLBT",
    "AdaptiveRandomForestRegressor",
    "FadingTargetMean",
    "KNNRegressor",
    "NoChange",
    "PassiveAggressiveRegressor",
    "SGDRegressor",
    "ShrubsRegressor",
    "StochasticGradientTree",
    "StreamingGradientBoostedRegression",
    "TargetMean",
]
