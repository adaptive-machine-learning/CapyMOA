"""Classification.

Classification assigns discrete labels to instances. In data stream learning,
classifiers must learn incrementally from a single pass over the data,
adapting to concept drift while making predictions in real time.
"""

from capymoa._optional import lazy_torch_attrs

from ._adaptive_random_forest import AdaptiveRandomForestClassifier
from ._csmote import CSMOTE
from ._dems import DynamicEnsembleMemberSelection
from ._dynamic_weighted_majority import DynamicWeightedMajority
from ._efdt import EFDT

# `_hoeffding_tree` must be imported before `_hoeffding_adaptive_tree`: the
# latter does `from capymoa.classifier import HoeffdingTree`, which requires
# this module's own re-export to already be bound. isort would normally
# alphabetize these the other way, reintroducing that circular import,
# hence `isort: skip`.
from ._hoeffding_tree import HoeffdingTree  # isort: skip
from ._hoeffding_adaptive_tree import HoeffdingAdaptiveTree
from ._knn import KNN
from ._last import LAST
from ._leveraging_bagging import LeveragingBagging
from ._majority_class import MajorityClass
from ._naive_bayes import NaiveBayes
from ._no_change import NoChange
from ._online_adwin_bagging import OnlineAdwinBagging
from ._online_bagging import OnlineBagging
from ._online_smooth_boost import OnlineSmoothBoost
from ._oza_boost import OzaBoost
from ._passive_aggressive_classifier import PassiveAggressiveClassifier
from ._plastic import PLASTIC
from ._samknn import SAMkNN
from ._sgbt import StreamingGradientBoostedTrees
from ._sgd_classifier import SGDClassifier
from ._sgt import StochasticGradientTree
from ._shrubs_classifier import ShrubsClassifier
from ._srp import StreamingRandomPatches
from ._weightedknn import WeightedkNN

__all__ = [
    "CSMOTE",
    "EFDT",
    "KNN",
    "LAST",
    "PLASTIC",
    "AdaptiveRandomForestClassifier",
    "DynamicEnsembleMemberSelection",
    "DynamicWeightedMajority",
    "Finetune",
    "HoeffdingAdaptiveTree",
    "HoeffdingTree",
    "LeveragingBagging",
    "MajorityClass",
    "NaiveBayes",
    "NoChange",
    "OnlineAdwinBagging",
    "OnlineBagging",
    "OnlineSmoothBoost",
    "OzaBoost",
    "PassiveAggressiveClassifier",
    "SAMkNN",
    "SGDClassifier",
    "ShrubsClassifier",
    "StochasticGradientTree",
    "StreamingGradientBoostedTrees",
    "StreamingRandomPatches",
    "WeightedkNN",
]


#: Names that need PyTorch. Imported on first access so ``import capymoa`` stays
#: torch-free -- see :mod:`capymoa._optional`.
_LAZY = {
    "Finetune": "._finetune",
}

__getattr__, __dir__ = lazy_torch_attrs(__name__, _LAZY, "Finetune", __all__)
