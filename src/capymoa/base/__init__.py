from capymoa._optional import lazy_torch_attrs
from capymoa.base._base import (
    AnomalyDetector,
    Clusterer,
    ClusteringResult,
    MOAAnomalyDetector,
    MOAClusterer,
    MOAPredictionIntervalLearner,
    PredictionIntervalLearner,
)
from capymoa.base._classifier import (
    Classifier,
    MOAClassifier,
    SKClassifier,
)
from capymoa.base._regressor import MOARegressor, Regressor, SKRegressor
from capymoa.base._ssl import (
    ClassifierSSL,
    MOAClassifierSSL,
)

__all__ = [
    "Classifier",
    "Batch",
    "BatchClassifier",
    "MOAClassifier",
    "SKClassifier",
    "ClassifierSSL",
    "MOAClassifierSSL",
    "Regressor",
    "BatchRegressor",
    "MOARegressor",
    "SKRegressor",
    "AnomalyDetector",
    "Clusterer",
    "ClusteringResult",
    "MOAAnomalyDetector",
    "MOAClusterer",
    "MOAPredictionIntervalLearner",
    "PredictionIntervalLearner",
]

#: Names that need PyTorch. Imported on first access so ``import capymoa`` stays
#: torch-free -- see :mod:`capymoa._optional`.
_LAZY = {
    "Batch": "._batch",
    "BatchClassifier": "._batch_classifier",
    "BatchRegressor": "._batch_regressor",
}

__getattr__, __dir__ = lazy_torch_attrs(
    __name__, _LAZY, "the Batch* base classes", __all__
)
