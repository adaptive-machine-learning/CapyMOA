"""Base classes for CapyMOA learners.

This module defines the abstract interfaces that all CapyMOA learners implement,
including classifiers, regressors, anomaly detectors, clusterers, prediction
interval learners, and their semi-supervised and MOA-backed variants.
"""

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
from capymoa.base._learner_params import (
    LearnerParamsMixin,
    LearnerSpec,
    learner_from_params,
)
from capymoa.base._regressor import MOARegressor, Regressor, SKRegressor
from capymoa.base._ssl import (
    ClassifierSSL,
    MOAClassifierSSL,
)

__all__ = [
    "AnomalyDetector",
    "Batch",
    "BatchClassifier",
    "BatchRegressor",
    "Classifier",
    "ClassifierSSL",
    "Clusterer",
    "ClusteringResult",
    "LearnerParamsMixin",
    "LearnerSpec",
    "MOAAnomalyDetector",
    "MOAClassifier",
    "MOAClassifierSSL",
    "MOAClusterer",
    "MOAPredictionIntervalLearner",
    "MOARegressor",
    "PredictionIntervalLearner",
    "Regressor",
    "SKClassifier",
    "SKRegressor",
    "learner_from_params",
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
