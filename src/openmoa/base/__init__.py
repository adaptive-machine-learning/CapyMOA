# SPDX-License-Identifier: BSD-3-Clause
# Pure-Python base classes – no Java required at import time.
from openmoa.base._classifier import (
    BatchClassifier,
    Classifier,
    MOAClassifier,
    SKClassifier,
)
from openmoa.base._sparse_mixin import SparseInputMixin
from openmoa.base._regressor import BatchRegressor, MOARegressor, Regressor, SKRegressor

# MOA-dependent base classes are loaded lazily so that
# ``import openmoa.base`` (and thus ``from openmoa.base import Classifier``)
# does not require Java.
_MOA_BASE_ATTRS = {
    "AnomalyDetector",
    "Clusterer",
    "ClusteringResult",
    "MOAAnomalyDetector",
    "MOAClusterer",
    "MOAPredictionIntervalLearner",
    "PredictionIntervalLearner",
    "_extract_moa_drift_detector_CLI",
    "_extract_moa_learner_CLI",
    # _ssl.py imports _base.Instance which requires moa at module level
    "ClassifierSSL",
    "MOAClassifierSSL",
}


def __getattr__(name: str):
    if name in _MOA_BASE_ATTRS:
        from openmoa._prepare_jpype import _start_jpype

        _start_jpype()
        from openmoa.base._base import (
            AnomalyDetector,
            Clusterer,
            ClusteringResult,
            MOAAnomalyDetector,
            MOAClusterer,
            MOAPredictionIntervalLearner,
            PredictionIntervalLearner,
            _extract_moa_drift_detector_CLI,
            _extract_moa_learner_CLI,
        )
        from openmoa.base._ssl import ClassifierSSL, MOAClassifierSSL
        # Populate globals so subsequent accesses are direct (no __getattr__ overhead).
        g = globals()
        g["AnomalyDetector"] = AnomalyDetector
        g["Clusterer"] = Clusterer
        g["ClusteringResult"] = ClusteringResult
        g["MOAAnomalyDetector"] = MOAAnomalyDetector
        g["MOAClusterer"] = MOAClusterer
        g["MOAPredictionIntervalLearner"] = MOAPredictionIntervalLearner
        g["PredictionIntervalLearner"] = PredictionIntervalLearner
        g["_extract_moa_drift_detector_CLI"] = _extract_moa_drift_detector_CLI
        g["_extract_moa_learner_CLI"] = _extract_moa_learner_CLI
        g["ClassifierSSL"] = ClassifierSSL
        g["MOAClassifierSSL"] = MOAClassifierSSL
        return g[name]
    raise AttributeError(f"module 'openmoa.base' has no attribute {name!r}")


__all__ = [
    "_extract_moa_drift_detector_CLI",
    "_extract_moa_learner_CLI",
    "Classifier",
    "BatchClassifier",
    "MOAClassifier",
    "SKClassifier",
    "ClassifierSSL",
    "MOAClassifierSSL",
    "SparseInputMixin",
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
