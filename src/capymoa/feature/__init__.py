"""Feature selection.

Feature selection identifies the subset of features that are most relevant
to the learning task. In data stream learning, feature relevance can change
over time, so importance must be estimated and updated incrementally.
"""

from ._feature_importance import (
    FeatureImportanceClassifier,
    MOAFeatureImportanceClassifier,
)
from .visualization import (
    plot_feature_importance,
    plot_windowed_feature_importance,
)

__all__ = [
    "FeatureImportanceClassifier",
    "MOAFeatureImportanceClassifier",
    "plot_feature_importance",
    "plot_windowed_feature_importance",
]
