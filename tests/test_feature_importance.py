import numpy as np

from capymoa.base import Classifier, MOAClassifier
from capymoa.datasets import ElectricityTiny
from capymoa.feature_selection import (
    FeatureImportanceClassifier,
    MOAFeatureImportanceClassifier,
)


class DummyFeatureImportanceClassifier(FeatureImportanceClassifier):
    def train(self, instance) -> None:
        self._on_train_complete()

    def predict_proba(self, instance):
        return np.array([1.0, 0.0])

    def get_feature_importances(self, normalize: bool = True) -> list[float]:
        return [0.2, 0.7, 0.1]


def test_feature_importance_classifier_is_generic_base():
    stream = ElectricityTiny()
    learner = DummyFeatureImportanceClassifier(
        schema=stream.get_schema(),
        window_size=2,
    )

    assert isinstance(learner, Classifier)
    assert not isinstance(learner, MOAClassifier)

    learner.train(next(stream))
    learner.train(next(stream))

    assert learner.get_top_k_features(2) == [1, 0]
    assert learner.get_windowed_feature_importances() == [
        {"instances_seen": 2, "importances": [0.2, 0.7, 0.1]}
    ]


def test_moa_feature_importance_classifier_has_expected_hierarchy():
    assert issubclass(MOAFeatureImportanceClassifier, FeatureImportanceClassifier)
    assert issubclass(MOAFeatureImportanceClassifier, MOAClassifier)
