import numpy as np
import pytest

from capymoa.base import Classifier, MOAClassifier
from capymoa.classifier import HoeffdingTree
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


class DummyPythonLearner:
    pass


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


def test_feature_importance_classifier_validates_window_size():
    stream = ElectricityTiny()

    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        DummyFeatureImportanceClassifier(
            schema=stream.get_schema(),
            window_size=0,
        )


def test_moa_feature_importance_classifier_wraps_real_moa_learner():
    stream = ElectricityTiny()
    learner = MOAFeatureImportanceClassifier(
        schema=stream.get_schema(),
        base_learner=HoeffdingTree(schema=stream.get_schema(), random_seed=1),
        random_seed=1,
    )

    assert isinstance(learner, MOAClassifier)
    assert str(learner.moa_learner.getClass().getSimpleName()) == (
        "FeatureImportanceHoeffdingTree"
    )


def test_moa_feature_importance_classifier_rejects_invalid_python_object():
    stream = ElectricityTiny()

    with pytest.raises(TypeError, match="base_learner must be"):
        MOAFeatureImportanceClassifier(
            schema=stream.get_schema(),
            base_learner=DummyPythonLearner(),
            random_seed=1,
        )


def test_moa_feature_importance_classifier_rejects_invalid_python_class():
    stream = ElectricityTiny()

    with pytest.raises(TypeError, match="base_learner must be"):
        MOAFeatureImportanceClassifier(
            schema=stream.get_schema(),
            base_learner=DummyPythonLearner,
            random_seed=1,
        )
