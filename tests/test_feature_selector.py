"""Tests for :class:`capymoa.feature.TopKFeatureSelector` and ``Transformer.train``."""

import warnings

import numpy as np
import pytest

from capymoa.classifier import HoeffdingTree
from capymoa.core import Instance, LabeledInstance
from capymoa.datasets import ElectricityTiny, FriedTiny
from capymoa.feature import FeatureImportanceClassifier, TopKFeatureSelector
from capymoa.stream import Schema
from capymoa.stream.preprocessing import ClassifierPipeline, Transformer

SCHEMA = Schema.from_custom(
    features=["a", "b", "colour", "d", "class"],
    target="class",
    categories={"colour": ["red", "green"], "class": ["no", "yes"]},
    name="toy",
)


class FixedScorer(FeatureImportanceClassifier):
    """A scorer whose importances are fixed, counting its training calls."""

    def __init__(self, schema: Schema, importances: list[float]):
        super().__init__(schema=schema)
        self.importances = importances
        self.trained = 0

    def train(self, instance):
        self.trained += 1

    def predict(self, instance):
        return 0

    def predict_proba(self, instance):
        return np.array([1.0, 0.0])

    def get_feature_importances(self, normalize: bool = True) -> list[float]:
        return list(self.importances)


def _instance(x=(1.0, 2.0, 1.0, 4.0), y=1) -> LabeledInstance:
    return LabeledInstance.from_array(SCHEMA, np.array(x), y)


def _selector(importances, k=2, warmup=3) -> TopKFeatureSelector:
    return TopKFeatureSelector(
        SCHEMA, k=k, warmup=warmup, scorer=FixedScorer(SCHEMA, importances)
    )


def _warm_up(selector: TopKFeatureSelector) -> None:
    for _ in range(selector.warmup):
        selector.train(_instance())


def test_keeps_the_k_most_important_in_their_original_order():
    selector = _selector([0.1, 0.2, 0.0, 0.7], k=2)
    _warm_up(selector)
    assert selector.selected_features == ["b", "d"]
    out = selector.transform_instance(_instance())
    assert out.x.tolist() == [2.0, 4.0]


def test_selects_only_when_the_warmup_ends():
    selector = _selector([0.1, 0.2, 0.0, 0.7], warmup=3)
    selector.train(_instance())
    selector.train(_instance())
    assert not selector.is_selected
    assert selector.get_schema() is None
    assert selector.selected_features is None
    with pytest.raises(RuntimeError, match="2 so far"):
        selector.transform_instance(_instance())
    selector.train(_instance())
    assert selector.is_selected


def test_stops_training_the_scorer_once_selected():
    selector = _selector([0.1, 0.2, 0.0, 0.7], warmup=3)
    _warm_up(selector)
    selector.train(_instance())
    assert selector.scorer.trained == 3


def test_nan_scores_rank_as_zero_and_ties_keep_feature_order():
    # A tree that has not split reports NaN. Sorting raw NaNs is arbitrary.
    selector = _selector([float("nan"), 0.0, float("nan"), 0.5], k=2)
    with pytest.warns(UserWarning, match="Only 1 of 4 features"):
        _warm_up(selector)
    assert selector.selected_features == ["a", "d"]


def test_no_warning_when_k_features_are_scored():
    selector = _selector([0.1, 0.2, 0.0, 0.7], k=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warm_up(selector)


def test_output_schema_keeps_names_nominal_values_and_target():
    selector = _selector([0.0, 0.1, 0.9, 0.0], k=2)
    _warm_up(selector)
    schema = selector.get_schema()
    assert schema.get_num_attributes() == 2
    assert schema.get_nominal_attributes() == {"colour": ["red", "green"]}
    assert list(schema.get_label_values()) == ["no", "yes"]


def test_labels_are_kept_and_unlabeled_instances_stay_unlabeled():
    selector = _selector([0.1, 0.2, 0.0, 0.7], k=2)
    _warm_up(selector)
    labeled = selector.transform_instance(_instance(y=0))
    assert isinstance(labeled, LabeledInstance)
    assert labeled.y_index == 0
    unlabeled = selector.transform_instance(
        Instance.from_array(SCHEMA, np.array([1.0, 2.0, 1.0, 4.0]))
    )
    assert not isinstance(unlabeled, LabeledInstance)
    assert unlabeled.x.tolist() == [2.0, 4.0]


def test_restart_starts_a_new_warmup():
    selector = _selector([0.1, 0.2, 0.0, 0.7])
    _warm_up(selector)
    selector.restart()
    assert not selector.is_selected
    assert selector.get_schema() is None
    assert selector.instances_seen == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [({"k": 0}, "k must be between 1 and 4"), ({"k": 5}, "k must be between 1 and 4")]
    + [({"k": 1, "warmup": 0}, "warmup must be at least 1")],
)
def test_rejects_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TopKFeatureSelector(SCHEMA, scorer=FixedScorer(SCHEMA, [0] * 4), **kwargs)


def test_rejects_a_regression_schema():
    with pytest.raises(ValueError, match="classification schema"):
        TopKFeatureSelector(FriedTiny().get_schema(), k=2)


def test_default_scorer_runs_in_a_pipeline():
    stream = ElectricityTiny()
    selector = TopKFeatureSelector(stream.get_schema(), k=3, warmup=500, random_seed=1)
    for _ in range(500):
        selector.train(stream.next_instance())
    pipeline = (
        ClassifierPipeline()
        .add_transformer(selector)
        .add_classifier(HoeffdingTree(selector.get_schema()))
    )
    for _ in range(100):
        instance = stream.next_instance()
        pipeline.predict(instance)
        pipeline.train(instance)
    assert selector.instances_seen == 500


def test_pipeline_rejects_a_learner_built_on_the_full_schema():
    stream = ElectricityTiny()
    selector = _selector_for(stream)
    with pytest.raises(ValueError, match="expects a different schema"):
        ClassifierPipeline().add_transformer(selector).add_classifier(
            HoeffdingTree(stream.get_schema())
        )


def _selector_for(stream) -> TopKFeatureSelector:
    schema = stream.get_schema()
    selector = TopKFeatureSelector(
        schema, k=2, warmup=1, scorer=FixedScorer(schema, [1, 1, 0, 0, 0, 0])
    )
    selector.train(stream.next_instance())
    return selector


class RecordingTransformer(Transformer):
    """Passes instances through, recording each call."""

    def __init__(self, schema: Schema):
        self.schema = schema
        self.calls: list[str] = []

    def transform_instance(self, instance):
        self.calls.append("transform")
        return instance

    def train(self, instance):
        self.calls.append("train")

    def get_schema(self):
        return self.schema

    def restart(self):
        pass


def test_pipeline_trains_transformers_only_on_the_train_path():
    stream = ElectricityTiny()
    recorder = RecordingTransformer(stream.get_schema())
    pipeline = (
        ClassifierPipeline()
        .add_transformer(recorder)
        .add_classifier(HoeffdingTree(stream.get_schema()))
    )
    instance = stream.next_instance()
    pipeline.predict(instance)
    assert recorder.calls == ["transform"]
    pipeline.train(instance)
    # Transform first, so the output for an instance never uses its own label.
    assert recorder.calls == ["transform", "transform", "train"]


def test_transformer_train_does_nothing_by_default():
    class PassThrough(Transformer):
        def transform_instance(self, instance):
            return instance

        def get_schema(self):
            return SCHEMA

        def restart(self):
            pass

    assert PassThrough().train(_instance()) is None
