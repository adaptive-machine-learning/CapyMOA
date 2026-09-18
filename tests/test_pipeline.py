"""Tests for pipelines, pipeline elements, and schema propagation through them.

Before these, `src/capymoa/stream/preprocessing/` had no pytest coverage at all --
the only thing exercising it was `notebooks/07_pipelines.ipynb`, which nbmake runs
without checking output. See adaptive-machine-learning/backlog#87 and #154.
"""

import pytest
from moa.streams.filters import (
    AddNoiseFilter,
    HashingTrickFilter,
    NormalisationFilter,
)

from capymoa.anomaly import HalfSpaceTrees
from capymoa.classifier import OnlineBagging
from capymoa.datasets import ElectricityTiny, FriedTiny
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import ClassificationEvaluator, prequential_evaluation
from capymoa.regressor import AdaptiveRandomForestRegressor
from capymoa.stream.preprocessing import (
    BasePipeline,
    ClassifierPipeline,
    ClassifierPipelineElement,
    DriftDetectorPipelineElement,
    MOATransformer,
    RegressorPipeline,
    TransformerPipelineElement,
)


@pytest.fixture
def elec():
    return ElectricityTiny()


def _normaliser(schema):
    return MOATransformer(schema=schema, moa_filter=NormalisationFilter())


def _hasher(schema, dimension):
    """A transformer that genuinely changes the feature set.

    The hashing trick is the only filter reachable from a CapyMOA pipeline that
    does: `SelectAttributesFilter` would be the natural choice but is not a
    `moa.streams.filters.StreamFilter`, so `FilteredQueueStream` rejects it.
    """
    return MOATransformer(
        schema=schema, moa_filter=HashingTrickFilter(), CLI=f"-d {dimension}"
    )


# ---------------------------------------------------------------- schema comparison


def test_schema_is_compatible_with_itself(elec):
    schema = elec.get_schema()
    assert schema.is_compatible_with(schema)
    assert schema.describe_difference(schema) == []


def test_schema_detects_incompatible_stream(elec):
    """A classification and a regression schema must not be interchangeable."""
    elec_schema = elec.get_schema()
    fried_schema = FriedTiny().get_schema()

    assert not elec_schema.is_compatible_with(fried_schema)
    differences = elec_schema.describe_difference(fried_schema)
    assert differences, "expected describe_difference to explain the mismatch"
    assert any("number of attributes" in d for d in differences)
    assert any("task" in d for d in differences)


def test_schema_compatibility_ignores_dataset_name(elec):
    """Two reads of the same dataset are interchangeable for a learner."""
    assert elec.get_schema().is_compatible_with(ElectricityTiny().get_schema())


# ---------------------------------------------------------------- element schemas


def test_transformer_element_reports_transformer_schema(elec):
    transformer = _normaliser(elec.get_schema())
    element = TransformerPipelineElement(transformer)
    assert element.get_schema() is transformer.get_schema()


def test_classifier_element_reports_learner_schema(elec):
    learner = OnlineBagging(schema=elec.get_schema(), ensemble_size=3)
    element = ClassifierPipelineElement(learner)
    assert element.get_schema() is elec.get_schema()


def test_drift_detector_element_has_no_schema():
    """A drift detector neither knows nor alters the schema."""
    element = DriftDetectorPipelineElement(ADWIN(), lambda instance, prediction: 0)
    assert element.get_schema() is None


# ---------------------------------------------------------------- pipeline schemas


def test_empty_pipeline_has_no_schema():
    assert BasePipeline().get_schema() is None


def test_pipeline_reports_last_known_schema(elec):
    transformer = _normaliser(elec.get_schema())
    pipeline = BasePipeline().add_transformer(transformer)
    assert pipeline.get_schema() is transformer.get_schema()


def test_drift_detector_does_not_hide_upstream_schema(elec):
    """An element without a schema must not mask the one before it."""
    transformer = _normaliser(elec.get_schema())
    pipeline = BasePipeline().add_transformer(transformer)
    pipeline.add_drift_detector(ADWIN(), lambda instance, prediction: 0)
    assert pipeline.get_schema() is transformer.get_schema()


def test_nested_pipeline_reports_inner_schema(elec):
    transformer = _normaliser(elec.get_schema())
    inner = BasePipeline().add_transformer(transformer)
    outer = BasePipeline().add_pipeline_element(inner)
    assert outer.get_schema() is transformer.get_schema()


def test_explicit_input_schema_is_used_for_an_empty_pipeline(elec):
    pipeline = BasePipeline(schema=elec.get_schema())
    assert pipeline.get_input_schema() is elec.get_schema()
    assert pipeline.get_schema() is elec.get_schema()


def test_declared_input_schema_validates_the_first_element(elec):
    """Declaring the input schema lets the very first element be checked too."""
    pipeline = BasePipeline(schema=FriedTiny().get_schema())
    with pytest.raises(ValueError, match="different schema"):
        pipeline.add_transformer(_normaliser(elec.get_schema()))


def test_input_and_output_schema_differ_across_a_reducing_transformer(elec):
    """The pipeline consumes 6 attributes and emits 3; both must be reported."""
    transformer = _hasher(elec.get_schema(), dimension=3)
    pipeline = BasePipeline().add_transformer(transformer)
    pipeline.pass_forward(elec.next_instance())

    assert pipeline.get_input_schema().get_num_attributes() == 6
    assert pipeline.get_schema().get_num_attributes() == 3


# ---------------------------------------------- learner interface conformance


def test_classifier_pipeline_satisfies_classifier_contract(elec):
    """`Classifier` declares `schema` and `random_seed`; a pipeline must have both.

    Before #154 a ClassifierPipeline raised AttributeError on `.schema`, because
    no `__init__` in the MRO ever called `Classifier.__init__`.
    """
    learner = OnlineBagging(schema=elec.get_schema(), ensemble_size=3)
    pipeline = ClassifierPipeline().add_classifier(learner)

    assert pipeline.schema is elec.get_schema()
    assert isinstance(pipeline.random_seed, int)


def test_regressor_pipeline_satisfies_regressor_contract():
    stream = FriedTiny()
    learner = AdaptiveRandomForestRegressor(schema=stream.get_schema(), ensemble_size=3)
    pipeline = RegressorPipeline().add_regressor(learner)

    assert pipeline.schema is stream.get_schema()
    assert isinstance(pipeline.random_seed, int)


def test_pipeline_schema_tracks_elements_added_later(elec):
    """`schema` is a property, so it must reflect elements added after construction."""
    pipeline = ClassifierPipeline()
    assert pipeline.schema is None
    pipeline.add_classifier(OnlineBagging(schema=elec.get_schema(), ensemble_size=3))
    assert pipeline.schema is elec.get_schema()


# ---------------------------------------------------------------- validation


def test_adding_incompatible_learner_raises(elec):
    """A learner built on a different stream must be rejected, not silently accepted."""
    pipeline = RegressorPipeline().add_transformer(_normaliser(elec.get_schema()))
    mismatched = AdaptiveRandomForestRegressor(
        schema=FriedTiny().get_schema(), ensemble_size=3
    )

    with pytest.raises(ValueError, match="different schema"):
        pipeline.add_regressor(mismatched)


def test_constructor_validates_its_elements_too(elec):
    """Elements passed to the constructor get the same check as appended ones."""
    with pytest.raises(ValueError, match="different schema"):
        BasePipeline(
            [
                TransformerPipelineElement(_normaliser(elec.get_schema())),
                ClassifierPipelineElement(
                    AdaptiveRandomForestRegressor(
                        schema=FriedTiny().get_schema(), ensemble_size=3
                    )
                ),
            ]
        )


def test_validation_can_be_disabled(elec):
    pipeline = RegressorPipeline(validate_schema=False)
    pipeline.add_transformer(_normaliser(elec.get_schema()))
    pipeline.add_regressor(
        AdaptiveRandomForestRegressor(schema=FriedTiny().get_schema(), ensemble_size=3)
    )
    assert len(pipeline.elements) == 2


def test_compatible_learner_is_accepted(elec):
    """Normalisation preserves the attribute set, so the learner still fits."""
    pipeline = ClassifierPipeline().add_transformer(_normaliser(elec.get_schema()))
    pipeline.add_classifier(OnlineBagging(schema=elec.get_schema(), ensemble_size=3))
    assert len(pipeline.elements) == 2


def test_anomaly_detector_is_still_rejected_as_a_classifier(elec):
    """Anomaly support is workstream 2 of #154; until then the guard must hold."""
    detector = HalfSpaceTrees(schema=elec.get_schema())
    with pytest.raises(AssertionError):
        ClassifierPipeline().add_classifier(detector)


# ---------------------------------------------------- transformers and feature sets


def test_transformer_output_schema_for_feature_preserving_filter(elec):
    transformer = _normaliser(elec.get_schema())
    assert transformer.get_schema().is_compatible_with(elec.get_schema())
    assert transformer.get_input_schema() is elec.get_schema()


def test_transformer_reports_reduced_feature_set(elec):
    """A filter that changes the feature set must report the *new* output schema.

    This is the case the pipeline could not express before #154: get_schema()
    returned the input schema, so the next element was told 6 attributes while
    the transformer actually emitted 3.
    """
    in_schema = elec.get_schema()
    transformer = _hasher(in_schema, dimension=3)
    transformed = transformer.transform_instance(elec.next_instance())

    assert len(transformed.x) == 3, "the filter really does emit 3 features"
    assert transformer.get_input_schema().get_num_attributes() == 6
    assert transformer.get_schema().get_num_attributes() == 3
    assert not transformer.get_schema().is_compatible_with(in_schema)


def test_chained_transformers_report_final_schema(elec):
    first = _normaliser(elec.get_schema())
    second = MOATransformer(schema=first.get_schema(), moa_filter=AddNoiseFilter())
    pipeline = BasePipeline().add_transformer(first).add_transformer(second)
    assert pipeline.get_schema() is second.get_schema()


# ---------------------------------------------------------------- behaviour


def test_str_has_no_trailing_separator(elec):
    pipeline = ClassifierPipeline()
    pipeline.add_transformer(_normaliser(elec.get_schema()))
    pipeline.add_classifier(OnlineBagging(schema=elec.get_schema(), ensemble_size=3))

    rendered = str(pipeline)
    assert not rendered.endswith(" | ")
    assert rendered.count(" | ") == 1


def test_elements_are_applied_in_order(elec):
    """pass_forward must transform before it trains, not after."""
    seen = []

    class Recorder(TransformerPipelineElement):
        def pass_forward(self, instance):
            seen.append("transform")
            return instance

    class Trainer(ClassifierPipelineElement):
        def pass_forward(self, instance):
            seen.append("train")
            return instance

    pipeline = BasePipeline(
        [
            Recorder(_normaliser(elec.get_schema())),
            Trainer(OnlineBagging(schema=elec.get_schema(), ensemble_size=3)),
        ]
    )
    pipeline.pass_forward(elec.next_instance())
    assert seen == ["transform", "train"]


def test_pipeline_matches_equivalent_manual_loop(elec):
    """A ClassifierPipeline must score exactly as the hand-written loop it replaces."""
    stream_a = ElectricityTiny()
    learner = OnlineBagging(schema=stream_a.get_schema(), ensemble_size=3)
    evaluator = ClassificationEvaluator(schema=stream_a.get_schema())
    while stream_a.has_more_instances():
        instance = stream_a.next_instance()
        evaluator.update(instance.y_index, learner.predict(instance))
        learner.train(instance)
    expected = evaluator.accuracy()

    stream_b = ElectricityTiny()
    pipeline = ClassifierPipeline().add_classifier(
        OnlineBagging(schema=stream_b.get_schema(), ensemble_size=3)
    )
    results = prequential_evaluation(stream_b, pipeline, optimise=False)

    assert results["cumulative"].accuracy() == pytest.approx(expected)


def test_transformer_pipeline_matches_equivalent_manual_loop():
    """A pipeline with a transformer must score as the hand-written loop it replaces.

    Guards the schema swap in MOATransformer.transform_instance: adopting the
    derived output schema must not change the instances the learner sees.
    """
    stream_a = ElectricityTiny()
    transformer_a = _normaliser(stream_a.get_schema())
    learner_a = OnlineBagging(schema=transformer_a.get_schema(), ensemble_size=3)
    evaluator = ClassificationEvaluator(schema=stream_a.get_schema())
    while stream_a.has_more_instances():
        instance = stream_a.next_instance()
        transformed = transformer_a.transform_instance(instance)
        evaluator.update(instance.y_index, learner_a.predict(transformed))
        learner_a.train(transformed)
    expected = evaluator.accuracy()

    stream_b = ElectricityTiny()
    transformer_b = _normaliser(stream_b.get_schema())
    pipeline = (
        ClassifierPipeline()
        .add_transformer(transformer_b)
        .add_classifier(
            OnlineBagging(schema=transformer_b.get_schema(), ensemble_size=3)
        )
    )
    results = prequential_evaluation(stream_b, pipeline, optimise=False)

    assert results["cumulative"].accuracy() == pytest.approx(expected)
