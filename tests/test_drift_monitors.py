"""Tests for the ready-made drift detector inputs in `capymoa.drift.monitors`.

See adaptive-machine-learning/backlog#158. The point of these helpers is that a
misplaced drift detector used to fail silently: an accuracy monitor placed
before the learner receives `prediction=None` forever, scores every instance 0,
and the detector never fires.
"""

import warnings

import pytest
from moa.streams.filters import NormalisationFilter

from capymoa.classifier import OnlineBagging
from capymoa.datasets import ElectricityTiny, FriedTiny
from capymoa.drift.detectors import ADWIN
from capymoa.drift.monitors import absolute_error, feature_value, prediction_is_correct
from capymoa.regressor import AdaptiveRandomForestRegressor
from capymoa.stream.preprocessing import (
    ClassifierPipeline,
    MOATransformer,
    RegressorPipeline,
)


def _feed(monitor, instance, prediction, times):
    """Call `monitor` repeatedly; a single statement for `pytest.warns` to wrap."""
    for _ in range(times):
        monitor(instance, prediction)


def _run(pipeline, stream, instances):
    """Drive a pipeline for `instances` steps; one statement for `pytest.warns`."""
    for _ in range(instances):
        instance = stream.next_instance()
        pipeline.predict(instance)
        pipeline.train(instance)


# ---------------------------------------------------------------- values


def test_prediction_is_correct_scores_a_hit_and_a_miss():
    instance = ElectricityTiny().next_instance()
    monitor = prediction_is_correct()

    assert monitor(instance, instance.y_index) == 1
    assert monitor(instance, 1 - instance.y_index) == 0


def test_prediction_is_correct_counts_none_as_incorrect():
    """Matches what the hand-written `int(label == prediction)` already did."""
    instance = ElectricityTiny().next_instance()
    assert prediction_is_correct()(instance, None) == 0


def test_absolute_error_measures_distance():
    instance = FriedTiny().next_instance()
    monitor = absolute_error()

    assert monitor(instance, instance.y_value) == 0.0
    assert monitor(instance, instance.y_value + 2.5) == pytest.approx(2.5)
    assert monitor(instance, instance.y_value - 2.5) == pytest.approx(2.5)


def test_absolute_error_counts_none_as_zero():
    instance = FriedTiny().next_instance()
    assert absolute_error()(instance, None) == 0.0


def test_feature_value_reads_the_requested_feature():
    instance = ElectricityTiny().next_instance()
    for index in range(len(instance.x)):
        assert feature_value(index)(instance, None) == float(instance.x[index])


def test_feature_value_ignores_the_prediction():
    instance = ElectricityTiny().next_instance()
    monitor = feature_value(0)
    assert monitor(instance, None) == monitor(instance, 1) == monitor(instance, 0)


# ---------------------------------------------- the misplaced-detector warning


def test_persistent_none_warns_once_with_an_actionable_message():
    instance = ElectricityTiny().next_instance()
    monitor = prediction_is_correct(warn_after=3)

    with pytest.warns(UserWarning, match="before the learner") as caught:
        _feed(monitor, instance, None, 20)

    assert len(caught) == 1, "the warning must fire once, not once per instance"


def test_warm_up_nones_do_not_warn():
    """A learner returns None until it can predict; that is not a misplacement."""
    instance = ElectricityTiny().next_instance()
    monitor = prediction_is_correct(warn_after=3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _feed(monitor, instance, None, 2)
        monitor(instance, instance.y_index)
        _feed(monitor, instance, None, 2)


def test_absolute_error_warns_on_persistent_none():
    instance = FriedTiny().next_instance()
    monitor = absolute_error(warn_after=3)

    with pytest.warns(UserWarning, match="before the learner"):
        _feed(monitor, instance, None, 10)


def test_a_detector_placed_before_the_learner_now_warns():
    """The silent failure that motivated backlog#158, end to end."""
    stream = ElectricityTiny()
    detector = ADWIN()
    pipeline = (
        ClassifierPipeline()
        .add_drift_detector(detector, prediction_is_correct(warn_after=10))
        .add_classifier(OnlineBagging(schema=stream.get_schema(), ensemble_size=3))
    )

    with pytest.warns(UserWarning, match="before the learner"):
        _run(pipeline, stream, 50)

    assert detector.detection_index == [], (
        "the detector still sees a constant stream; the warning is the fix, "
        "not the scoring"
    )


def test_a_correctly_placed_detector_does_not_warn():
    stream = ElectricityTiny()
    pipeline = (
        ClassifierPipeline()
        .add_classifier(OnlineBagging(schema=stream.get_schema(), ensemble_size=3))
        .add_drift_detector(ADWIN(), prediction_is_correct(warn_after=10))
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _run(pipeline, stream, 200)


# ---------------------------------------------------------------- equivalence


def _hand_written(instance, prediction):
    """The callable `07_pipelines.ipynb` section 7.6.1 asks users to write."""
    return int(instance.y_index == prediction)


def _run_classifier_pipeline(monitor):
    stream = ElectricityTiny()
    detector = ADWIN()
    pipeline = (
        ClassifierPipeline()
        .add_classifier(OnlineBagging(schema=stream.get_schema(), ensemble_size=3))
        .add_drift_detector(detector, monitor)
    )
    while stream.has_more_instances():
        instance = stream.next_instance()
        pipeline.predict(instance)
        pipeline.train(instance)
    return detector.detection_index


def test_prediction_is_correct_matches_the_hand_written_callable():
    """Equivalence against today's behaviour rather than a pinned constant."""
    assert _run_classifier_pipeline(
        prediction_is_correct()
    ) == _run_classifier_pipeline(_hand_written)


def test_absolute_error_matches_a_hand_written_regression_callable():
    def hand_written(instance, prediction):
        return 0.0 if prediction is None else float(abs(instance.y_value - prediction))

    def run(monitor):
        stream = FriedTiny()
        detector = ADWIN()
        pipeline = (
            RegressorPipeline()
            .add_regressor(
                AdaptiveRandomForestRegressor(
                    schema=stream.get_schema(), ensemble_size=3
                )
            )
            .add_drift_detector(detector, monitor)
        )
        while stream.has_more_instances():
            instance = stream.next_instance()
            pipeline.predict(instance)
            pipeline.train(instance)
        return detector.detection_index

    assert run(absolute_error()) == run(hand_written)


def test_feature_value_sees_the_transformed_feature():
    """Position matters: the same index differs before and after a transformer."""
    stream = ElectricityTiny()
    instance = stream.next_instance()
    transformer = MOATransformer(
        schema=stream.get_schema(), moa_filter=NormalisationFilter()
    )
    transformed = transformer.transform_instance(instance)

    # Normalising the very first instance yields all zeros, so compare an index
    # whose raw value is not already zero.
    index = next(i for i, value in enumerate(instance.x) if value != 0.0)
    monitor = feature_value(index)
    assert monitor(instance, None) != monitor(transformed, None)
