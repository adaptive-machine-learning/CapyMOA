from contextlib import nullcontext
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from typing_extensions import override

from capymoa.anomaly import HalfSpaceTrees, evaluate_anomaly
from capymoa.base import MOAClassifier
from capymoa.classifier import (
    HoeffdingTree,
    NaiveBayes,
    NoChange,
    evaluate_classifier,
    evaluate_classifiers,
)
from capymoa.classifier.evaluate import (
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
)
from capymoa.datasets import Electricity, ElectricityTiny
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation._loop import (
    _prequential_loop_fast,
    _Run,
    _use_java_loop,
)
from capymoa.exception import StreamTypeError
from capymoa.regressor import KNNRegressor
from capymoa.ssl import evaluate_ssl
from capymoa.stream.generator import (
    SEA,
    HyperPlaneRegression,
    RandomTreeGenerator,
    STAGGERGenerator,
)


def test_evaluate_classifier():
    """The stream should be restarted every time we run the evaluation, so the 11th instance should be the same, also
    the accuracy of models from the same learner (but different models) should be the same
    """
    stream = SEA(function=1)
    model1 = NaiveBayes(schema=stream.get_schema())
    model2 = NaiveBayes(schema=stream.get_schema())

    results_1st_run = evaluate_classifier(stream, model1, max_instances=10)
    eleventh_instance_1st_run = stream.next_instance().x
    results_2nd_run = evaluate_classifier(stream, model2, max_instances=10)
    eleventh_instance_2nd_run = stream.next_instance().x

    assert eleventh_instance_1st_run == pytest.approx(eleventh_instance_2nd_run)
    assert results_1st_run["stream"] == results_2nd_run["stream"] == str(stream)
    assert results_1st_run["accuracy"] == pytest.approx(
        results_2nd_run["accuracy"], abs=0.001
    )


def test_evaluate_classifiers():
    """One pass over the stream gives the same results as one pass per learner."""
    stream = SEA(function=1)
    learners = {
        "nb": NaiveBayes(schema=stream.get_schema()),
        "ht": HoeffdingTree(schema=stream.get_schema()),
    }

    together = evaluate_classifiers(stream, learners, max_instances=100)
    assert list(together) == ["nb", "ht"]
    assert together["nb"]["learner"] == "nb"
    # Evaluated together, the stream has been read once.
    hundredth_first_instance = stream.next_instance().x

    alone = {
        name: evaluate_classifier(
            stream,
            type(learner)(schema=stream.get_schema()),
            max_instances=100,
            optimise=False,
        )
        for name, learner in learners.items()
    }
    for name in learners:
        assert together[name]["accuracy"] == pytest.approx(
            alone[name]["accuracy"], abs=0.001
        )
    stream.restart()
    for _ in range(100):
        stream.next_instance()
    assert stream.next_instance().x == pytest.approx(hundredth_first_instance)


def test_evaluate_ssl():
    """The stream should be restarted every time we run the evaluation."""
    stream = SEA(function=1)
    model1 = NaiveBayes(schema=stream.get_schema())
    model2 = NaiveBayes(schema=stream.get_schema())

    results_1st_run = evaluate_ssl(stream, model1, max_instances=10)
    eleventh_instance_1st_run = stream.next_instance().x
    results_2nd_run = evaluate_ssl(stream, model2, max_instances=10)
    eleventh_instance_2nd_run = stream.next_instance().x

    assert eleventh_instance_1st_run == pytest.approx(eleventh_instance_2nd_run)
    assert results_1st_run["accuracy"] == pytest.approx(
        results_2nd_run["accuracy"], abs=0.001
    )
    assert results_1st_run["label_probability"] == 0.01


def test_prequential_evaluation_dispatches_on_learner_type():
    """``prequential_evaluation`` calls the ``evaluate_*`` of the learner's domain."""
    classification = ElectricityTiny()
    results = prequential_evaluation(
        classification, NaiveBayes(classification.get_schema()), max_instances=50
    )
    assert "accuracy" in results and "per_class" in results

    regression = HyperPlaneRegression()
    results = prequential_evaluation(
        regression, KNNRegressor(regression.get_schema()), max_instances=50
    )
    assert "rmse" in results and "accuracy" not in results

    many = prequential_evaluation(
        classification,
        {"a": NaiveBayes(classification.get_schema())},
        max_instances=50,
    )
    assert list(many) == ["a"]

    with pytest.raises(TypeError):
        prequential_evaluation(classification, object(), max_instances=50)


def test_single_and_many_learners_are_not_mixed_up():
    stream = ElectricityTiny()
    learner = NaiveBayes(stream.get_schema())
    with pytest.raises(TypeError, match="evaluate_classifiers"):
        evaluate_classifier(stream, {"nb": learner})
    with pytest.raises(TypeError, match="evaluate_classifier\\b"):
        evaluate_classifiers(stream, learner)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluate_classifiers(stream, {})


def test_run_info():
    """Every result starts with the shared run information."""
    stream = ElectricityTiny()
    results = evaluate_classifier(
        stream,
        HoeffdingTree(stream.get_schema()),
        max_instances=1500,
        window_size=500,
    )
    assert results["learner"] == "HoeffdingTree"
    assert results["stream"] == "ElectricityTiny"
    assert results["instances"] == 1500
    assert results["window_size"] == 500
    assert results["wallclock"] > 0
    assert results["cpu_time"] > 0
    assert list(results["windowed"]["instances"]) == [500, 1000, 1500]
    assert "drifts" not in results


@pytest.mark.parametrize("optimise", [True, False])
def test_no_windowed_results(optimise):
    stream = ElectricityTiny()
    results = evaluate_classifier(
        stream,
        HoeffdingTree(stream.get_schema()),
        max_instances=100,
        window_size=None,
        optimise=optimise,
    )
    assert "window_size" not in results
    assert "windowed" not in results


def test_evaluate_anomaly():
    """Fast and Python loops give the same AUC."""
    stream = Electricity()
    model1 = HalfSpaceTrees(schema=stream.get_schema())
    model2 = HalfSpaceTrees(schema=stream.get_schema())

    results_1st_run = evaluate_anomaly(
        stream=stream, learner=model1, window_size=1000, optimise=True
    )
    results_2nd_run = evaluate_anomaly(
        stream=stream, learner=model2, window_size=1000, optimise=False
    )

    assert results_1st_run["windowed"]["auc"][-1] == pytest.approx(
        results_2nd_run["windowed"]["auc"][-1], abs=0.001
    )
    assert results_1st_run["auc"] == pytest.approx(results_2nd_run["auc"], abs=0.001)


@pytest.mark.parametrize(
    ["restart_stream", "optimise", "regression", "evaluation"],
    list(
        product(
            [True, False],
            [True, False],
            [True, False],
            [
                prequential_evaluation,
                evaluate_ssl,
            ],
        )
    ),
)
def test_restart_stream_flag(restart_stream, optimise, regression, evaluation):
    """Ensure that the stream is restarted when the restart_stream flag is set to True"""
    expect_error = False
    # Some configurations are not supported by some evaluation methods.
    # When these are eventually supported, this test will need to be updated.

    # Create a stream and learner
    stream = (
        HyperPlaneRegression() if regression else RandomTreeGenerator(num_classes=10)
    )

    # This evaluation function does not yet support regression
    if evaluation == evaluate_ssl and regression:
        expect_error = True

    if not regression:
        learner = NaiveBayes(
            schema=stream.get_schema()
        )  # The type of model is not important
    else:
        learner = KNNRegressor(schema=stream.get_schema())
    assert _use_java_loop(stream, learner, optimise=True, window_size=10), (
        "Fast mode should always be compilable for this test"
    )

    def _take_y(num_instances):
        if regression:
            return [stream.next_instance().y_value for _ in range(num_instances)]
        else:
            return [stream.next_instance().y_index for _ in range(num_instances)]

    # Store targets from the stream for use in assertions later.
    y_stream = _take_y(20)
    stream.restart()  # Must restart the stream to get the same instances again

    # Consume the first 10 instances
    _take_y(10)
    with (
        pytest.raises((StreamTypeError, ValueError)) if expect_error else nullcontext()
    ):
        # Consume either the next 5 instances or the same 5 instances again
        # depending on the ``restart_stream`` flag
        evaluation(
            stream=stream,
            learner=learner,
            max_instances=5,
            optimise=optimise,
            restart_stream=restart_stream,
        )

        # If the stream is restarted, the next 5 instances should be the same as those
        # we remembered. Otherwise, they should be different.
        y_remaining = _take_y(5)
        if restart_stream is True:
            assert y_remaining == y_stream[5:10]
        else:
            assert y_remaining == y_stream[15:20]


@pytest.mark.parametrize("optimise", [False, True])
@pytest.mark.parametrize("store_y", [True, False])
@pytest.mark.parametrize("store_predictions", [True, False])
@pytest.mark.parametrize("eval_func", [prequential_evaluation, evaluate_ssl])
def test_store_y_and_store_predictions(
    eval_func, optimise: bool, store_y: bool, store_predictions: bool
):
    """Test ``evaluate_classifier``'s ``store_predictions`` and ``store_y`` flags."""
    n = 10
    stream = ElectricityTiny()
    expected_true_y = [stream.next_instance().y_index for _ in range(n)]
    stream.restart()

    learner = NoChange(schema=stream.get_schema())

    assert (
        _use_java_loop(stream, learner, optimise=True, window_size=10) or not optimise
    ), "Fast mode should be compilable for this test if optimise is True"
    results = eval_func(
        stream=stream,
        learner=learner,
        window_size=10,
        max_instances=n,
        store_predictions=store_predictions,
        store_y=store_y,
        optimise=optimise,
    )
    true_y = results.get("y_true")
    pred_y = results.get("y_pred")

    if store_y is True:
        assert true_y is not None
        assert len(true_y) == n
        assert isinstance(true_y, list)
        assert_array_equal(true_y, expected_true_y)
    else:
        assert true_y is None, "ground truth should not be stored"

    if store_predictions is True:
        assert pred_y is not None
        assert len(pred_y) == n
        assert isinstance(pred_y, list) and np.asarray(pred_y).dtype == np.int64

        # TODO: `evaluate_ssl` sometimes removes labels so we cannot
        # expect a match
        if eval_func != evaluate_ssl:
            assert_array_equal(
                pred_y[1:], expected_true_y[:-1]
            )  # NoChange predicts previous y
    else:
        assert pred_y is None, "predictions should not be stored"


@pytest.mark.parametrize(
    "make_stream",
    [
        lambda: SEA(function=1),
        RandomTreeGenerator,
        STAGGERGenerator,
        lambda: ElectricityTiny(),
    ],
)
def test_optimise_flag_does_not_change_results(make_stream):
    """``optimise`` selects a loop, so it must not change the answer.

    `MOAClassifier.predict_proba` used to discard any prediction whose
    unnormalised vote total was below 1e-2. MOA's votes are not probabilities
    and their scale depends on the learner, so for Naive Bayes -- products of
    likelihoods -- that discarded nearly everything. The Python loop scored
    each discarded prediction as a miss while the Java loop scored it
    normally, so the two disagreed by up to 52 accuracy points.
    """
    accuracies = []
    for optimise in (True, False):
        stream = make_stream()
        results = evaluate_classifier(
            stream=stream,
            learner=NaiveBayes(schema=stream.get_schema()),
            max_instances=2000,
            optimise=optimise,
        )
        accuracies.append(results["accuracy"])

    assert accuracies[0] == pytest.approx(accuracies[1], abs=0.01)


@pytest.mark.parametrize(
    ["votes", "expected"],
    [
        ([], None),  # no prediction available
        ([0.0, 0.0], None),  # nothing but zeros
        ([float("nan"), 1.0], None),
        ([float("inf"), 1.0], None),
        ([1e-30, 3e-30], [0.25, 0.75]),  # tiny but perfectly valid
        ([2.0, 6.0], [0.25, 0.75]),
    ],
)
def test_predict_proba_only_rejects_absent_predictions(votes, expected):
    """Small vote totals are valid; only absent or degenerate ones are not."""

    class _FakeMOALearner:
        def getVotesForInstance(self, _):
            return votes

    class _FakeInstance:
        java_instance = None

    class _FakeSchema:
        def get_num_classes(self):
            return 2

    classifier = MOAClassifier.__new__(MOAClassifier)
    classifier.moa_learner = _FakeMOALearner()
    classifier.schema = _FakeSchema()

    result = MOAClassifier.predict_proba(classifier, _FakeInstance())
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_fast_loop_refuses_custom_test_then_train():
    """The Java loop would skip a custom ``test_then_train``, so it must refuse it."""

    class _CustomRun(_Run):
        @override
        def test_then_train(self, batch):
            return super().test_then_train(batch)

    stream = ElectricityTiny()
    schema = stream.get_schema()
    run = _CustomRun(
        NaiveBayes(schema),
        ClassificationEvaluator(schema=schema),
        ClassificationWindowedEvaluator(schema=schema, window_size=10),
        store_y=False,
        store_predictions=False,
    )
    with pytest.raises(TypeError, match="custom test_then_train"):
        _prequential_loop_fast(stream, run, max_instances=10)
