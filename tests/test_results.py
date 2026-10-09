"""The results of every domain are typed dictionaries."""

import pickle
import types
import typing
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd
import pytest
from typing_extensions import is_typeddict

from capymoa.anomaly import HalfSpaceTrees, evaluate_anomaly
from capymoa.anomaly.evaluate import AnomalyResults
from capymoa.classifier import HoeffdingTree, NaiveBayes, evaluate_classifier
from capymoa.classifier.evaluate import ClassifierResults
from capymoa.datasets import ElectricityTiny, Fried, FriedTiny
from capymoa.evaluation import RunInfo
from capymoa.regressor import FIMTDD, evaluate_regressor
from capymoa.regressor.evaluate import RegressorResults
from capymoa.ssl import evaluate_ssl
from capymoa.ssl.evaluate import SSLResults
from capymoa.stream.drift import (
    AbruptDrift,
    DriftStream,
    GradualDrift,
    RecurrentConceptDriftStream,
)
from capymoa.stream.generator import SEA
from capymoa.uncertainty import MVE, evaluate_prediction_interval
from capymoa.uncertainty.evaluate import PredictionIntervalResults


def _matches(value: Any, hint: Any) -> bool:
    origin = get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        return any(_matches(value, arg) for arg in get_args(hint))
    if origin is list:
        return isinstance(value, list)
    if is_typeddict(hint):
        hints = get_type_hints(hint)
        return (
            isinstance(value, dict)
            and hint.__required_keys__ <= set(value) <= set(hints)
            and all(_matches(v, hints[k]) for k, v in value.items())
        )
    if hint is type(None):
        return value is None
    if hint is float:
        return isinstance(value, (int, float, np.floating)) and not isinstance(
            value, bool
        )
    if hint is int:
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
    return isinstance(value, hint)


def assert_is_result(results: dict, result_type: type) -> None:
    hints = get_type_hints(result_type)
    keys = set(results)
    assert result_type.__required_keys__ <= keys <= set(hints), (
        f"missing {result_type.__required_keys__ - keys}, extra {keys - set(hints)}"
    )
    for key, value in results.items():
        assert _matches(value, hints[key]), (
            f"{key}={value!r} is not {hints[key]} in {result_type.__name__}"
        )
    for key in ("windowed", "per_class"):
        if key in results:
            lengths = {len(column) for column in results[key].values()}
            assert len(lengths) == 1, f"{key} columns have different lengths"


def assert_same(actual: Any, expected: Any) -> None:
    """Deep comparison of two results."""
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape
        if expected.dtype.kind in "OUS":
            assert list(actual) == list(expected)
        else:
            np.testing.assert_allclose(actual, expected)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def _classifier(**kwargs):
    stream = ElectricityTiny()
    return evaluate_classifier(
        stream, HoeffdingTree(stream.get_schema()), max_instances=500, **kwargs
    )


def _regressor(**kwargs):
    stream = FriedTiny()
    return evaluate_regressor(
        stream, FIMTDD(stream.get_schema()), max_instances=500, **kwargs
    )


def _prediction_interval(**kwargs):
    stream = Fried()
    return evaluate_prediction_interval(
        stream, MVE(stream.get_schema()), max_instances=500, **kwargs
    )


def _anomaly(**kwargs):
    stream = ElectricityTiny()
    return evaluate_anomaly(
        stream, HalfSpaceTrees(stream.get_schema()), max_instances=500, **kwargs
    )


def _ssl(**kwargs):
    stream = ElectricityTiny()
    return evaluate_ssl(
        stream, NaiveBayes(stream.get_schema()), max_instances=500, **kwargs
    )


CASES = [
    pytest.param(_classifier, ClassifierResults, id="classifier"),
    pytest.param(_regressor, RegressorResults, id="regressor"),
    pytest.param(_prediction_interval, PredictionIntervalResults, id="interval"),
    pytest.param(_anomaly, AnomalyResults, id="anomaly"),
    pytest.param(_ssl, SSLResults, id="ssl"),
]


def test_inheritance():
    assert set(get_type_hints(RunInfo)) < set(get_type_hints(ClassifierResults))
    assert set(get_type_hints(RegressorResults)) < set(
        get_type_hints(PredictionIntervalResults)
    )
    assert set(get_type_hints(ClassifierResults)) < set(get_type_hints(SSLResults))
    assert set(get_type_hints(RunInfo)) < set(get_type_hints(AnomalyResults))


@pytest.mark.parametrize("evaluate, result_type", CASES)
@pytest.mark.parametrize("stored", [True, False])
def test_keys_and_types(evaluate, result_type, stored):
    results = evaluate(store_y=stored, store_predictions=stored)
    assert_is_result(results, result_type)
    assert ("y_true" in results) == stored
    assert ("y_pred" in results) == stored
    if stored:
        assert isinstance(results["y_true"], list)
        assert len(results["y_true"]) == results["instances"] == 500
        assert len(results["y_pred"]) == 500


@pytest.mark.parametrize(
    "evaluate, result_type",
    [case for case in CASES if case.id != "interval"],  # no fast loop
)
def test_fast_and_python_loops_have_the_same_keys(evaluate, result_type):
    fast = evaluate(optimise=True)
    slow = evaluate(optimise=False)
    assert list(fast) == list(slow)
    assert_is_result(slow, result_type)
    assert list(fast["windowed"]) == list(slow["windowed"])
    assert fast["instances"] == slow["instances"]


def test_drifts():
    stream = DriftStream(
        stream=[SEA(function=1), AbruptDrift(position=300), SEA(function=2)]
    )
    results = evaluate_classifier(
        stream, NaiveBayes(stream.get_schema()), max_instances=500
    )
    assert results["drifts"] == [300]
    assert results["drift_widths"] == [0]
    assert "concepts" not in results
    assert_is_result(results, ClassifierResults)


def test_gradual_drift_widths():
    stream = DriftStream(
        stream=[
            SEA(function=1),
            GradualDrift(position=300, width=50),
            SEA(function=2),
        ]
    )
    results = evaluate_classifier(
        stream, NaiveBayes(stream.get_schema()), max_instances=500
    )
    assert results["drifts"] == [300]
    assert results["drift_widths"] == [50]


def test_recurrent_concepts():
    stream = RecurrentConceptDriftStream(
        concept_list=[SEA(function=1), SEA(function=2)],
        max_recurrences_per_concept=2,
        transition_type_template=AbruptDrift(position=200),
    )
    results = evaluate_classifier(
        stream, NaiveBayes(stream.get_schema()), max_instances=500
    )
    assert results["concepts"] == [
        {"id": str(c["id"]), "start": c["start"], "end": c["end"]}
        for c in stream.concept_info
    ]
    assert_is_result(results, ClassifierResults)


@pytest.mark.parametrize("delay_length", [0, 10])
def test_ssl_unlabeled_same_in_both_loops(delay_length):
    fast = _ssl(label_probability=0.1, delay_length=delay_length)
    slow = _ssl(label_probability=0.1, delay_length=delay_length, optimise=False)
    assert fast["unlabeled"] == slow["unlabeled"]
    assert 0 < fast["unlabeled"] < fast["instances"]
    assert fast["unlabeled_ratio"] == pytest.approx(
        fast["unlabeled"] / fast["instances"]
    )


def test_empty_mapping_of_learners():
    with pytest.raises(ValueError, match="No learners"):
        evaluate_classifier(ElectricityTiny(), {})


@pytest.mark.parametrize("optimise", [True, False])
def test_evaluate_checks_stream_type(optimise):
    classification, regression = ElectricityTiny(), FriedTiny()
    with pytest.raises(ValueError, match="not a classification stream"):
        evaluate_classifier(
            regression, NaiveBayes(classification.get_schema()), optimise=optimise
        )
    with pytest.raises(ValueError, match="not a regression stream"):
        evaluate_regressor(
            classification, FIMTDD(regression.get_schema()), optimise=optimise
        )


def test_per_class():
    results = _classifier()
    assert list(results["per_class"]) == [
        "label",
        "precision",
        "recall",
        "f1_score",
    ]
    assert len(results["per_class"]["label"]) == 2


@pytest.mark.parametrize("evaluate, result_type", CASES)
def test_windowed_round_trips_with_pandas(evaluate, result_type):
    windowed = evaluate()["windowed"]
    frame = pd.DataFrame(windowed)
    assert list(frame.columns) == list(windowed)
    assert len(frame) == len(windowed["instances"])
    assert frame["instances"].dtype.kind == "i"
    back = {column: frame[column].to_numpy() for column in frame}
    assert_same(back, windowed)


def test_optional_keys_are_left_out():
    """Optional keys are absent, not ``None``, when there is nothing to put in."""
    results = _classifier(window_size=None)
    for key in ("window_size", "windowed", "y_true", "y_pred", "drifts"):
        assert key not in results
    assert_is_result(results, ClassifierResults)

    results = _classifier()
    assert "windowed" in results and "window_size" in results
    assert "y_true" not in results and "y_pred" not in results


@pytest.mark.parametrize("evaluate, result_type", CASES)
def test_pickle_round_trip(evaluate, result_type):
    """Results are plain data, so users can store them as they like."""
    results = evaluate(store_y=True, store_predictions=True)
    assert_same(pickle.loads(pickle.dumps(results)), results)


@pytest.mark.torch
def test_ocl_results():
    from capymoa.ocl import evaluate_ocl
    from capymoa.ocl.datasets import TinySplitMNIST
    from capymoa.ocl.evaluation import OCLResults, OnlineResults

    scenario = TinySplitMNIST()
    learner = NaiveBayes(scenario.schema)
    results = evaluate_ocl(
        learner, scenario.train_loaders(32), scenario.test_loaders(32)
    )
    hints = get_type_hints(OCLResults)
    assert set(results) == set(hints)
    for key, hint in hints.items():
        if key != "ttt":
            assert _matches(results[key], hint), f"{key} is not {hint}"
    assert_is_result(results["ttt"], OnlineResults)

    assert_same(pickle.loads(pickle.dumps(results)), results)
