"""Tests for learner parameter round-trips and YAML construction.

TODO: Remove this test module in the future and replace it with the standard
`tests/test_classifiers.py` tests using serialized test cases.

The round-trip tests reuse the existing classifier/regressor test-case
registries from `test_classifiers.py`/`test_regressors.py`: for each learner,
we construct it, capture its hyper-parameters via `get_params()`, reconstruct
a clone via `from_params()`, train both on the same stream, and assert they
reach the same accuracy/RMSE.
"""

from pathlib import Path

import pytest
import yaml

from capymoa.anomaly import HalfSpaceTrees
from capymoa.base import learner_from_params
from capymoa.cluster import ClusTree
from capymoa.datasets import ElectricityTiny, Fried
from capymoa.evaluation import prequential_evaluation
from capymoa.regressor import AdaptiveRandomForestRegressor

from .test_classifiers import test_cases
from .test_regressors import CASES


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(c, marks=pytest.mark.torch) if c.needs_torch else c
        for c in test_cases
    ],
    ids=[c.test_name for c in test_cases],
)
def test_classifier_params_roundtrip(test_case):
    if test_case.skip_reason:
        pytest.skip(test_case.skip_reason)

    stream = ElectricityTiny()
    schema = stream.get_schema()
    learner = test_case.learner_constructor(schema=schema)

    params = learner.get_params()
    clone = type(learner).from_params(schema, params, learner.random_seed)
    assert clone.get_params() == params

    stream.restart()
    original_results = prequential_evaluation(
        stream, learner, window_size=100, batch_size=test_case.batch_size
    )
    stream.restart()
    clone_results = prequential_evaluation(
        stream, clone, window_size=100, batch_size=test_case.batch_size
    )

    assert clone_results.cumulative.accuracy() == pytest.approx(
        original_results.cumulative.accuracy(), abs=1e-6
    )


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_regressor_params_roundtrip(case):
    stream = Fried()
    schema = stream.get_schema()
    learner = case.type(schema=schema, **(case.options or {}))

    params = learner.get_params()
    clone = type(learner).from_params(schema, params, learner.random_seed)
    assert clone.get_params() == params

    stream.restart()
    i = 0
    while stream.has_more_instances():
        i += 1
        if i > 300:
            break
        instance = stream.next_instance()
        original_pred = learner.predict(instance)
        clone_pred = clone.predict(instance)
        assert original_pred == pytest.approx(clone_pred, abs=1e-9) or (
            original_pred is None and clone_pred is None
        )
        learner.train(instance)
        clone.train(instance)


def test_nested_learner_params_roundtrip():
    """Test an ensemble with a configured nested CapyMOA base learner."""
    from capymoa.classifier import HoeffdingTree, OnlineBagging

    stream = ElectricityTiny()
    schema = stream.get_schema()

    learner = OnlineBagging(
        schema,
        base_learner=HoeffdingTree(schema, grace_period=33),
        ensemble_size=7,
    )
    params = learner.get_params()
    assert params["base_learner"]["learner"] == "capymoa.classifier.HoeffdingTree"
    assert params["base_learner"]["params"]["grace_period"] == 33

    clone = OnlineBagging.from_params(schema, params, learner.random_seed)
    assert clone.get_params() == params

    stream.restart()
    original_results = prequential_evaluation(stream, learner, max_instances=300)
    stream.restart()
    clone_results = prequential_evaluation(stream, clone, max_instances=300)
    assert clone_results.cumulative.accuracy() == pytest.approx(
        original_results.cumulative.accuracy(), abs=1e-6
    )


def test_arf_regressor_nested_drift_detector_roundtrip():
    """Test AdaptiveRandomForestRegressor nested drift detector parameters."""
    from capymoa.drift.detectors import ADWIN

    stream = Fried()
    schema = stream.get_schema()

    learner = AdaptiveRandomForestRegressor(
        schema,
        ensemble_size=5,
        drift_detection_method=ADWIN(delta=0.01),
    )
    params = learner.get_params()
    assert (
        params["drift_detection_method"]["learner"]
        == "capymoa.drift.detectors.adwin.ADWIN"
    )
    assert params["drift_detection_method"]["params"]["delta"] == 0.01

    clone = AdaptiveRandomForestRegressor.from_params(
        schema, params, learner.random_seed
    )
    assert clone.get_params() == params


def test_anomaly_detector_params_roundtrip():
    stream = ElectricityTiny()
    schema = stream.get_schema()

    learner = HalfSpaceTrees(
        schema,
        window_size=100,
        number_of_trees=5,
        max_depth=7,
        anomaly_threshold=0.7,
        size_limit=0.2,
    )
    params = learner.get_params()
    clone = HalfSpaceTrees.from_params(schema, params, learner.random_seed)

    assert params == {
        "CLI": None,
        "window_size": 100,
        "number_of_trees": 5,
        "max_depth": 7,
        "anomaly_threshold": 0.7,
        "size_limit": 0.2,
    }
    assert clone.get_params() == params


def test_clusterer_params_roundtrip():
    stream = ElectricityTiny()
    schema = stream.get_schema()

    learner = ClusTree(
        schema,
        horizon=500,
        max_height=6,
        breadth_first_strategy=True,
    )
    params = learner.get_params()
    clone = ClusTree.from_params(schema, params)

    assert params == {
        "horizon": 500,
        "max_height": 6,
        "breadth_first_strategy": True,
    }
    assert clone.get_params() == params


LEARNER_PARAMS_FIXTURE = Path(__file__).parent / "fixtures" / "learner_params.yaml"


def test_learner_from_params_yaml():
    specs = yaml.safe_load(LEARNER_PARAMS_FIXTURE.read_text(encoding="utf-8"))

    stream = ElectricityTiny()
    schema = stream.get_schema()

    learners = [learner_from_params(spec, schema=schema) for spec in specs]

    hoeffding_tree, online_bagging = learners
    assert hoeffding_tree.get_params()["grace_period"] == 50
    assert hoeffding_tree.get_params()["confidence"] == 0.01

    assert online_bagging.get_params()["ensemble_size"] == 7
    nested = online_bagging.get_params()["base_learner"]
    assert nested["learner"] == "capymoa.classifier.HoeffdingTree"
    assert nested["params"]["grace_period"] == 33

    # Smoke test: constructed learners must actually be trainable/usable.
    for learner in learners:
        for _ in range(20):
            instance = stream.next_instance()
            learner.predict(instance)
            learner.train(instance)
        stream.restart()
