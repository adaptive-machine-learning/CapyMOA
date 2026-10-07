"""Round-trip tests for learner parameter serialization outside classifiers."""

from capymoa.anomaly import HalfSpaceTrees
from capymoa.cluster import ClusTree
from capymoa.datasets import ElectricityTiny, Fried
from capymoa.drift.detectors import ADWIN
from capymoa.regressor import AdaptiveRandomForestRegressor


def test_anomaly_detector_params_roundtrip():
    schema = ElectricityTiny().get_schema()
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
    schema = ElectricityTiny().get_schema()
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


def test_arf_regressor_nested_drift_detector_roundtrip():
    schema = Fried().get_schema()
    learner = AdaptiveRandomForestRegressor(
        schema,
        ensemble_size=5,
        drift_detection_method=ADWIN(delta=0.01),
    )

    params = learner.get_params()
    assert params["drift_detection_method"]["learner"] == (
        "capymoa.drift.detectors.adwin.ADWIN"
    )
    assert params["drift_detection_method"]["params"]["delta"] == 0.01

    clone = AdaptiveRandomForestRegressor.from_params(
        schema, params, learner.random_seed
    )
    assert clone.get_params() == params
