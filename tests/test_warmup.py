import pytest

from capymoa.anomaly import HalfSpaceTrees, evaluate_anomaly
from capymoa.classifier import HoeffdingTree, evaluate_classifier
from capymoa.datasets import ElectricityTiny, FriedTiny
from capymoa.regressor import FIMTDD, evaluate_regressor


def test_classifier_warmup_skips_tests():
    stream = ElectricityTiny()
    r = evaluate_classifier(
        stream, HoeffdingTree(stream.get_schema()), max_instances=500, warmup=200
    )
    assert r["instances"] == 300


def test_regressor_warmup_skips_tests():
    stream = FriedTiny()
    r = evaluate_regressor(
        stream, FIMTDD(stream.get_schema()), max_instances=500, warmup=200
    )
    assert r["instances"] == 300


def test_anomaly_warmup_skips_tests():
    stream = ElectricityTiny()
    r = evaluate_anomaly(
        stream, HalfSpaceTrees(stream.get_schema()), max_instances=500, warmup=200
    )
    assert r["instances"] == 300


def test_negative_warmup_raises():
    stream = ElectricityTiny()
    with pytest.raises(ValueError):
        evaluate_classifier(stream, HoeffdingTree(stream.get_schema()), warmup=-1)
