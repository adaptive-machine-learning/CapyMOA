"""Tests for the proxies that let MOA ensembles hold Python learners."""

import pytest
from jpype import JObject
from moa.classifiers import Classifier
from moa.classifiers.meta import AdaptiveRandomForest as _ARF
from moa.classifiers.meta import LeveragingBag as _LeveragingBag
from moa.classifiers.meta import OzaBag as _OzaBag
from moa.classifiers.meta import StreamingRandomPatches as _SRP

from capymoa.base import MOAClassifier, MOARegressor
from capymoa.classifier import HoeffdingTree, PassiveAggressiveClassifier
from capymoa.core.moa._python_proxy import (
    resolve_base_learner,
    resolve_base_regressor,
    resolve_drift_detector,
)
from capymoa.datasets import ElectricityTiny, FriedTiny
from capymoa.drift.detectors import ABCD
from capymoa.evaluation import prequential_evaluation
from capymoa.regressor import SGDRegressor


def _ensemble(java_class, schema, learner, option="baseLearner", seed=1):
    """Build a MOA ensemble with a (maybe Python) base learner."""
    cli, java_options = resolve_base_learner(learner, option)
    return MOAClassifier(
        java_class(),
        schema=schema,
        CLI=f"{f'-l {cli} ' if cli else ''}-s 5",
        random_seed=seed,
        java_options=java_options,
    )


def _accuracy(learner, n=300, optimise=True):
    results = prequential_evaluation(
        ElectricityTiny(), learner, max_instances=n, optimise=optimise
    )
    return results["cumulative"].accuracy()


@pytest.mark.parametrize("java_class", [_OzaBag, _LeveragingBag, _SRP])
def test_python_base_learner_in_moa_ensemble(java_class):
    schema = ElectricityTiny().get_schema()
    learner = _ensemble(java_class, schema, PassiveAggressiveClassifier(schema))
    assert _accuracy(learner) > 50


def test_moa_base_learner_has_no_java_options():
    schema = ElectricityTiny().get_schema()
    cli, java_options = resolve_base_learner(HoeffdingTree(schema))
    assert cli.startswith("(trees.HoeffdingTree")
    assert java_options == {}


@pytest.mark.parametrize("java_class", [_OzaBag, _SRP])
def test_deterministic_and_optimise_agrees(java_class):
    def run(optimise):
        schema = ElectricityTiny().get_schema()
        learner = _ensemble(java_class, schema, PassiveAggressiveClassifier(schema))
        return _accuracy(learner, 100, optimise)

    assert run(True) == run(True)
    assert run(True) == run(False)


@pytest.mark.parametrize("option", ["driftDetectionMethod", "warningDetectionMethod"])
@pytest.mark.parametrize("java_class", [_SRP, _ARF])
def test_python_drift_detector_in_moa_ensemble(java_class, option):
    schema = ElectricityTiny().get_schema()
    cli, java_options = resolve_drift_detector(ABCD(), option)
    assert cli is None
    learner = MOAClassifier(
        java_class(), schema=schema, CLI="-s 5", java_options=java_options
    )
    assert _accuracy(learner) > 50


def test_python_regressor_proxy():
    from moa.classifiers.predictioninterval import MVEPredictionInterval

    from capymoa.base import MOAPredictionIntervalLearner

    stream = FriedTiny()
    schema = stream.get_schema()
    cli, java_options = resolve_base_regressor(SGDRegressor(schema), "learner")
    learner = MOAPredictionIntervalLearner(
        schema=schema,
        moa_learner=MVEPredictionInterval(),
        java_options=java_options,
    )
    assert isinstance(learner, MOARegressor)
    for i, instance in enumerate(stream):
        if i >= 100:
            break
        assert len(learner.predict(instance)) == 3
        learner.train(instance)


def test_copy_makes_an_untrained_learner():
    schema = ElectricityTiny().get_schema()
    _, java_options = resolve_base_learner(PassiveAggressiveClassifier(schema))
    proxy = JObject(java_options["baseLearner"], Classifier)
    proxy.setModelContext(schema.get_moa_header())
    copy = JObject(proxy.copy(), Classifier)
    assert copy is not proxy
    assert not copy.trainingHasStarted()


@pytest.mark.parametrize(
    "resolve", [resolve_base_learner, resolve_base_regressor, resolve_drift_detector]
)
def test_bad_type_message(resolve):
    args = (object(), "option") if resolve is resolve_drift_detector else (object(),)
    with pytest.raises(TypeError, match="must be a"):
        resolve(*args)
