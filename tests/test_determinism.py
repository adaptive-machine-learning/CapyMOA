"""Seed-reproducibility contract tests for stochastic learners.

Guards the random_seed -> MOA learner wiring: two learners constructed with
the same seed must produce identical predictions on the same stream prefix.
If the setRandomSeed call in the MOA wrapper bases is ever lost in a
refactor, these tests fail instead of silently producing non-reproducible
research results.

Deterministic learners (MajorityClass, NoChange, HoeffdingTree, ...) are
intentionally excluded: they ignore the seed by design.
"""

import pytest

from capymoa.classifier import (
    AdaptiveRandomForestClassifier,
    LeveragingBagging,
    OnlineBagging,
    StreamingRandomPatches,
)
from capymoa.datasets import ElectricityTiny

STOCHASTIC_CLASSIFIERS = [
    AdaptiveRandomForestClassifier,
    LeveragingBagging,
    OnlineBagging,
    StreamingRandomPatches,
]


def _prediction_trace(learner, stream, limit=100):
    """Train-and-predict over the stream prefix; returns comparable bytes."""
    trace = []
    for i, instance in enumerate(stream):
        if i >= limit:
            break
        learner.train(instance)
        proba = learner.predict_proba(instance)
        trace.append(proba.tobytes() if proba is not None else b"<None>")
    return trace


@pytest.mark.parametrize("learner_cls", STOCHASTIC_CLASSIFIERS)
def test_same_seed_produces_identical_predictions(learner_cls):
    stream = ElectricityTiny()
    schema = stream.get_schema()

    first = _prediction_trace(learner_cls(schema=schema, random_seed=7), stream)
    stream.restart()
    second = _prediction_trace(learner_cls(schema=schema, random_seed=7), stream)

    assert first == second, (
        f"{learner_cls.__name__}: same random_seed produced different "
        f"predictions (first difference at instance "
        f"{next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)}) - "
        "the seed is not reaching the learner's randomness"
    )


@pytest.mark.parametrize("learner_cls", STOCHASTIC_CLASSIFIERS)
def test_different_seeds_change_stochastic_predictions(learner_cls):
    """Stochastic learners must actually consume the seed.

    A learner whose predictions never depend on the seed is either
    deterministic by design (and belongs in the other test's exclusion
    list) or is silently re-seeded from a fixed source.
    """
    stream = ElectricityTiny()
    schema = stream.get_schema()

    first = _prediction_trace(learner_cls(schema=schema, random_seed=1), stream)
    stream.restart()
    second = _prediction_trace(learner_cls(schema=schema, random_seed=42), stream)

    n_differ = sum(1 for a, b in zip(first, second) if a != b)
    assert n_differ > 0, (
        f"{learner_cls.__name__}: predictions identical under seeds 1 and 42 - "
        "a stochastic learner should be seed-sensitive"
    )


# ---------------------------------------------------------------------------
# Regressors: same contract, different module. Regression trees branch on
# numeric split points, so a lost setRandomSeed call shows up as drifting
# predictions rather than reordered class probabilities. The window is 500
# instances because ARFFIMTDD only starts consuming its seed after its
# 200-instance grace period (verified: first cross-seed difference at 201).
#
# Excluded as deterministic by design (verified on Fried, 500 instances,
# cross-seed traces identical): SGDRegressor, PassiveAggressiveRegressor,
# KNNRegressor, StochasticGradientTree.
# ---------------------------------------------------------------------------

from capymoa.datasets import Fried  # noqa: E402
from capymoa.regressor import (  # noqa: E402
    AdaptiveRandomForestRegressor,
    ARFFIMTDD,
    FIMTDD,
    ORTO,
    SOKNL,
    StreamingGradientBoostedRegression,
)

STOCHASTIC_REGRESSORS = [
    AdaptiveRandomForestRegressor,
    ARFFIMTDD,
    FIMTDD,
    ORTO,
    SOKNL,
    StreamingGradientBoostedRegression,
]


def _regressor_trace(learner, stream, limit=500):
    """Train-and-predict over the stream prefix; returns comparable values."""
    trace = []
    for i, instance in enumerate(stream):
        if i >= limit:
            break
        learner.train(instance)
        trace.append(repr(learner.predict(instance)))
    return trace


@pytest.mark.parametrize("learner_cls", STOCHASTIC_REGRESSORS)
def test_same_seed_produces_identical_regressor_predictions(learner_cls):
    stream = Fried()
    schema = stream.get_schema()

    first = _regressor_trace(learner_cls(schema=schema, random_seed=7), stream)
    stream.restart()
    second = _regressor_trace(learner_cls(schema=schema, random_seed=7), stream)

    assert first == second, (
        f"{learner_cls.__name__}: same random_seed produced different "
        f"predictions (first difference at instance "
        f"{next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)}) - "
        "the seed is not reaching the learner's randomness"
    )


@pytest.mark.parametrize("learner_cls", STOCHASTIC_REGRESSORS)
def test_different_seeds_change_stochastic_regressor_predictions(learner_cls):
    """Stochastic regressors must actually consume the seed."""
    stream = Fried()
    schema = stream.get_schema()

    first = _regressor_trace(learner_cls(schema=schema, random_seed=1), stream)
    stream.restart()
    second = _regressor_trace(learner_cls(schema=schema, random_seed=42), stream)

    n_differ = sum(1 for a, b in zip(first, second) if a != b)
    assert n_differ > 0, (
        f"{learner_cls.__name__}: predictions identical under seeds 1 and 42 - "
        "a stochastic regressor should be seed-sensitive"
    )
