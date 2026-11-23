import inspect
import os
from contextlib import nullcontext
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import pytest
from jpype import JException

from capymoa.base import MOAClassifier, MOARegressor, Regressor
from capymoa.core.io import load_model, save_model
from capymoa.datasets import Fried, FriedTiny
from capymoa.regressor import (
    ARFFIMTDD,
    FIMTDD,
    ORTO,
    SOKNL,
    SOKNLBT,
    AdaptiveRandomForestRegressor,
    FadingTargetMean,
    KNNRegressor,
    NoChange,
    OnlineBaggingRegressor,
    PassiveAggressiveRegressor,
    SGDRegressor,
    ShrubsRegressor,
    StochasticGradientTree,
    StreamingGradientBoostedRegression,
    TargetMean,
    evaluate_regressor,
)
from capymoa.regressor.evaluate import RegressionEvaluator, RegressionWindowedEvaluator
from capymoa.stream import Stream

SEED = 1
"""Seed for every regressor that takes one, so results are exact."""

FRIED_SCHEMA = FriedTiny().get_schema()


@dataclass(frozen=True)
class Case:
    type: type[Regressor]
    """The regressor class to test."""
    rmse: float
    """Expected RMSE"""
    win_rmse: float
    """Expected windowed RMSE"""
    options: dict | None = None
    """Keyword arguments to pass to the regressor constructor."""
    warmup: int = 0
    """Number of instances to train on before testing."""
    serializable: bool = True
    """If not set, expect saving the regressor to raise."""

    @property
    def id(self) -> str:
        """A string identifier for the test case."""
        options = dict(self.options or {})
        if self.warmup:
            options["warmup"] = self.warmup
        if options:
            args = ", ".join(f"{k}={v}" for k, v in options.items())
            return f"{self.type.__name__}({args})"
        else:
            return self.type.__name__


CASES = [
    Case(NoChange, 6.89, 6.06),
    Case(TargetMean, 4.98, 4.64),
    Case(FadingTargetMean, 5.09, 4.68, {"factor": 0.9}),
    Case(AdaptiveRandomForestRegressor, 4.15, 3.84),
    Case(KNNRegressor, 3.03, 2.49),
    Case(FIMTDD, 7.36, 5.25),
    Case(ARFFIMTDD, 4.95, 4.57),
    Case(ORTO, 9.23, 7.51),
    Case(SOKNLBT, 4.95, 4.57),
    Case(SOKNL, 3.35, 2.69),
    Case(PassiveAggressiveRegressor, 3.70, 3.68),
    Case(SGDRegressor, 4.63, 3.61),
    Case(ShrubsRegressor, 5.21, 4.76),
    Case(StochasticGradientTree, 15.49, 16.19),
    Case(StreamingGradientBoostedRegression, 3.12, 2.15),
    # Train on the first 500 instances without testing them.
    Case(KNNRegressor, 2.93, 2.49, warmup=500),
    Case(SGDRegressor, 3.86, 3.61, warmup=500),
    Case(OnlineBaggingRegressor, 7.10, 4.38),
    # Java cannot serialize the proxies MOA uses to run Python learners.
    Case(
        OnlineBaggingRegressor,
        4.68,
        3.61,
        {"base_learner": SGDRegressor(FRIED_SCHEMA), "ensemble_size": 5},
        serializable=False,
    ),
]
"""Add your new test cases here ^^"""


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_regressor(case: Case):
    """Update TEST_CASES to add more regressors to test."""
    stream = Fried()
    schema = stream.get_schema()
    options = dict(case.options or {})
    if "random_seed" in inspect.signature(case.type).parameters:
        options["random_seed"] = SEED
    learner = case.type(schema=schema, **options)
    results = evaluate_regressor(
        stream, learner, max_instances=1000, window_size=100, warmup=case.warmup
    )

    # Check that the model's accuracy is as expected.
    actual = {
        "rmse": results["cumulative"]["rmse"],
        "win_rmse": results["windowed"]["rmse"][-1],
    }
    expected = {"rmse": case.rmse, "win_rmse": case.win_rmse}
    assert actual == pytest.approx(expected, abs=0.01)

    # Test that the model can be saved and loaded.
    subtest_save_and_load(learner, stream, case.serializable)


def test_streaming_gradient_boosted_regression_predicts_numeric_targets():
    stream = FriedTiny()
    learner = StreamingGradientBoostedRegression(schema=stream.get_schema())

    assert isinstance(learner, MOARegressor)
    assert not isinstance(learner, MOAClassifier)

    predictions = []
    for _ in range(20):
        instance = stream.next_instance()
        predictions.append(learner.predict(instance))
        learner.train(instance)

    numeric_predictions = [
        float(prediction) for prediction in predictions if prediction is not None
    ]
    rounded_predictions = {round(prediction, 6) for prediction in numeric_predictions}

    assert numeric_predictions
    assert any(prediction != 0.0 for prediction in numeric_predictions)
    assert len(rounded_predictions) > 1


def test_online_bagging_regressor_base_learner():
    """A CLI string and a CapyMOA regressor configure the same base learner."""
    schema = FriedTiny().get_schema()
    from_cli = OnlineBaggingRegressor(
        schema, base_learner="trees.FIMTDD -g 50", ensemble_size=3
    )
    from_wrapper = OnlineBaggingRegressor(
        schema, base_learner=FIMTDD(schema, grace_period=50), ensemble_size=3
    )
    assert str(from_cli) == "OnlineBaggingRegressor"
    for learner in (from_cli, from_wrapper):
        sub_learners = learner.moa_learner.getSubClassifiers()
        assert len(sub_learners) == 3
        assert all(
            str(sub.getClass().getSimpleName()) == "FIMTDD" for sub in sub_learners
        )
        assert sub_learners[0].gracePeriodOption.getValue() == 50


def _score(classifier: Regressor, stream: Stream, limit=100) -> float:
    """Eval without training the classifier."""
    stream.restart()
    evaluator = RegressionEvaluator(schema=stream.get_schema())
    i = 0
    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = classifier.predict(instance)
        evaluator.update(instance.y_value, prediction)
        i += 1
        if i > limit:
            break

    return evaluator.mae()


def subtest_save_and_load(
    regressor: Regressor,
    stream: Stream,
    is_serializable: bool,
):
    """A subtest to check if a classifier can be saved and loaded."""

    with TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "model.pkl")
        with pytest.raises(JException) if not is_serializable else nullcontext():
            # Save and load the model
            with open(tmp_file, "wb") as f:
                save_model(regressor, f)
            with open(tmp_file, "rb") as f:
                loaded_regressor: Regressor = load_model(f)

            # Check that the saved and loaded model have the same accuracy
            expected_acc = _score(regressor, stream)
            loaded_acc = _score(loaded_regressor, stream)
            assert expected_acc == loaded_acc, (
                f"Original accuracy {expected_acc * 100:.2f} != loaded accuracy {loaded_acc * 100:.2f}"
            )

            # Check that the loaded model can still be trained
            loaded_regressor.train(stream.next_instance())
