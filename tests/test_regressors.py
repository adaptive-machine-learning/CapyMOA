from contextlib import nullcontext
import os
from typing import Optional
from capymoa.evaluation import RegressionEvaluator, RegressionWindowedEvaluator
from capymoa.datasets import Fried
from capymoa.misc import load_model, save_model
from capymoa.regressor import (
    KNNRegressor,
    AdaptiveRandomForestRegressor,
    FIMTDD,
    ARFFIMTDD,
    ORTO,
    SOKNLBT,
    SOKNL,
    PassiveAggressiveRegressor,
    SGDRegressor,
    ShrubsRegressor,
    NoChange,
    TargetMean,
    FadingTargetMean,
)
from jpype import JException
import pytest
from capymoa.base import Regressor
from capymoa.stream import Schema, Stream
from tempfile import TemporaryDirectory
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    type: type[Regressor]
    """The regressor class to test."""
    rmse: float
    """Expected RMSE"""
    win_rmse: float
    """Expected windowed RMSE"""
    options: Optional[dict] = None
    """Keyword arguments to pass to the regressor constructor."""

    @property
    def id(self) -> str:
        """A string identifier for the test case."""
        if self.options:
            args = ", ".join(f"{k}={v}" for k, v in self.options.items())
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
    Case(SOKNL, 3.37, 2.72),
    Case(PassiveAggressiveRegressor, 3.70, 3.68),
    Case(SGDRegressor, 4.63, 3.61),
    Case(ShrubsRegressor, 5.21, 4.76),
]
"""Add your new test cases here ^^"""


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_regressor(case: Case):
    """Update TEST_CASES to add more regressors to test."""
    stream = Fried()
    schema = stream.get_schema()
    learner = case.type(schema=schema, **(case.options or {}))
    evaluator = RegressionEvaluator(schema=schema)
    win_evaluator = RegressionWindowedEvaluator(schema=schema, window_size=100)

    i = 0
    while stream.has_more_instances():
        i += 1
        if i > 1000:
            break
        instance = stream.next_instance()
        prediction = learner.predict(instance)
        evaluator.update(instance.y_value, prediction)
        win_evaluator.update(instance.y_value, prediction)
        learner.train(instance)

    # Check that the model's accuracy is as expected.
    actual = {"rmse": evaluator.rmse(), "win_rmse": win_evaluator.rmse()[-1]}
    expected = {"rmse": case.rmse, "win_rmse": case.win_rmse}
    assert actual == pytest.approx(expected, abs=0.01)

    # Test that the model can be saved and loaded.
    subtest_save_and_load(learner, stream, True)


def test_none_predict():
    """Test that a prediction of None is handled."""
    schema = Schema.from_custom(
        feature_names=["x"], target_attribute_name="y", target_type="numeric"
    )
    evaluator = RegressionEvaluator(schema=schema)
    win_evaluator = RegressionWindowedEvaluator(schema=schema, window_size=100)
    evaluator.update(1.0, None)
    win_evaluator.update(1.0, None)


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
