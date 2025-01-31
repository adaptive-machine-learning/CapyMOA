from contextlib import nullcontext
import os
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
)
from jpype import JException
import pytest
from functools import partial

from capymoa.base import Regressor
from capymoa.stream import Schema, Stream
from tempfile import TemporaryDirectory

from pytest_subtests import SubTests


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


@pytest.mark.parametrize(
    "learner_constructor,rmse,win_rmse",
    [
        (partial(AdaptiveRandomForestRegressor), 3.66, 3.00),
        (partial(KNNRegressor), 3.03, 2.49),
        (partial(FIMTDD), 7.4, 5.3),
        (partial(ARFFIMTDD), 4.95, 4.6),
        (partial(ORTO), 9.2, 7.6),
        (partial(SOKNLBT), 4.95, 4.6),
        (partial(SOKNL), 3.37, 2.77),
        (partial(PassiveAggressiveRegressor), 3.67, 3.68),
        (partial(SGDRegressor), 4.63, 3.6),
        (partial(ShrubsRegressor), 5.12, 4.75),
    ],
    ids=[
        "AdaptiveRandomForestRegressor",
        "KNNRegressor",
        "FIMTDD",
        "ARFFIMTDD",
        "ORTO",
        "SOKNLBT",
        "SOKNL",
        "PassiveAggressiveRegressor",
        "SGDRegressor",
        "ShrubsRegressor",
    ],
)
def test_regressor(subtests: SubTests, learner_constructor, rmse, win_rmse):
    """Test on tiny is a fast running simple test to check if a learner's
    accuracy has changed.

    Notice how we use the `partial` function to create a new function with
    hyperparameters already set. This allows us to use the same test function
    for different learners with different hyperparameters.
    """
    stream = Fried()
    evaluator = RegressionEvaluator(schema=stream.get_schema())
    win_evaluator = RegressionWindowedEvaluator(
        schema=stream.get_schema(), window_size=100
    )
    learner: Regressor = learner_constructor(schema=stream.get_schema())

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

    actual_rmse = evaluator.rmse()
    actual_win_rmse = win_evaluator.rmse()[-1]
    assert actual_rmse == pytest.approx(rmse, abs=0.1), (
        f"Basic Eval: Expected {rmse:0.1f} RMSE got {actual_rmse: 0.1f} RMSE"
    )
    assert actual_win_rmse == pytest.approx(win_rmse, abs=0.1), (
        f"Windowed Eval: Expected {win_rmse:0.1f} RMSE got {actual_win_rmse:0.1f} RMSE"
    )

    with subtests.test(msg="save_and_load"):
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
