from capymoa.evaluation import RegressionEvaluator, RegressionWindowedEvaluator
from capymoa.datasets import Fried
from capymoa.regressor import (
    KNNRegressor,
    AdaptiveRandomForestRegressor,
    FIMTDD,
    ARFFIMTDD,
    ORTO,
    SOKNLBT,
    SOKNL,
    PassiveAggressiveRegressor,
    SGDRegressor
)
import pytest
from functools import partial

from capymoa.base import Regressor
from capymoa.stream import Schema

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
        "SGDRegressor"
    ]
)
def test_regressor(learner_constructor, rmse, win_rmse):
    """Test on tiny is a fast running simple test to check if a learner's
    accuracy has changed.

    Notice how we use the `partial` function to create a new function with
    hyperparameters already set. This allows us to use the same test function
    for different learners with different hyperparameters.
    """
    stream = Fried()
    evaluator = RegressionEvaluator(schema=stream.get_schema())
    win_evaluator = RegressionWindowedEvaluator(schema=stream.get_schema(), window_size=100)
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


    actual_rmse = evaluator.RMSE()
    actual_win_rmse = win_evaluator.RMSE()['windowed RMSE'][-1]
    assert actual_rmse == pytest.approx(rmse, abs=0.1), \
        f"Basic Eval: Expected {rmse:0.1f} RMSE got {actual_rmse: 0.1f} RMSE"
    assert actual_win_rmse == pytest.approx(win_rmse, abs=0.1), \
        f"Windowed Eval: Expected {win_rmse:0.1f} RMSE got {actual_win_rmse:0.1f} RMSE"

def test_none_predict():
    """Test that a prediction of None is handled."""
    schema = Schema.from_custom(feature_names=["x"], target_attribute_name="y", target_type='numeric')
    evaluator = RegressionEvaluator(schema=schema)
    win_evaluator = RegressionWindowedEvaluator(schema=schema, window_size=100)
    evaluator.update(1.0, None)
    win_evaluator.update(1.0, None)
