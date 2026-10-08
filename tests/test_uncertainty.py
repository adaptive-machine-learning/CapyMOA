from functools import partial
from itertools import islice

import pytest

from capymoa.base import PredictionIntervalLearner
from capymoa.datasets import Fried
from capymoa.regressor.evaluate import RegressionEvaluator
from capymoa.uncertainty import (
    MVE,
    AdaPI,
    evaluate_prediction_interval,
)
from capymoa.uncertainty.evaluate import (
    PredictionIntervalEvaluator,
    PredictionIntervalWindowedEvaluator,
)


@pytest.mark.parametrize(
    "learner_constructor,coverage,win_coverage",
    [
        (partial(MVE), 97.8, 98.0),
        (partial(AdaPI), 96.2, 97.0),
    ],
    ids=[
        "MVE",
        "AdaPI",
    ],
)
def test_PI(learner_constructor, coverage, win_coverage):
    """Test on tiny is a fast running simple test to check if a learner's
    accuracy has changed.

    Notice how we use the `partial` function to create a new function with
    hyperparameters already set. This allows us to use the same test function
    for different learners with different hyperparameters.
    """
    stream = Fried()
    evaluator = PredictionIntervalEvaluator(schema=stream.get_schema())
    win_evaluator = PredictionIntervalWindowedEvaluator(
        schema=stream.get_schema(), window_size=100
    )
    learner: PredictionIntervalLearner = learner_constructor(schema=stream.get_schema())

    i = 0
    while stream.has_more_instances():
        i += 1
        if i >= 1000:
            break
        instance = stream.next_instance()
        prediction = learner.predict(instance)
        evaluator.update(instance.y_value, prediction)
        win_evaluator.update(instance.y_value, prediction)
        learner.train(instance)

    actual_coverage = evaluator.coverage()
    actual_win_coverage = win_evaluator.coverage()[-1]
    assert actual_coverage == pytest.approx(coverage, abs=0.1), (
        f"Basic Eval: Expected {coverage:0.1f} coverage got {actual_coverage: 0.1f} coverage"
    )
    assert actual_win_coverage == pytest.approx(win_coverage, abs=0.1), (
        f"Windowed Eval: Expected {win_coverage:0.1f} coverage got {actual_win_coverage:0.1f} coverage"
    )


def test_interval_evaluator_scores_the_point():
    """The regression metrics of an interval evaluator score the middle value."""
    stream = Fried()
    schema = stream.get_schema()
    evaluator = PredictionIntervalEvaluator(schema=schema)
    point = RegressionEvaluator(schema=schema)
    learner = MVE(schema=schema)
    for instance in islice(stream, 500):
        prediction = learner.predict(instance)
        evaluator.update(instance.y_value, prediction)
        point.update(instance.y_value, prediction[1])
        learner.train(instance)
    assert evaluator.mae() == pytest.approx(point.mae())
    assert evaluator.rmse() == pytest.approx(point.rmse())

    results = evaluate_prediction_interval(
        stream, MVE(schema=schema), max_instances=500
    )
    assert results["mae"] == pytest.approx(evaluator.mae())
