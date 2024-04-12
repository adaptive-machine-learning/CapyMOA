from capymoa.evaluation import ClassificationEvaluator, ClassificationWindowedEvaluator
from capymoa.learner.classifier import EFDT, HoeffdingTree, AdaptiveRandomForest, OnlineBagging
from capymoa.learner import Classifier
from capymoa.datasets import ElectricityTiny
import pytest
from functools import partial
from typing import Callable

from capymoa.stream.stream import Schema

@pytest.mark.parametrize(
    "learner_constructor,accuracy,win_accuracy",
    [
        (partial(OnlineBagging, ensemble_size=5), 84.6, 89.0),
        (partial(AdaptiveRandomForest), 89.0, 91.0),
        (partial(HoeffdingTree), 73.85, 73.0),
        (partial(EFDT), 82.7, 82.0)
    ],
    ids=[
        "OnlineBagging",
        "AdaptiveRandomForest",
        "HoeffdingTree",
        "EFDT"
    ]
)
def test_classifiers(learner_constructor: Callable[[Schema], Classifier], accuracy: float, win_accuracy: float):
    """Test on tiny is a fast running simple test to check if a learner's
    accuracy has changed.

    Notice how we use the `partial` function to creates a new function with
    hyperparameters already set. This allows us to use the same test function
    for different learners with different hyperparameters.

    :param learner_constructor: A partially applied constructor for the learner
    :param accuracy: Expected accuracy
    :param win_accuracy: Expected windowed accuracy
    """    
    stream = ElectricityTiny()
    evaluator = ClassificationEvaluator(schema=stream.get_schema())
    win_evaluator = ClassificationWindowedEvaluator(schema=stream.get_schema(), window_size=100)
    learner = learner_constructor(schema=stream.get_schema())

    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = learner.predict(instance)
        evaluator.update(instance.y_index, prediction)
        win_evaluator.update(instance.y_index, prediction)
        learner.train(instance)

    actual_acc = evaluator.accuracy()
    actual_win_acc = win_evaluator.accuracy()
    assert actual_acc == pytest.approx(accuracy, abs=0.1), \
        f"Basic Eval: Expected accuracy of {accuracy:0.1f} got {actual_acc: 0.1f}"
    assert actual_win_acc == pytest.approx(win_accuracy, abs=0.1), \
        f"Windowed Eval: Expected accuracy of {win_accuracy:0.1f} got {actual_win_acc:0.1f}"