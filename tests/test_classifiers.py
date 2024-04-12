from capymoa.evaluation import ClassificationEvaluator
from capymoa.evaluation.evaluation import ClassificationWindowedEvaluator
from capymoa.learner.classifier.classifiers import AdaptiveRandomForest, OnlineBagging
from capymoa.learner.classifier import EFDT, HoeffdingTree, NaiveBayes
from capymoa.datasets import ElectricityTiny
import pytest
from functools import partial


@pytest.mark.parametrize(
    "learner_constructor,accuracy,win_accuracy",
    [
        (partial(OnlineBagging, ensemble_size=5), 84.6, 89.0),
        (partial(AdaptiveRandomForest), 89.6, 91.0),
        (partial(HoeffdingTree), 73.85, 73.0),
        (partial(EFDT), 82.7, 82.0),
        (partial(NaiveBayes), 84.0, 91.0)
    ],
    ids=[
        "OnlineBagging",
        "AdaptiveRandomForest",
        "HoeffdingTree",
        "EFDT",
        "NaiveBayes"
    ]
)
def test_on_tiny(learner_constructor, accuracy, win_accuracy):
    """Test on tiny is a fast running simple test to check if a learner's
    accuracy has changed.

    Notice how we use the `partial` function to create a new function with
    hyperparameters already set. This allows us to use the same test function
    for different learners with different hyperparameters.
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

    assert evaluator.accuracy() == pytest.approx(accuracy, abs=0.1)
    assert win_evaluator.accuracy() == pytest.approx(win_accuracy, abs=0.1)

