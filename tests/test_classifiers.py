from capymoa.evaluation import ClassificationEvaluator, ClassificationWindowedEvaluator
from capymoa.learner.classifier import (
    EFDT,
    HoeffdingTree,
    AdaptiveRandomForest,
    OnlineBagging,
    NaiveBayes,
)
from capymoa.learner import Classifier, MOAClassifier
from capymoa.datasets import ElectricityTiny
import pytest
from functools import partial
from typing import Callable, Optional
from capymoa.learner.learners import _extract_moa_learner_CLI
from capymoa.learner.splitcriteria import InfoGainSplitCriterion

from capymoa.stream.stream import Schema

from capymoa.learner.classifier.sklearn import PassiveAggressiveClassifier


@pytest.mark.parametrize(
    "learner_constructor,accuracy,win_accuracy,cli_string",
    [
        (partial(OnlineBagging, ensemble_size=5), 84.6, 89.0, None),
        (partial(AdaptiveRandomForest), 89.0, 91.0, None),
        (partial(HoeffdingTree), 73.85, 73.0, None),
        (partial(EFDT), 82.7, 82.0, None),
        (
            partial(EFDT, grace_period=10, split_criterion=InfoGainSplitCriterion(0.2)),
            86.2,
            84.0,
            "trees.EFDT -R 200 -m 33554433 -g 10 -s (InfoGainSplitCriterion -f 0.2) -c 0.001 -z -p -l MC",
        ),
        (partial(NaiveBayes), 84.0, 91.0, None),
    ],
    ids=["OnlineBagging", "AdaptiveRandomForest", "HoeffdingTree", "EFDT", "EFDT_gini", "NaiveBayes"],

)
def test_classifiers(
    learner_constructor: Callable[[Schema], Classifier],
    accuracy: float,
    win_accuracy: float,
    cli_string: Optional[str],
):
    """Test on tiny is a fast running simple test to check if a learner's
    accuracy has changed.

    Notice how we use the `partial` function to creates a new function with
    hyperparameters already set. This allows us to use the same test function
    for different learners with different hyperparameters.

    :param learner_constructor: A partially applied constructor for the learner
    :param accuracy: Expected accuracy
    :param win_accuracy: Expected windowed accuracy
    :param cli_string: Expected CLI string for the learner or None
    """
    stream = ElectricityTiny()
    evaluator = ClassificationEvaluator(schema=stream.get_schema())
    win_evaluator = ClassificationWindowedEvaluator(
        schema=stream.get_schema(), window_size=100
    )

    learner: Classifier = learner_constructor(schema=stream.get_schema())
    # learner = learner_constructor(schema=stream.get_schema())

    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = learner.predict(instance)
        evaluator.update(instance.y_index, prediction)
        win_evaluator.update(instance.y_index, prediction)
        learner.train(instance)

    actual_acc = evaluator.accuracy()
    actual_win_acc = win_evaluator.accuracy()
    assert actual_acc == pytest.approx(
        accuracy, abs=0.1
    ), f"Basic Eval: Expected accuracy of {accuracy:0.1f} got {actual_acc: 0.1f}"
    assert actual_win_acc == pytest.approx(
        win_accuracy, abs=0.1
    ), f"Windowed Eval: Expected accuracy of {win_accuracy:0.1f} got {actual_win_acc:0.1f}"

    if isinstance(learner, MOAClassifier) and cli_string is not None:
        cli_str = _extract_moa_learner_CLI(learner).strip("()")
        assert (
            cli_str == cli_string
        ), "CLI does not match expected value"

    # assert evaluator.accuracy() == pytest.approx(accuracy, abs=0.1)
    # assert win_evaluator.accuracy() == pytest.approx(win_accuracy, abs=0.1)
