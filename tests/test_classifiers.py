from capymoa.evaluation import ClassificationEvaluator, ClassificationWindowedEvaluator
from capymoa.classifier import (
    EFDT,
    HoeffdingTree,
    AdaptiveRandomForestClassifier,
    OnlineBagging,
    NaiveBayes,
    KNN,
    StreamingGradientBoostedTrees,
    OzaBoost,
    MajorityClass,
    NoChange,
    OnlineSmoothBoost,
    StreamingRandomPatches,
    HoeffdingAdaptiveTree,
    SAMkNN,
    DynamicWeightedMajority,
    CSMOTE,
    LeveragingBagging,
    OnlineAdwinBagging,
    OnlineAccuracyUpdatedEnsemble,
    RWkNN
)

from capymoa.base import Classifier
from capymoa.base import MOAClassifier
from capymoa.datasets import ElectricityTiny
import pytest
from functools import partial
from typing import Callable, Optional
from capymoa.base import _extract_moa_learner_CLI
from capymoa.splitcriteria import GiniSplitCriterion

from capymoa.stream._stream import Schema

from capymoa.classifier import PassiveAggressiveClassifier, SGDClassifier


@pytest.mark.parametrize(
    "learner_constructor,accuracy,win_accuracy,cli_string",
    [
        (partial(OnlineBagging, ensemble_size=5), 84.6, 89.0, None),
        (partial(AdaptiveRandomForestClassifier), 89.0, 91.0, None),
        (partial(HoeffdingTree), 82.65, 83.0, None),
        (partial(EFDT), 82.69, 82.0, None),
        (
            partial(EFDT, grace_period=10, split_criterion=GiniSplitCriterion(), leaf_prediction="NaiveBayes"),
            87.8,
            85.0,
            "trees.EFDT -R 200 -m 33554433 -g 10 -s GiniSplitCriterion -c 0.001 -z -p -l NB",
        ),
        (partial(NaiveBayes), 84.0, 91.0, None),
        (partial(KNN), 81.6, 74.0, None),
        (partial(PassiveAggressiveClassifier), 84.7, 81.0, None),
        (partial(SGDClassifier), 84.7, 83.0, None),
        (partial(StreamingGradientBoostedTrees), 88.75, 88.0, None),
        (partial(OzaBoost), 89.95, 89.0, None),
        (partial(MajorityClass), 60.199999999999996, 66.0, None),
        (partial(NoChange), 85.95, 81.0, None),
        (partial(OnlineSmoothBoost), 87.85, 90.0, None),
        (partial(StreamingRandomPatches), 90.2, 89.0, None),
        (partial(HoeffdingAdaptiveTree), 84.15, 92.0, None),
        (partial(SAMkNN), 82.65, 82.0, None),
        (partial(DynamicWeightedMajority), 84.05, 89.0, None),
        (partial(CSMOTE), 80.55, 79.0, None),
        (partial(LeveragingBagging), 86.7, 91.0, None),
        (partial(OnlineAdwinBagging), 85.25, 92.0, None),
        (partial(OnlineAccuracyUpdatedEnsemble), 85.25, 92.0, None),
        (partial(RWkNN), 85.25, 92.0, None),
    ],
    ids=[
        "OnlineBagging",
        "AdaptiveRandomForest",
        "HoeffdingTree",
        "EFDT",
        "EFDT_gini",
        "NaiveBayes",
        "KNN",
        "PassiveAggressiveClassifier",
        "SGDClassifier",
        "StreamingGradientBoostedTrees",
        "OzaBoost",
        "MajorityClass",
        "NoChange",
        "OnlineAccuracyUpdatedEnsemble",
        "OnlineSmoothBoost",
        "StreamingRandomPatches",
        "HoeffdingAdaptiveTree",
        "SAMkNN",
        "DynamicWeightedMajority",
        "CSMOTE",
        "LeveragingBagging",
        "OnlineAdwinBagging",
        "RWkNN"
    ],
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

    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = learner.predict(instance)
        evaluator.update(instance.y_index, prediction)
        win_evaluator.update(instance.y_index, prediction)
        learner.train(instance)

    # Check if the accuracy matches the expected value for both evaluator types
    actual_acc = evaluator.accuracy()
    actual_win_acc = win_evaluator.accuracy()
    assert actual_acc == pytest.approx(
        accuracy, abs=0.1
    ), f"Basic Eval: Expected accuracy of {accuracy:0.1f} got {actual_acc: 0.1f}"
    assert actual_win_acc == pytest.approx(
        win_accuracy, abs=0.1
    ), f"Windowed Eval: Expected accuracy of {win_accuracy:0.1f} got {actual_win_acc:0.1f}"

    # Optionally check the CLI string if it was provided
    if isinstance(learner, MOAClassifier) and cli_string is not None:
        cli_str = _extract_moa_learner_CLI(learner).strip("()")
        assert cli_str == cli_string, "CLI does not match expected value"
