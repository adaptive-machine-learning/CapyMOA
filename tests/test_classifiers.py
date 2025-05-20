import os
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from tempfile import TemporaryDirectory
from typing import Callable, Optional

import pytest
import torch
from java.lang import Exception as JException
from pytest_subtests import SubTests

from capymoa.ann import Perceptron
from capymoa.base import Classifier, MOAClassifier, _extract_moa_learner_CLI
from capymoa.classifier import (
    CSMOTE,
    EFDT,
    KNN,
    AdaptiveRandomForestClassifier,
    DynamicWeightedMajority,
    Finetune,
    HoeffdingAdaptiveTree,
    HoeffdingTree,
    LeveragingBagging,
    MajorityClass,
    NaiveBayes,
    NoChange,
    OnlineAdwinBagging,
    OnlineBagging,
    OnlineSmoothBoost,
    OzaBoost,
    PassiveAggressiveClassifier,
    SAMkNN,
    SGDClassifier,
    ShrubsClassifier,
    StreamingGradientBoostedTrees,
    StreamingRandomPatches,
    WeightedkNN,
)
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import ClassificationEvaluator, prequential_evaluation
from capymoa.misc import load_model, save_model
from capymoa.splitcriteria import GiniSplitCriterion
from capymoa.stream import Schema, Stream


@dataclass
class ClassifierTestCase:
    test_name: str
    """A unique name to identify your test case. Usually the name of the learner."""
    learner_constructor: Callable[[Schema], Classifier]
    """A function that returns a new instance of the learner."""
    accuracy: float
    """The expected accuracy of the learner."""
    win_accuracy: float
    """The expected windowed accuracy of the learner."""
    cli_string: Optional[str] = None
    """The expected CLI string of the learner."""
    is_serializable: bool = True
    """Whether the learner is serializable."""
    batch_size: int = 1


"""
Add your test cases here. Each test case is a `ClassifierTestCase` object

Notice how we use the `partial` function to creates a new function with
hyperparameters already set. This allows us to use the same test function
for different learners with different hyperparameters.
"""
test_cases = [
    ClassifierTestCase(
        "OnlineBagging",
        partial(OnlineBagging, ensemble_size=5),
        84.6,
        89.0,
    ),
    ClassifierTestCase(
        "AdaptiveRandomForestClassifier",
        partial(AdaptiveRandomForestClassifier),
        89.0,
        91.0,
    ),
    ClassifierTestCase(
        "HoeffdingTree",
        partial(HoeffdingTree),
        82.65,
        83.0,
    ),
    ClassifierTestCase(
        "EFDT",
        partial(EFDT),
        82.69,
        82.0,
    ),
    ClassifierTestCase(
        "EFDT",
        partial(
            EFDT,
            grace_period=10,
            split_criterion=GiniSplitCriterion(),
            leaf_prediction="NaiveBayes",
        ),
        87.35,
        84.0,
        cli_string="trees.EFDT -R 200 -m 33554433 -g 10 -s GiniSplitCriterion -c 0.001 -z -p -l NB",
    ),
    ClassifierTestCase(
        "NaiveBayes",
        partial(NaiveBayes),
        84.0,
        91.0,
    ),
    ClassifierTestCase(
        "KNN",
        partial(KNN),
        81.6,
        74.0,
    ),
    ClassifierTestCase(
        "PassiveAggressiveClassifier",
        partial(PassiveAggressiveClassifier),
        84.7,
        81.0,
    ),
    ClassifierTestCase(
        "SGDClassifier",
        partial(SGDClassifier),
        84.7,
        83.0,
    ),
    ClassifierTestCase(
        "StreamingGradientBoostedTrees",
        partial(StreamingGradientBoostedTrees),
        88.75,
        88.0,
    ),
    ClassifierTestCase(
        "OzaBoost",
        partial(OzaBoost),
        89.95,
        89.0,
    ),
    ClassifierTestCase(
        "MajorityClass",
        partial(MajorityClass),
        60.199999999999996,
        66.0,
    ),
    ClassifierTestCase(
        "NoChange",
        partial(NoChange),
        85.95,
        81.0,
    ),
    ClassifierTestCase(
        "OnlineSmoothBoost",
        partial(OnlineSmoothBoost),
        87.85,
        90.0,
    ),
    ClassifierTestCase(
        "StreamingRandomPatches",
        partial(StreamingRandomPatches),
        90.2,
        89.0,
        is_serializable=False,
    ),
    ClassifierTestCase(
        "HoeffdingAdaptiveTree",
        partial(HoeffdingAdaptiveTree),
        84.15,
        92.0,
    ),
    ClassifierTestCase(
        "SAMkNN",
        partial(SAMkNN),
        82.65,
        82.0,
    ),
    ClassifierTestCase(
        "DynamicWeightedMajority",
        partial(DynamicWeightedMajority),
        84.05,
        89.0,
    ),
    ClassifierTestCase(
        "CSMOTE",
        partial(CSMOTE),
        80.55,
        79.0,
    ),
    ClassifierTestCase(
        "LeveragingBagging",
        partial(LeveragingBagging),
        86.7,
        91.0,
    ),
    ClassifierTestCase(
        "OnlineAdwinBagging",
        partial(OnlineAdwinBagging),
        85.25,
        92.0,
    ),
    ClassifierTestCase(
        "WeightedkNN",
        partial(WeightedkNN),
        78.15,
        70,
    ),
    ClassifierTestCase("ShrubsClassifier", partial(ShrubsClassifier), 89.6, 91),
    ClassifierTestCase(
        "Finetune",
        partial(
            Finetune,
            model=Perceptron,
            optimizer=partial(torch.optim.Adam, lr=0.001),
        ),
        60.4,
        66.0,
        batch_size=32,
    ),
]


def _score(classifier: Classifier, stream: Stream, limit=100) -> float:
    """Eval without training the classifier."""
    evaluator = ClassificationEvaluator(schema=stream.get_schema())
    stream.restart()
    for i, instance in enumerate(stream):
        if i >= limit:
            break
        prediction = classifier.predict(instance)
        evaluator.update(instance.y_index, prediction)
    return evaluator.accuracy()


def subtest_save_and_load(
    classifier: Classifier,
    stream: Stream,
    is_serializable: bool,
):
    """A subtest to check if a classifier can be saved and loaded."""

    with TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "model.pkl")
        with pytest.raises(JException) if not is_serializable else nullcontext():
            # Save and load the model
            with open(tmp_file, "wb") as f:
                save_model(classifier, f)

            with open(tmp_file, "rb") as f:
                loaded_classifier: Classifier = load_model(f)

            # Check that the saved and loaded model have the same accuracy
            expected_acc = _score(classifier, stream)
            loaded_acc = _score(loaded_classifier, stream)
            assert expected_acc == loaded_acc, (
                f"Original accuracy {expected_acc * 100:.2f} != loaded accuracy {loaded_acc * 100:.2f}"
            )

            # Check that the loaded model can still be trained
            loaded_classifier.train(next(stream))


@pytest.mark.parametrize(
    "test_case",
    test_cases,
    ids=[c.test_name for c in test_cases],
)
def test_classifiers(test_case: ClassifierTestCase, subtests: SubTests):
    """``test_classifiers`` is a fast running complex test that checks:

    * Did the classifier reach the expected accuracy?
    * Did the classifier reach the expected windowed accuracy?
    * Can the classifier be saved and loaded?
    * Does the CLI string match the expected value?
    """
    stream = ElectricityTiny()
    learner: Classifier = test_case.learner_constructor(schema=stream.get_schema())
    results = prequential_evaluation(
        stream, learner, window_size=100, batch_size=test_case.batch_size
    )

    # Check if the accuracy matches the expected value for both evaluator types
    actual_acc = results.cumulative.accuracy()
    actual_win_acc = results.windowed.accuracy()[-1]

    assert actual_acc == pytest.approx(test_case.accuracy, abs=0.1), (
        f"Basic Eval: Expected accuracy of {test_case.accuracy:0.1f} got {actual_acc: 0.1f}"
    )
    assert actual_win_acc == pytest.approx(test_case.win_accuracy, abs=0.1), (
        f"Windowed Eval: Expected accuracy of {test_case.win_accuracy:0.1f} got {actual_win_acc:0.1f}"
    )

    # Check if the classifier can be saved and loaded
    with subtests.test(msg="save_and_load"):
        subtest_save_and_load(learner, stream, test_case.is_serializable)

    # Optionally check the CLI string if it was provided
    if isinstance(learner, MOAClassifier) and test_case.cli_string is not None:
        cli_str = _extract_moa_learner_CLI(learner).strip("()")
        assert cli_str == test_case.cli_string, "CLI does not match expected value"
