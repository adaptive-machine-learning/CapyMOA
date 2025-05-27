from dataclasses import dataclass
from functools import partial
from typing import Callable, List

import pytest

from capymoa.ann import Perceptron
from capymoa.base import Classifier
from capymoa.classifier import Finetune, HoeffdingTree
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.evaluation import ocl_train_eval_loop
from capymoa.ocl.strategy import ExperienceReplay
from capymoa.stream import Schema

approx = partial(pytest.approx, abs=0.001)


@dataclass(frozen=True)
class Case:
    """Define a test case for OCL classifiers."""

    name: str
    constructor: Callable[[Schema], Classifier]
    accuracy_final: float = 0
    anytime_accuracy_all_avg: float = 0
    ttt_accuracy: float = 0
    batch_size: int = 32


"""
Add new test cases here.

Use the `partial` function to create a new function with hyperparameters already
set.
"""
TEST_CASES: List[Case] = [
    Case("HoeffdingTree", HoeffdingTree, 0.690, 0.465, 0.570, batch_size=1),
    Case("HoeffdingTree", HoeffdingTree, 0.690, 0.465, 0.518, batch_size=32),
    Case("Finetune", partial(Finetune, model=Perceptron), 0.305, 0.207, 0.029),
    Case(
        "ExperienceReplay",
        lambda schema: ExperienceReplay(Finetune(schema, Perceptron)),
        0.300,
        0.201,
        0.03,
    ),
]


@pytest.mark.parametrize("case", TEST_CASES, ids=[test.name for test in TEST_CASES])
def test_ocl_classifier(case: Case):
    scenario = TinySplitMNIST()
    learner = case.constructor(scenario.schema)
    result = ocl_train_eval_loop(
        learner,
        scenario.train_loaders(case.batch_size),
        scenario.test_loaders(case.batch_size),
    )

    assert result.accuracy_final == approx(case.accuracy_final), (
        f"`accuracy_final` is {result.accuracy_final:.3f}, expected {case.accuracy_final:.3f}"
    )
    assert result.anytime_accuracy_all_avg == approx(case.anytime_accuracy_all_avg), (
        f"`anytime_accuracy_all_avg` is {result.anytime_accuracy_all_avg:.3f}, "
        f"expected {case.anytime_accuracy_all_avg:.3f}"
    )
    ttt_accuracy = result.ttt.accuracy() / 100
    assert ttt_accuracy == approx(case.ttt_accuracy), (
        f"`ttt_accuracy` is {ttt_accuracy:.3f}, expected {case.ttt_accuracy:.3f}"
    )
