from typing import List, Callable
from capymoa.base import Classifier
from capymoa.stream import Schema
from dataclasses import dataclass
import pytest
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.evaluation import ocl_train_eval_loop
from functools import partial

from capymoa.classifier import HoeffdingTree

approx = partial(pytest.approx, abs=0.001)


@dataclass(frozen=True)
class Case:
    """Define a test case for OCL classifiers."""

    name: str
    constructor: Callable[[Schema], Classifier]
    accuracy_final: float
    anytime_accuracy_all_avg: float
    ttt_accuracy: float


"""
Add new test cases here.

Use the `partial` function to create a new function with hyperparameters already
set.
"""
TEST_CASES: List[Case] = [
    Case("HoeffdingTree", HoeffdingTree, 0.690, 0.465, 58.5),
]


@pytest.mark.parametrize("case", TEST_CASES, ids=[test.name for test in TEST_CASES])
def test_ocl_classifier(case: Case):
    scenario = TinySplitMNIST()
    learner = case.constructor(scenario.schema)
    result = ocl_train_eval_loop(learner, scenario.train_streams, scenario.test_streams)

    assert result.accuracy_final == approx(case.accuracy_final)
    assert result.anytime_accuracy_all_avg == approx(case.anytime_accuracy_all_avg)
    assert result.ttt.accuracy() == approx(case.ttt_accuracy)
