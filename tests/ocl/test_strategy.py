from dataclasses import dataclass, asdict
from functools import partial
from typing import Callable, List
import os

import pytest

from capymoa.ann import Perceptron
from capymoa.base import Classifier
from capymoa.classifier import Finetune, HoeffdingTree
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.evaluation import ocl_train_eval_loop
from capymoa.ocl.strategy import ExperienceReplay, SLDA, NCM
from capymoa.stream import Schema

import torch
from torch import nn

approx = partial(pytest.approx, abs=0.1)


@dataclass(frozen=True)
class Result:
    accuracy_final: float
    anytime_accuracy_all_avg: float
    ttt_accuracy: float


@dataclass(frozen=True)
class Case:
    name: str
    constructor: Callable[[Schema], Classifier]
    expected: Result
    batch_size: int = 32


def pre_processor() -> nn.Module:
    """Create a pre-processor for the schema."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
    )


"""
Add new test cases here.

Use the `partial` function to create a new function with hyperparameters already
set.
"""
TEST_CASES: List[Case] = [
    Case("HoeffdingTree", HoeffdingTree, Result(69.0, 46.5, 57.0), batch_size=1),
    Case("HoeffdingTree", HoeffdingTree, Result(69.0, 46.5, 51.8), batch_size=32),
    Case("Finetune", partial(Finetune, model=Perceptron), Result(30.5, 20.7, 2.9)),
    Case("SLDA", SLDA, Result(75.5, 48.70, 74.2)),
    Case(
        "SLDA_with_preprocessor",
        lambda s: SLDA(s, pre_processor(), 512),
        Result(83.5, 52.7, 75.8),
    ),
    Case("NCM", NCM, Result(71.5, 46.2, 67.5)),
    Case(
        "NCM_with_preprocessor",
        lambda s: NCM(s, pre_processor(), 512),
        Result(69.5, 45.0, 66.8),
    ),
    Case(
        "ExperienceReplay",
        lambda schema: ExperienceReplay(Finetune(schema, Perceptron)),
        Result(30.0, 20.1, 3.0),
    ),
]


@pytest.mark.parametrize("case", TEST_CASES, ids=[test.name for test in TEST_CASES])
def test_ocl_classifier(case: Case):
    if os.environ.get("CI") == "true" and "SLDA" in case.name:
        pytest.skip("Skipping SLDA case on CI due to unreliable dataset download")
    scenario = TinySplitMNIST()
    learner = case.constructor(scenario.schema)
    result = ocl_train_eval_loop(
        learner,
        scenario.train_loaders(case.batch_size),
        scenario.test_loaders(case.batch_size),
    )
    actual = Result(
        result.accuracy_final * 100,
        result.anytime_accuracy_all_avg * 100,
        result.ttt.accuracy(),
    )
    assert asdict(actual) == approx(asdict(case.expected)), f"Case {case.name} failed."
