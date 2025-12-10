from dataclasses import dataclass, asdict
from functools import partial
from typing import Callable, List
import os

import numpy as np
import pytest

from capymoa.ann import Perceptron
from capymoa.base import Classifier
from capymoa.classifier import Finetune, HoeffdingTree
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.evaluation import ocl_train_eval_loop
from capymoa.ocl.strategy import ExperienceReplay, SLDA, NCM, GDumb, RAR
from capymoa.stream import Schema

import torch
from torch import nn

# PyTorch is notorious for non-deterministic behavior between versions and platforms.
# Here we set a fixed absolute tolerance of +-1.5 for percentage-based metrics.
approx = partial(pytest.approx, abs=1.5)


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
    epochs: int = 1
    continual_evaluations: int = 1


def pre_processor() -> nn.Module:
    """Create a pre-processor for the schema."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
    )


def _new_rar(schema):
    # RAR test case constructor
    return RAR(
        Finetune(schema, Perceptron(schema)),
        augment=nn.Dropout(p=0.2),
        repeats=2,
    )


"""
Add new test cases here.

Use the `partial` function to create a new function with hyperparameters already
set.
"""
TEST_CASES: List[Case] = [
    Case("HoeffdingTree", HoeffdingTree, Result(59.49, 42.59, 45.8), batch_size=1),
    Case("HoeffdingTree", HoeffdingTree, Result(59.00, 42.80, 42.5), batch_size=32),
    Case("RAR", _new_rar, Result(41.50, 28.20, 8.20)),
    Case(
        "Finetune",
        partial(Finetune, model=Perceptron),
        Result(27.00, 24.70, 8.3),
        epochs=2,
        continual_evaluations=2,
    ),
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
    Case(
        "GDumb",
        lambda schema: GDumb(schema, Perceptron(schema), 2, 32, 200),
        Result(38.5, 26.6, 0.0),
    ),
]


@pytest.mark.parametrize("case", TEST_CASES, ids=[test.name for test in TEST_CASES])
def test_ocl_classifier(case: Case):
    if os.environ.get("CI") == "true" and "SLDA" in case.name:
        pytest.skip("Skipping SLDA case on CI due to unreliable dataset download")
    scenario = TinySplitMNIST()

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    learner = case.constructor(scenario.schema)
    r = ocl_train_eval_loop(
        learner,
        scenario.train_loaders(case.batch_size),
        scenario.test_loaders(case.batch_size),
        epochs=case.epochs,
    )
    actual = Result(
        r.accuracy_final * 100,
        r.anytime_accuracy_all_avg * 100,
        r.ttt.accuracy(),
    )
    assert asdict(actual) == approx(asdict(case.expected)), f"Case {case.name} failed."

    def assert_ndarray(value: np.ndarray, shape: tuple, dtype: type = np.float32):
        assert isinstance(value, np.ndarray), (
            f"Expected np.ndarray but got {type(value)}"
        )
        assert value.shape == shape, f"Expected shape {shape} but got {value.shape}"

    def assert_float(value: float):
        assert isinstance(value, float), f"Expected float but got {type(value)}"

    total_eval = r.n_continual_evaluations * r.n_tasks
    assert_float(r.accuracy_all_avg)
    assert_float(r.accuracy_final)
    assert_float(r.accuracy_seen_avg)
    assert_float(r.anytime_accuracy_all_avg)
    assert_float(r.anytime_accuracy_seen_avg)
    assert_float(r.backward_transfer)
    assert_float(r.forward_transfer)
    assert_ndarray(r.accuracy_all, (r.n_tasks,))
    assert_ndarray(r.accuracy_matrix, (r.n_tasks, r.n_tasks))
    assert_ndarray(r.accuracy_seen, (r.n_tasks,))
    assert_ndarray(r.anytime_accuracy_all, (total_eval,))
    assert_ndarray(r.anytime_accuracy_matrix, (total_eval, r.n_tasks))
    assert_ndarray(r.anytime_task_index, (total_eval,), dtype=np.integer)
    assert_ndarray(r.boundaries, (r.n_tasks + 1,), dtype=np.integer)
    assert_ndarray(r.class_cm, (r.n_tasks, r.n_classes, r.n_classes), dtype=np.integer)
    assert_ndarray(r.task_index, (r.n_tasks,), dtype=np.integer)
