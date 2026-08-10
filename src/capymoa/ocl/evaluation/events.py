"""Event definitions for OCL evaluation loops.

Events emit in the following order:

..  code-block:: text

    TrainBegin (once)
        TrainTaskBegin (for number of tasks)
            TrainBatchPredict
                TestBegin
                TestTaskBegin (for number of tasks)
                    EvalBatchPredict
                TestTaskEnd
            TestEnd
        TrainTaskEnd
    TrainEnd
"""

from capymoa.ocl.events import Event
from dataclasses import dataclass
from torch import Tensor


@dataclass
class TrainBegin(Event):
    """On training start."""


@dataclass
class TrainTaskBegin(Event):
    """On training start for a task."""

    train_task: int
    """The ID of the training task that has begun."""
    global_step: int
    """A monotonically increasing integer that counts training and evaluation steps."""


@dataclass
class TrainBatchPredict(TrainTaskBegin):
    """After predicting on a training batch, but before training on it."""

    batch: int
    """Batch ID within the current task stream."""
    x: Tensor
    """The input batch."""
    y: Tensor
    """The target batch."""
    y_hat: Tensor
    """The predicted batch."""


class TrainTaskEnd(TrainTaskBegin):
    """On training end for a task."""


class TestBegin(Event):
    """On evaluation start."""


@dataclass
class TestTaskBegin(Event):
    """On evaluation start for a task."""

    train_task: int
    """The ID of the training task that is being evaluated."""
    test_task: int
    """The ID of the test task that has begun."""
    continual_eval: int
    """If multiple evaluations are performed during each training task, this counts
    which evaluation pass is being performed. Otherwise will be 0."""
    global_step: int
    """A monotonically increasing integer that counts training and evaluation steps."""


@dataclass
class EvalBatchPredict(TestTaskBegin):
    """After predicting on an evaluation batch, but before any updates."""

    batch: int
    """Batch ID within the current task stream."""
    x: Tensor
    """The input batch."""
    y: Tensor
    """The target batch."""
    y_hat: Tensor
    """The predicted batch."""


class TestTaskEnd(TestTaskBegin):
    """On evaluation end for a task."""


class TestEnd(Event):
    """On evaluation end."""


class TrainEnd(Event):
    """On training end."""
