"""Event definitions for OCL evaluation loops.

Each event class is a subclass of :class:`~capymoa.events.Event`, and none
subclasses another, so subscribing to :class:`TrainTaskBegin` receives only the
start of each task. Subscribe to :class:`TrainTaskEvent` or :class:`TestTaskEvent`
to receive every event within a task.

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

from dataclasses import dataclass

from torch import Tensor

from capymoa.ocl.events import Event


@dataclass
class TrainBegin(Event):
    """On training start."""


@dataclass
class TrainTaskEvent(Event):
    """Base class for events during a training task.

    Subscribing to it receives :class:`TrainTaskBegin`,
    :class:`TrainBatchPredict` and :class:`TrainTaskEnd`.
    """

    train_task: int
    """The ID of the current training task."""
    global_step: int
    """A monotonically increasing integer that counts training and evaluation steps."""


@dataclass
class TrainTaskBegin(TrainTaskEvent):
    """On training start for a task."""


@dataclass
class TrainBatchPredict(TrainTaskEvent):
    """After predicting on a training batch, but before training on it."""

    batch: int
    """Batch ID within the current task stream."""
    x: Tensor
    """The input batch."""
    y: Tensor
    """The target batch."""
    y_hat: Tensor
    """The predicted batch."""


class TrainTaskEnd(TrainTaskEvent):
    """On training end for a task."""


class TestBegin(Event):
    """On evaluation start."""


@dataclass
class TestTaskEvent(Event):
    """Base class for events during an evaluation task.

    Subscribing to it receives :class:`TestTaskBegin`,
    :class:`EvalBatchPredict` and :class:`TestTaskEnd`.
    """

    train_task: int
    """The ID of the training task that is being evaluated."""
    test_task: int
    """The ID of the current test task."""
    continual_eval: int
    """If multiple evaluations are performed during each training task, this counts
    which evaluation pass is being performed. Otherwise will be 0."""
    global_step: int
    """A monotonically increasing integer that counts training and evaluation steps."""


@dataclass
class TestTaskBegin(TestTaskEvent):
    """On evaluation start for a task."""


@dataclass
class EvalBatchPredict(TestTaskEvent):
    """After predicting on an evaluation batch, but before any updates."""

    batch: int
    """Batch ID within the current task stream."""
    x: Tensor
    """The input batch."""
    y: Tensor
    """The target batch."""
    y_hat: Tensor
    """The predicted batch."""


class TestTaskEnd(TestTaskEvent):
    """On evaluation end for a task."""


class TestEnd(Event):
    """On evaluation end."""


class TrainEnd(Event):
    """On training end."""
