from typing import Optional, Sequence, Union

from capymoa.base import Classifier
from capymoa.instance import LabeledInstance
from capymoa.ocl._base import TaskAware, TaskBoundaryAware
from capymoa.stream import Stream

from capymoa.type_alias import LabelIndex
import torch
import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class OCLMetrics:
    """A collection of metrics evaluating an online continual learner."""

    anytime_accuracy: float
    """Anytime Accuracy is the accuracy of the learner on all previous tasks."""
    average_anytime_accuracy: float
    """Average Anytime Accuracy is the average Anytime Accuracy over the model's lifetime."""
    online_accuracy: float
    """Online accuracy evaluates the anytime inference ability of a model."""

    anytime_accuracy_series: np.ndarray
    """The anytime accuracy (see :attr:`anytime_accuracy`) at each task boundary."""
    average_anytime_accuracy_series: np.ndarray
    """The average anytime accuracy (see :attr:`average_anytime_accuracy`) at each task boundary."""
    online_accuracy_series: np.ndarray
    """The online accuracy (see :attr:`online_accuracy`) at each task boundary."""

    task_accuracy_matrix: np.ndarray
    """The matrix containing the accuracy on each task at each task boundary.
    
    1. Rows are the training tasks
    2. Columns are the testing tasks
    """

    class_confusion: np.ndarray
    """The class confusion matrix over all classes at the end of the last task.

    1. Rows are the true class
    2. Columns are the predicted class
    """

    @property
    def accuracy(self) -> float:
        """Alias for :attr:`anytime_accuracy`."""
        return self.anytime_accuracy


class _OCLEvaluator:
    """A builder used to collect statistics during online continual learning evaluation."""

    cm: torch.Tensor
    """Confusion 'Matrix' of shape (task_count, task_count, class_count, class_count)
    
    1. Dimension one is the training task id
    2. Dimension two is the testing task id
    3. Dimension three is the true class
    4. Dimension four is the predicted class
    """

    pcm: torch.Tensor
    """Prequential Confusion 'Matrix' is the confusion matrix for online prequential evaluation.
    """

    def __init__(self, task_count: int, class_count: int):
        self.task_count = task_count
        self.class_count = class_count
        self.seen_tasks = 0
        self.pcm = torch.zeros(task_count, class_count, class_count, dtype=torch.int)
        self.cm = torch.zeros(
            task_count, task_count, class_count, class_count, dtype=torch.int
        )

    def prequential_update(
        self, train_task_id: int, y_true: LabelIndex, y_pred: Optional[LabelIndex]
    ):
        """Record a prediction during online evaluation."""
        self.seen_tasks = max(self.seen_tasks, train_task_id + 1)
        if y_pred is not None:
            self.pcm[train_task_id, y_true, y_pred] += 1
        # TODO: handle missing predictions

    def holdout_update(
        self,
        train_task_id: int,
        test_task_id: int,
        y_true: LabelIndex,
        y_pred: Optional[LabelIndex],
    ):
        """Record a prediction when using holdout evaluation."""
        if y_pred is not None:
            self.cm[train_task_id, test_task_id, y_true, y_pred] += 1
        # TODO: handle missing predictions

    def anytime_accuracy(self, task_id: int) -> float:
        """See :attr:`OCLMetrics.anytime_accuracy`."""
        cm = self.cm[task_id, : task_id + 1].sum(0)
        return float(cm.diag().sum() / cm.sum())

    def average_anytime_accuracy(self, task_id: int) -> float:
        """See :attr:`OCLMetrics.average_anytime_accuracy`."""
        cm = self.cm[: task_id + 1, : task_id + 1].sum(dim=(0, 1))
        return float(cm.diag().sum() / cm.sum())

    def online_accuracy(self, task_id: int) -> float:
        """See :attr:`OCLMetrics.online_accuracy`."""
        cm = self.pcm[: task_id + 1].sum(0)
        return float(cm.diag().sum() / cm.sum())

    def task_accuracy(self, train_task_id: int, test_task_id: int) -> float:
        """See :attr:`OCLMetrics.task_accuracy_matrix`."""
        return float(
            self.cm[train_task_id, test_task_id].diag().sum()
            / self.cm[train_task_id, test_task_id].sum()
        )

    # TODO: Add backwards transfer and forward transfer metrics

    def build(self) -> OCLMetrics:
        """Creates metrics using collected statistics."""
        assert self.seen_tasks == self.task_count, (
            "All tasks must be seen before building metrics"
        )
        tasks = np.arange(self.task_count)
        task_id = self.task_count - 1

        class_cm = self.cm[task_id].sum(0).numpy()

        task_accuracy_matrix = np.zeros((self.task_count, self.task_count))
        for i, j in np.ndindex(task_accuracy_matrix.shape):
            task_accuracy_matrix[i, j] = self.task_accuracy(i, j)

        return OCLMetrics(
            anytime_accuracy=self.anytime_accuracy(task_id),
            average_anytime_accuracy=self.average_anytime_accuracy(task_id),
            online_accuracy=self.online_accuracy(task_id),
            anytime_accuracy_series=np.vectorize(self.anytime_accuracy)(tasks),
            average_anytime_accuracy_series=np.vectorize(self.average_anytime_accuracy)(
                tasks
            ),
            online_accuracy_series=np.vectorize(self.online_accuracy)(tasks),
            class_confusion=class_cm / class_cm.sum(1),
            task_accuracy_matrix=task_accuracy_matrix,
        )


@torch.no_grad()
def ocl_train_eval_loop(
    learner: Union[TaskBoundaryAware, TaskAware, Classifier],
    train_streams: Sequence[Stream[LabeledInstance]],
    test_streams: Sequence[Stream[LabeledInstance]],
) -> OCLMetrics:
    """Train and evaluate a learner on a sequence of tasks.

    :param learner: A classifier that is possibly task-aware or task-boundary-aware.
    :param train_streams: A sequence of streams containing the training tasks.
    :param test_streams: A sequence of streams containing the testing tasks.
    :return: A collection of metrics evaluating the learner's performance.
    """

    n_tasks = len(train_streams)
    if n_tasks != len(test_streams):
        raise ValueError("Number of train and test tasks must be equal")
    if not isinstance(learner, Classifier):
        raise ValueError("Learner must be a classifier")

    metrics = _OCLEvaluator(n_tasks, learner.schema.get_num_classes())

    for train_task_id, train_stream in enumerate(train_streams):
        train_stream.restart()
        assert len(train_stream)

        if isinstance(learner, TaskBoundaryAware):
            learner.set_train_task(train_task_id)

        # train and evaluation loop
        with torch.enable_grad():
            for instance in train_stream:
                y_pred = learner.predict(instance)
                metrics.prequential_update(train_task_id, instance.y_index, y_pred)
                learner.train(instance)

        # evaluate the learner on past and future tasks
        for test_task_id, test_stream in enumerate(test_streams):
            test_stream.restart()
            assert len(test_stream)

            if isinstance(learner, TaskAware):
                learner.set_test_task(test_task_id)

            # predict instances in the current task
            for instance in test_stream:
                y_pred = learner.predict(instance)
                metrics.holdout_update(
                    train_task_id, test_task_id, instance.y_index, y_pred
                )

    return metrics.build()
