from typing import Optional, Sequence, Union

from capymoa.base import Classifier
from capymoa.instance import LabeledInstance
from capymoa.ocl._base import TaskAware, TaskBoundaryAware
from capymoa.stream import Stream

from capymoa.type_alias import LabelIndex
import torch
import numpy as np
from tqdm.auto import tqdm

from dataclasses import dataclass


@dataclass(frozen=True)
class OCLMetrics:
    r"""A collection of metrics evaluating an online continual learner.

    We define some metrics in terms of a matrix :math:`R\in\mathbb{R}^{T \times T}`
    (:attr:`accuracy_matrix`) where each element :math:`R_{i,j}` contains the
    the test accuracy on task :math:`j` after sequentially training on tasks
    :math:`1` through :math:`i`.

    Online learning make predictions continuously during training, so we also
    provide "anytime" versions of the metrics. These metrics are collected
    periodically during training. Specifically, :math:`H` times per task.
    The results of this evaluation are stored in a matrix
    :math:`A\in\mathbb{R}^{T \times H \times T}` (:attr:`anytime_accuracy_matrix`)
    where each element :math:`A_{i,h,j}` contains the test accuracy on task
    :math:`j` after sequentially training on tasks :math:`1` through :math:`i-1`
    and step :math:`h` of task :math:`i`.
    """

    anytime_accuracy_all: np.ndarray
    r"""The accuracy on all tasks after training on each step in each task.

    .. math::
    
        a_\text{any all}(t, h) = \frac{1}{T}\sum^T_{i=1} A_{t,h,i}

    We flatten the $t,h$ dimensions to a 1D array. Use
    :attr:`anytime_task_index` to get the corresponding task index for plotting.
    """
    anytime_accuracy_all_avg: float
    r"""The average of :attr:`anytime_accuracy_all` over all tasks.

    .. math::
    
        \bar{a}_\text{any all} = \frac{1}{T}\sum_{t=1}^T \frac{1}{H}\sum_{h=1}^H a_\text{any all}(t, h)

    """
    anytime_accuracy_seen: np.ndarray
    r"""The accuracy on **seen** tasks after training on each step in each task.
    
    .. math::
    
        a_\text{any seen}(t, h) = \frac{1}{t}\sum^t_{i=1} A_{t,h,i}

    We flatten the $t,h$ dimensions to a 1D array. Use
    :attr:`anytime_task_index` to get the corresponding task index for plotting.
    """
    anytime_accuracy_seen_avg: float
    r"""The average of :attr:`anytime_accuracy_seen` over all tasks.

    .. math::
    
        \bar{a}_\text{any seen} = \frac{1}{T}\sum_{t=1}^T \frac{1}{H}\sum_{h=1}^H a_\text{any seen}(t, h)
    """
    anytime_task_index: np.ndarray
    r"""The position in each task where the anytime accuracy was measured."""

    accuracy_all: np.ndarray
    r"""The accuracy on all tasks after training on each task.
    
    .. math::

        a_\text{all}(t) = \frac{1}{T} \sum_{i=1}^{T} R_{t,i}

    Use :attr:`task_index` to get the corresponding task index for plotting.
    """
    accuracy_all_avg: float
    r"""The average of :attr:`accuracy_all` over all tasks.
    
    .. math::

        \bar{a}_\text{all} = \frac{1}{T}\sum_{t=1}^T a_\text{all}(t)
    """
    accuracy_seen: np.ndarray
    r"""The accuracy on **seen** tasks after training on each task.
    
    .. math::

        a_\text{seen}(t) = \frac{1}{t}\sum^t_{i=1} R_{t,i}

    Use :attr:`task_index` to get the corresponding task index for plotting.
    """
    accuracy_seen_avg: float
    r"""The average of :attr:`accuracy_seen` over all tasks.
    
    .. math::

        \bar{a}_\text{seen} = \frac{1}{T}\sum_{t=1}^T a_\text{seen}(t)
    """
    accuracy_final: float
    r"""The accuracy on the final task after training on all tasks.

    .. math::
    
        a_\text{final} = a_\text{all}(T)
    """
    task_index: np.ndarray
    r"""The position of each task in the metrics."""

    forward_transfer: float
    r"""A scalar measuring the impact learning had on future tasks.

    .. math::

       r_\text{FWT} = \frac{2}{T(T-1)}\sum_{i=1}^{T} \sum_{j=i+1}^{T} R_{i,j}
    """
    backward_transfer: float
    r"""A scalar measuring the impact learning had on past tasks.

    .. math::

       r_\text{BWT} = \frac{2}{T(T-1)} \sum_{i=2}^{T} \sum_{j=1}^{i-1} (R_{i,j} - R_{j,j})
    """
    accuracy_matrix: np.ndarray
    r"""A matrix measuring the accuracy on each task after training on each task.
    
    ``R[i, j]`` is the accuracy on task :math:`j` after training on tasks
    :math:`1` through :math:`i`.
    """

    anytime_accuracy_matrix: np.ndarray
    r"""A matrix measuring the accuracy on each task after training on each task and step.
    
    This matrix is :math:`A` with the first two dimensions flattened to a 2D array.
    """

    prequential_cumulative_accuracy: float
    """The average test-then-train accuracy over the entire data-stream."""


def _backwards_transfer(R: torch.Tensor) -> float:
    n = R.size(0)
    assert R.shape == (n, n)
    return ((R - R.diag()).tril().sum() / (n * (n - 1) / 2)).item()


def _forwards_transfer(R: torch.Tensor) -> float:
    n = R.size(0)
    assert R.shape == (n, n)
    return (R.triu(1).sum() / (n * (n - 1) / 2)).item()


class _OCLEvaluator:
    """A builder used to collect statistics during online continual learning evaluation."""

    cm: torch.Tensor
    """Confusion 'Matrix' of shape: 
    ``(eval_step_id, train_task_id, test_task_id, true_class, predicted_class)``.
    """

    pcm: torch.Tensor
    """Prequential Confusion 'Matrix' is the confusion matrix for online prequential evaluation.
    """

    def __init__(self, task_count: int, eval_step_count: int, class_count: int):
        self.task_count = task_count
        self.class_count = class_count
        self.seen_tasks = 0
        self.step_count = eval_step_count
        self.pcm = torch.zeros(task_count, class_count, class_count, dtype=torch.int)
        self.cm = torch.zeros(
            task_count,
            eval_step_count,
            task_count,
            class_count,
            class_count,
            dtype=torch.int,
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
        eval_step_id: int,
        test_task_id: int,
        y_true: LabelIndex,
        y_pred: Optional[LabelIndex],
    ):
        """Record a prediction when using holdout evaluation."""
        if y_pred is not None:
            self.cm[train_task_id, eval_step_id, test_task_id, y_true, y_pred] += 1
        # TODO: handle missing predictions

    def build(self) -> OCLMetrics:
        """Creates metrics using collected statistics."""
        correct = self.cm.diagonal(dim1=3, dim2=4).sum(-1)
        total = self.cm.sum((3, 4))
        anytime_acc = correct / total  # (train task, step_id, test task)
        accuracy_matrix = anytime_acc[:, -1, :]  # (train task, test task)

        anytime_accuracy_seen = torch.zeros(self.task_count, self.step_count)
        anytime_accuracy_all = torch.zeros(self.task_count, self.step_count)
        for t_train in range(self.task_count):
            for s_step in range(self.step_count):
                anytime_accuracy_seen[t_train, s_step] = (
                    anytime_acc[t_train, s_step, : t_train + 1].mean().item()
                )
                anytime_accuracy_all[t_train, s_step] = (
                    anytime_acc[t_train, s_step, :].mean().item()
                )

        def _accuracy_seen(t: int) -> float:
            return accuracy_matrix[t, : t + 1].mean().item()

        def _accuracy_all(t: int) -> float:
            return accuracy_matrix[t, :].mean().item()

        tasks = np.arange(self.task_count, dtype=int)

        accuracy_seen = np.vectorize(_accuracy_seen)(tasks)
        accuracy_all = np.vectorize(_accuracy_all)(tasks)

        online_accuracy = (
            self.pcm.diagonal(dim1=1, dim2=2).sum().item() / self.pcm.sum()
        )

        return OCLMetrics(
            accuracy_seen=accuracy_seen,
            accuracy_all=accuracy_all,
            accuracy_final=_accuracy_all(self.task_count - 1),
            accuracy_all_avg=np.mean(accuracy_all),
            accuracy_seen_avg=np.mean(accuracy_seen),
            accuracy_matrix=accuracy_matrix.numpy(),
            anytime_accuracy_all=anytime_accuracy_all.flatten().numpy(),
            anytime_accuracy_seen=anytime_accuracy_seen.flatten().numpy(),
            anytime_accuracy_all_avg=anytime_accuracy_all.mean().item(),
            anytime_accuracy_seen_avg=anytime_accuracy_seen.mean().item(),
            anytime_task_index=np.linspace(
                0, self.task_count, self.step_count * self.task_count + 1
            )[1:],
            task_index=np.arange(self.task_count) + 1,
            anytime_accuracy_matrix=anytime_acc.flatten(end_dim=1).numpy(),
            backward_transfer=_backwards_transfer(accuracy_matrix),
            forward_transfer=_forwards_transfer(accuracy_matrix),
            prequential_cumulative_accuracy=online_accuracy,
        )


_OCLClassifier = Union[TaskBoundaryAware, TaskAware, Classifier]


def ocl_train_eval_loop(
    learner: _OCLClassifier,
    train_streams: Sequence[Stream[LabeledInstance]],
    test_streams: Sequence[Stream[LabeledInstance]],
    continual_evaluations: int = 1,
    progress_bar: bool = False,
) -> OCLMetrics:
    """Train and evaluate a learner on a sequence of tasks.

    :param learner: A classifier that is possibly task-aware or
        task-boundary-aware.
    :param train_streams: A sequence of streams containing the training tasks.
    :param test_streams: A sequence of streams containing the testing tasks.
    :param continual_evaluations: The number of times to evaluate the learner
        during each task. If 1, the learner is only evaluated at the end of each task.
    :param progress_bar: Whether to display a progress bar. The bar displayed
        will show the progress over all training and evaluation steps including
        the continual evaluations.
    :return: A collection of metrics evaluating the learner's performance.
    """
    n_tasks = len(train_streams)
    if n_tasks != len(test_streams):
        raise ValueError("Number of train and test tasks must be equal")
    if not isinstance(learner, Classifier):
        raise ValueError("Learner must be a classifier")
    if continual_evaluations < 1:
        raise ValueError("Continual evaluations must be at least 1")

    metrics = _OCLEvaluator(
        n_tasks, continual_evaluations, learner.schema.get_num_classes()
    )

    # Setup progress bar
    train_len = sum(len(stream) for stream in train_streams)
    test_len = sum(len(stream) for stream in test_streams)
    pbar = tqdm(
        total=train_len + test_len * continual_evaluations * n_tasks,
        disable=not progress_bar,
        desc="Train & Eval",
    )

    # Iterate over each task
    for train_task_id, train_stream in enumerate(train_streams):
        # Setup stream and inform learner of the test task
        train_stream.restart()
        if isinstance(learner, TaskBoundaryAware):
            learner.set_train_task(train_task_id)

        # Train and evaluation loop for a single task
        for step, instance in enumerate(train_stream):
            # Update the learner and collect prequential statistics
            pbar.update(1)
            y_pred = learner.predict(instance)
            metrics.prequential_update(train_task_id, instance.y_index, y_pred)
            learner.train(instance)

            # Evaluate the learner on evenly spaced steps during training
            evaluate_every = len(train_stream) // continual_evaluations
            if (step + 1) % evaluate_every != 0:
                continue
            eval_step = step // evaluate_every

            for test_task_id, test_stream in enumerate(test_streams):
                # Setup stream and inform learner of the test task
                test_stream.restart()
                if isinstance(learner, TaskAware):
                    learner.set_test_task(test_task_id)

                # predict instances in the current task
                for instance in test_stream:
                    pbar.update(1)
                    y_pred = learner.predict(instance)
                    metrics.holdout_update(
                        train_task_id, eval_step, test_task_id, instance.y_index, y_pred
                    )

    return metrics.build()
