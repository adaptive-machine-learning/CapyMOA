"""Results and metric helpers for OCL evaluation."""

import itertools
from typing import NotRequired, TypedDict

import numpy as np
import torch

from capymoa.classifier.evaluate import ClassifierResults, ClassifierWindows


class PerTask(TypedDict):
    """Accuracy after training on each task, one entry per task (in columns)."""

    #: The task that was just trained on, numbered from 0.
    task: np.ndarray
    #: The accuracy on all tasks.
    accuracy_all: np.ndarray
    #: The accuracy on the tasks seen so far.
    accuracy_seen: np.ndarray


class Anytime(TypedDict):
    """Accuracy during training, one entry per evaluation (in columns)."""

    #: The task being trained on, numbered from 0.
    task: np.ndarray
    #: The evaluation within the task (``0`` to ``n_continual_evaluations - 1``).
    step: np.ndarray
    #: The accuracy on all tasks.
    accuracy_all: np.ndarray
    #: The accuracy on the tasks seen so far.
    accuracy_seen: np.ndarray


class TaskWindows(ClassifierWindows):
    """Windowed metrics of the online evaluation with the position in the tasks."""

    #: The position of each window within the tasks: the task number plus the
    #: fraction of it done. Useful as the x axis.
    task: np.ndarray


class OnlineResults(ClassifierResults):
    """Test-then-train results of an OCL run."""

    #: The metrics of each window, with the position in the tasks (see
    #: :class:`TaskWindows`).
    windowed: NotRequired[TaskWindows]


class OCLResults(TypedDict):
    r"""Results of evaluating an online continual learner. See :func:`~capymoa.ocl.evaluate_ocl`.

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

    Tasks are numbered from 0.
    """

    #: Name of the learner.
    learner: str
    #: Name of the stream of tasks.
    stream: str
    #: Elapsed time in seconds.
    wallclock: float
    #: CPU time in seconds.
    cpu_time: float
    #: The number of classes :math:`C`.
    n_classes: int
    #: The number of tasks :math:`T`.
    n_tasks: int
    #: The number of continual evaluations per task :math:`H`.
    n_continual_evaluations: int

    #: The accuracy on all tasks after training on the final task.
    #:
    #: .. math:: a_\text{final} = a_\text{all}(T)
    accuracy_final: float
    #: The average of ``per_task["accuracy_all"]`` over all tasks.
    #:
    #: .. math:: \bar{a}_\text{all} = \frac{1}{T}\sum_{t=1}^T a_\text{all}(t)
    accuracy_all_avg: float
    #: The average of ``per_task["accuracy_seen"]`` over all tasks.
    #:
    #: .. math:: \bar{a}_\text{seen} = \frac{1}{T}\sum_{t=1}^T a_\text{seen}(t)
    accuracy_seen_avg: float
    #: The average of ``anytime["accuracy_all"]`` over all steps.
    #:
    #: .. math::
    #:
    #:     \bar{a}_\text{any all} = \frac{1}{T}\sum_{t=1}^T \frac{1}{H}\sum_{h=1}^H a_\text{any all}(t, h)
    anytime_accuracy_all_avg: float
    #: The average of ``anytime["accuracy_seen"]`` over all steps.
    #:
    #: .. math::
    #:
    #:     \bar{a}_\text{any seen} = \frac{1}{T}\sum_{t=1}^T \frac{1}{H}\sum_{h=1}^H a_\text{any seen}(t, h)
    anytime_accuracy_seen_avg: float
    #: A scalar measuring the impact learning had on future tasks.
    #:
    #: .. math:: r_\text{FWT} = \frac{2}{T(T-1)}\sum_{i=1}^{T} \sum_{j=i+1}^{T} R_{i,j}
    forward_transfer: float
    #: A scalar measuring the impact learning had on past tasks.
    #:
    #: .. math::
    #:
    #:     r_\text{BWT} = \frac{2}{T(T-1)} \sum_{i=2}^{T} \sum_{j=1}^{i-1} (R_{i,j} - R_{j,j})
    backward_transfer: float

    #: The accuracy on each task after training on each task.
    #: Shape ``(n_tasks, n_tasks)``. ``R[i, j]`` is the accuracy on task
    #: :math:`j` after training on tasks :math:`1` through :math:`i`.
    accuracy_matrix: np.ndarray
    #: The accuracy on each task after training on each task and step. Shape
    #: ``(n_tasks * n_continual_evaluations, n_tasks)``. This is :math:`A` with
    #: the first two dimensions flattened.
    anytime_accuracy_matrix: np.ndarray
    #: A confusion matrix of shape ``(task, true_class, predicted_class)``.
    class_cm: np.ndarray
    #: Instance index of the boundaries of the tasks while training. Shape
    #: ``(n_tasks + 1,)``.
    boundaries: np.ndarray

    #: The accuracy on all tasks, and on the **seen** tasks, after training on
    #: each task, in columns (see :class:`PerTask`).
    #:
    #: .. math::
    #:
    #:     a_\text{all}(t) = \frac{1}{T} \sum_{i=1}^{T} R_{t,i} \qquad
    #:     a_\text{seen}(t) = \frac{1}{t}\sum^t_{i=1} R_{t,i}
    per_task: PerTask
    #: The accuracy on all tasks, and on the **seen** tasks, at each evaluation
    #: during training, in columns (see :class:`Anytime`).
    #:
    #: .. math::
    #:
    #:     a_\text{any all}(t, h) = \frac{1}{T}\sum^T_{i=1} A_{t,h,i} \qquad
    #:     a_\text{any seen}(t, h) = \frac{1}{t}\sum^t_{i=1} A_{t,h,i}
    anytime: Anytime

    #: Test-then-train/prequential results. ``ttt["windowed"]`` has an extra
    #: ``task`` entry (see :class:`TaskWindows`).
    ttt: OnlineResults


def _backwards_transfer(R: torch.Tensor) -> float:
    n = R.size(0)
    assert R.shape == (n, n)
    return ((R - R.diag()).tril().sum() / (n * (n - 1) / 2)).item()


def _forwards_transfer(R: torch.Tensor) -> float:
    n = R.size(0)
    assert R.shape == (n, n)
    return (R.triu(1).sum() / (n * (n - 1) / 2)).item()


def _get_ttt_windowed_task_index(boundaries: np.ndarray, window_size: int):
    tasks = np.zeros(int(boundaries[-1]) // window_size)
    for task_id, (start, end) in enumerate(itertools.pairwise(boundaries)):
        win_start = int(start) // window_size
        win_end = int(end) // window_size
        tasks[win_start:win_end] = np.linspace(
            task_id, task_id + 1, win_end - win_start
        )
    return tasks
