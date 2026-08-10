"""Data structures and metric helpers for OCL evaluation."""

from dataclasses import dataclass

import numpy as np
import torch

from capymoa.evaluation.results import PrequentialResults


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

    Is a ndarray of shape (n_tasks * n_continual_evaluations,), dtype=np.float32.

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
    r"""The position in each task where the anytime accuracy was measured.

    Is a ndarray of shape (n_tasks * n_continual_evaluations,), dtype=np.integer.
    """

    accuracy_all: np.ndarray
    r"""The accuracy on all tasks after training on each task.

    Is a ndarray of shape (n_tasks,), dtype=np.float32

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

    Is a ndarray of shape (n_tasks,), dtype=np.float32.

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
    r"""The accuracy on all tasks after training on the final task.

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

    Is a ndarray of shape (n_tasks, n_tasks), dtype=np.float32.

    ``R[i, j]`` is the accuracy on task :math:`j` after training on tasks
    :math:`1` through :math:`i`.
    """

    class_cm: np.ndarray
    r"""A confusion matrix of shape ``(task, true_class, predicted_class)``.
    """

    anytime_accuracy_matrix: np.ndarray
    r"""A matrix measuring the accuracy on each task after training on each task and step.

    Is a ndarray of shape (n_tasks * n_continual_evaluations, n_tasks), dtype=np.float32.

    This matrix is :math:`A` with the first two dimensions flattened to a 2D array.
    """

    n_classes: int
    r"""The number of classes :math:`C`."""

    n_tasks: int
    r"""The number of tasks :math:`T`."""

    n_continual_evaluations: int
    r"""The number of continual evaluations per task :math:`H`."""

    ttt: PrequentialResults
    """Test-then-train/prequential results."""
    boundaries: np.ndarray
    r"""Instance index for the boundaries.

    Used to map online evaluation to specific tasks.

    Is a ndarray of shape (n_tasks + 1,), dtype=np.integer.
    """
    ttt_windowed_task_index: np.ndarray
    """The position of each window within each task.

    Useful as the ``x`` axis for
    :attr:`capymoa.evaluation.results.PrequentialResults.windowed`.
    """


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
    for task_id, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        win_start = int(start) // window_size
        win_end = int(end) // window_size
        tasks[win_start:win_end] = np.linspace(
            task_id, task_id + 1, win_end - win_start
        )
    return tasks
