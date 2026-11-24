"""Plots for the results of :func:`capymoa.ocl.evaluate_ocl`."""

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from capymoa.ocl.evaluate import OCLResults


def _anytime_x(results: OCLResults) -> np.ndarray:
    """Position of each anytime evaluation: the task plus the fraction done."""
    anytime = results["anytime"]
    return anytime["task"] + (anytime["step"] + 1) / results["n_continual_evaluations"]


def plot_accuracy_matrix(results: OCLResults, ax: Axes | None = None) -> Axes:
    """Plot the accuracy on each task while training on the tasks.

    Dots are the accuracy after each task, lines are the accuracy measured
    periodically (anytime) during training.

    :param results: The results of :func:`~capymoa.ocl.evaluate_ocl`.
    :param ax: The axes to plot on, defaults to new axes.
    :return: The axes.
    """
    ax = ax if ax is not None else plt.subplots(figsize=(8, 4))[1]
    cmap = plt.get_cmap("tab10")
    n_tasks = results["n_tasks"]
    x_task = results["per_task"]["task"] + 1
    x_anytime = _anytime_x(results)
    for t in range(n_tasks):
        color = cmap(t % 10)
        ax.scatter(
            x_task, results["accuracy_matrix"][:, t], color=color, label=f"Task {t}"
        )
        ax.plot(x_anytime, results["anytime_accuracy_matrix"][:, t], color=color)
    ax.set_xlabel("Task")
    ax.set_xticks(range(n_tasks + 1))
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Task Accuracy Over Tasks")
    ax.legend(frameon=False)
    return ax


def plot_accuracy(results: OCLResults, ax: Axes | None = None) -> Axes:
    """Plot the accuracy on all tasks and on seen tasks, and the online accuracy.

    - **Acc. (all/seen)**: the accuracy on all (or only seen) tasks after
      training on each task.
    - **Anytime Acc. (all/seen)**: as above, measured periodically while
      training on each task.
    - **Avg. Anytime Acc. (all/seen)**: the average of the anytime accuracy.
    - **Online Win. Acc.**: the windowed test-then-train accuracy.
    - **Online Avg. Acc.**: the cumulative test-then-train accuracy.

    :param results: The results of :func:`~capymoa.ocl.evaluate_ocl`.
    :param ax: The axes to plot on, defaults to new axes.
    :return: The axes.
    """
    ax = ax if ax is not None else plt.subplots(figsize=(8.2, 4))[1]
    cmap = plt.get_cmap("tab10")
    n_tasks = results["n_tasks"]
    per_task = results["per_task"]
    anytime = results["anytime"]
    x_task = per_task["task"] + 1
    x_anytime = _anytime_x(results)

    def hline(y: float, label: str, color) -> None:
        ax.hlines(y, 0, n_tasks, linestyles="--", label=label, color=color)

    ax.scatter(x_task, per_task["accuracy_all"], label="Acc. (all)")
    ax.plot(x_anytime, anytime["accuracy_all"], label="Anytime Acc. (all)")
    hline(results["anytime_accuracy_all_avg"], "Avg. Anytime Acc. (all)", cmap(0))

    ax.scatter(x_task, per_task["accuracy_seen"], label="Acc. (seen)")
    ax.plot(x_anytime, anytime["accuracy_seen"], label="Anytime Acc. (seen)")
    hline(results["anytime_accuracy_seen_avg"], "Avg. Anytime Acc. (seen)", cmap(1))

    ttt = results["ttt"]
    if "windowed" in ttt:
        ax.plot(
            ttt["windowed"]["task"],
            ttt["windowed"]["accuracy"] / 100,  # percentage to proportion
            label="Online Win. Acc.",
        )
    hline(ttt["cumulative"]["accuracy"] / 100, "Online Avg. Acc.", cmap(2))

    ax.legend(ncol=3, frameon=False)
    ax.set_xlabel("Task")
    ax.set_xticks(range(n_tasks + 1))
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Over Tasks")
    ax.set_ylim(-0.1, 1.05)
    return ax


def plot_confusion_matrix(
    results: OCLResults,
    task: int = -1,
    ax: Axes | None = None,
    show_ticks: bool = True,
) -> Axes:
    """Plot the confusion matrix with classes in the order they appear in the stream.

    Classes are sorted by their first appearance in the training stream. Inside
    a task, they are sorted by index. A learner that forgets has errors in the
    columns of old classes, and one that is biased to recent classes has
    errors in the last columns.

    Each row is normalised to sum to one, so the colour scale is always from 0
    to 1 (0% to 100%) and plots of different learners are comparable.

    :param results: The results of :func:`~capymoa.ocl.evaluate_ocl`.
    :param task: Show the confusion matrix on all classes after training on
        this task, defaults to the last task.
    :param ax: The axes to plot on, defaults to new axes.
    :param show_ticks: Show a tick for every class, labelled by position in the
        stream from 1. Otherwise show no ticks, defaults to True.
    :return: The axes.
    """
    ax = ax if ax is not None else plt.subplots(figsize=(5, 5))[1]
    order = results["class_order"]
    cm = results["class_cm"][task][np.ix_(order, order)]
    n_classes = len(order)
    # Each row is the fraction of a true class given each prediction.
    totals = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, totals, out=np.zeros(cm.shape), where=totals > 0)
    ax.imshow(cm, interpolation="nearest", vmin=0, vmax=1)
    if show_ticks:
        ticks = np.arange(n_classes)
        ax.set_xticks(ticks, ticks + 1)
        ax.set_yticks(ticks, ticks + 1)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix (Classes by Stream Order)")
    return ax


_MARKERS = "o^s*PvXD"


def plot_accuracy_comparison(
    results: Mapping[str, OCLResults | Sequence[OCLResults]],
    metric: Literal["accuracy_seen", "accuracy_all"] = "accuracy_seen",
    ax: Axes | None = None,
) -> Axes:
    """Compare learners by their accuracy after each task.

    Draws one line per learner. If a learner has results from several runs
    (for example seeds), the line is the mean and the shaded band is one
    standard deviation.

    :param results: Results by learner name. Each value is one result or a
        sequence of results from repeated runs. All must have the same number
        of tasks.
    :param metric: The accuracy after each task. ``"accuracy_seen"`` is on the
        tasks seen so far, ``"accuracy_all"`` is on all tasks.
    :param ax: The axes to plot on, defaults to new axes.
    :return: The axes.
    """
    ax = ax if ax is not None else plt.subplots(figsize=(6, 4))[1]
    n_tasks = None
    for i, (name, value) in enumerate(results.items()):
        runs = [value] if isinstance(value, Mapping) else list(value)
        accuracy = np.array([r["per_task"][metric] for r in runs])  # (runs, tasks)
        if n_tasks is None:
            n_tasks = accuracy.shape[1]
        elif accuracy.shape[1] != n_tasks:
            raise ValueError("All results must have the same number of tasks")
        x = np.arange(1, n_tasks + 1)
        mean = accuracy.mean(axis=0)
        (line,) = ax.plot(x, mean, marker=_MARKERS[i % len(_MARKERS)], label=name)
        if len(runs) > 1:
            std = accuracy.std(axis=0)
            ax.fill_between(
                x, mean - std, mean + std, color=line.get_color(), alpha=0.3
            )
    if n_tasks is not None:
        ax.set_xticks(range(1, n_tasks + 1))
    ax.set_xlabel("Number of Tasks")
    ax.set_ylabel("Accuracy")
    ax.legend()
    return ax
