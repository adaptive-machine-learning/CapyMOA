"""Plots for the results of :func:`capymoa.ocl.evaluate_ocl`."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from capymoa.ocl.evaluation import OCLResults


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
    hline(ttt["accuracy"] / 100, "Online Avg. Acc.", cmap(2))

    ax.legend(ncol=3, frameon=False)
    ax.set_xlabel("Task")
    ax.set_xticks(range(n_tasks + 1))
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Over Tasks")
    ax.set_ylim(-0.1, 1.05)
    return ax
