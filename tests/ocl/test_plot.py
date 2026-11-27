import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

pytest.importorskip("torch")

from capymoa.classifier import Finetune
from capymoa.core.torch.ann import Perceptron
from capymoa.ocl import evaluate_ocl
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.plot import plot_accuracy_comparison, plot_confusion_matrix


@pytest.mark.torch
def test_plot_confusion_matrix_stream_order():
    scenario = TinySplitMNIST()
    learner = Finetune(scenario.schema, Perceptron(scenario.schema), random_seed=0)
    results = evaluate_ocl(
        learner, scenario.train_loaders(32), scenario.test_loaders(32)
    )
    order = results["class_order"]
    # Classes of each task are together, and tasks are in stream order.
    start = 0
    for task in scenario.task_schedule:
        assert set(order[start : start + len(task)].tolist()) == task
        start += len(task)
    ax = plot_confusion_matrix(results)
    shown = ax.images[0].get_array()
    cm = results["class_cm"][-1][np.ix_(order, order)]
    assert np.allclose(shown, cm / cm.sum(axis=1, keepdims=True))
    assert ax.images[0].get_clim() == (0, 1)


@pytest.mark.torch
@pytest.mark.parametrize("show_ticks", [True, False])
def test_plot_confusion_matrix_ticks(show_ticks):
    scenario = TinySplitMNIST()
    learner = Finetune(scenario.schema, Perceptron(scenario.schema), random_seed=0)
    results = evaluate_ocl(
        learner, scenario.train_loaders(32), scenario.test_loaders(32)
    )
    ax = plot_confusion_matrix(results, show_ticks=show_ticks)
    expected = results["n_classes"] if show_ticks else 0
    assert len(ax.get_xticks()) == expected
    assert len(ax.get_yticks()) == expected


@pytest.mark.torch
def test_plot_accuracy_comparison():
    scenario = TinySplitMNIST()
    runs = []
    for seed in (0, 1):
        learner = Finetune(
            scenario.schema, Perceptron(scenario.schema), random_seed=seed
        )
        runs.append(
            evaluate_ocl(
                learner,
                scenario.train_loaders(32),
                scenario.test_loaders(32),
                continual_evaluations=3,
            )
        )
    ax = plot_accuracy_comparison({"a": runs, "b": runs[0]})
    assert len(ax.lines) == 2
    assert len(ax.collections) == 1  # only "a" has a band
    expected = np.mean([r["anytime"]["accuracy_seen"] for r in runs], axis=0)
    assert np.allclose(ax.lines[0].get_ydata(), expected)
    assert len(ax.lines[0].get_xdata()) == len(expected)
