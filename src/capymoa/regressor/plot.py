"""Plots for the results of :func:`capymoa.regressor.evaluate_regressor`.

The results need ``store_y=True`` and ``store_predictions=True``.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from capymoa.evaluation.plot import _plot_drifts
from capymoa.regressor.evaluate import RegressorResults


def plot_predictions_vs_ground_truth(
    *results,
    ground_truth=None,
    plot_interval=None,
    plot_title=None,
    xlabel=None,
    ylabel=None,
    figure_path="./",
    figure_name=None,
    save_only=False,
):
    """
    Plot predictions vs. ground truth for multiple results.

    If ground_truth is None, then the code should check if "ground_truth_y" is not None in the first result,
    i.e. results[0]["ground_truth_y"], and use it instead. If ground_truth is None and there is no data in
    results[0]["ground_truth_y"] (also None) then it raises an error stating that the ground truth y is None.

    The plot_interval parameter is a tuple (start, end) that determines when to start and stop plotting predictions.

    If save_only is True, then a figure will be saved at the specified path
    """

    # check if the results are prequential prediction interval results
    # for result in results:
    # if not hasattr(result.windowed, 'coverage'):
    #     raise ValueError('Cannot process results that do not include prediction interval results.')

    # Determine ground truth y
    if ground_truth is None and results and results[0].get("y_true") is not None:
        ground_truth = results[0]["y_true"]

    # Check if ground truth y is available
    if ground_truth is None:
        raise ValueError("Ground truth y is None.")

    # Create a figure
    plt.figure(figsize=(20, 6))

    # Determine indices to plot based on plot_interval
    start, end = plot_interval or (0, len(ground_truth))

    # Check if predictions have the same length as ground truth
    for i, result in enumerate(results):
        if result.get("y_pred") is not None:
            predictions = result["y_pred"][start:end]
            if len(predictions) != len(ground_truth[start:end]):
                raise ValueError(
                    f"Length of predictions for result {i + 1} does not match ground truth."
                )

    # Plot ground truth y vs. predictions for each result within the specified interval
    instance_numbers = list(range(start, end))
    # for i, result in enumerate(results):
    #     if "predictions" in result:
    #         predictions = result["predictions"][start:end]
    for result in results:
        predictions = result["y_pred"][start:end]
        plt.plot(
            instance_numbers,
            predictions,
            label=f"{result['learner']} predictions",
            alpha=0.7,
        )

    # Plot ground truth y
    plt.scatter(
        instance_numbers,
        ground_truth[start:end],
        label="ground truth",
        marker="*",
        s=20,
        color="red",
    )

    # TODO: Once Schema is updated to provide an easier access to the target name should remove direct access to MOA
    output_name = "target"

    # Add labels and title
    plt.xlabel(xlabel if xlabel else "# Instance")
    plt.ylabel(ylabel if ylabel else output_name)
    plt.title(plot_title if plot_title else "Predictions vs. Ground Truth")
    plt.grid(True)
    plt.legend()

    # Show the plot or save it to the specified path
    if not save_only:
        plt.show()
    elif figure_path:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        figure_name = (
            figure_name
            if figure_name
            else f"predictions_vs_ground_truth_{current_time}.pdf"
        )
        plt.savefig(figure_path + figure_name)


def plot_regression_results(
    *results: RegressorResults,
    ground_truth: None | np.ndarray | list[float] = None,
    start: int = 0,
    end: int | None = None,
    plot_target: bool = True,
    plot_predictions: bool = True,
    plot_residuals: bool = True,
    add_target_markers: bool = True,
    target_marker: str = "*",
    predictions_type: Literal["line", "dots"] = "dots",
    predictions_marker: str = ".",
    absolute_residuals: bool = False,
    plot_hist_residuals: bool = False,
    kde_residuals: bool = False,
    hist_bins: int | None = None,
    color_target: Any = None,
    color_predictions: Sequence[Any] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    plot_title: str | None = None,
    figure_path: str = "./",
    figure_name: str | None = None,
    figure_name_hist: str | None = None,
    save: bool = True,
    save_only: bool = False,
    prevent_plotting_drifts: bool = False,
) -> None:
    """Plot regression targets, predictions, and residuals.

    For example, evaluate a regressor while storing targets and predictions,
    then display the plot:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from capymoa.datasets import FriedTiny
    >>> from capymoa.regressor import FIMTDD, evaluate_regressor
    >>> stream = FriedTiny()
    >>> result = evaluate_regressor(
    ...     stream,
    ...     FIMTDD(stream.get_schema(), random_seed=1),
    ...     max_instances=100,
    ...     store_predictions=True,
    ...     store_y=True,
    ...     optimise=False,
    ... )
    >>> plot_regression_results(result, prevent_plotting_drifts=True)

    :param results: Regression results to plot; at least one is required.
    :param ground_truth: Target values as an array or list, or ``y_true`` from the first result.
    :param start: First instance index to plot; negative values become 0.
    :param end: Exclusive end index; defaults to and is clipped at the target length.
    :param plot_target: Draw target values as a line.
    :param plot_predictions: Draw predictions for results with ``y_pred``.
    :param plot_residuals: Draw residual bars; disabled for more than two dot-mode results.
    :param add_target_markers: Draw markers for target values.
    :param target_marker: Any Matplotlib marker for targets.
    :param predictions_type: Draw predictions as ``"line"`` or ``"dots"``.
    :param predictions_marker: Any Matplotlib marker for prediction dots.
    :param absolute_residuals: Plot absolute residual values.
    :param plot_hist_residuals: Also plot a residual histogram.
    :param kde_residuals: Add a kernel density estimate to each histogram.
    :param hist_bins: Histogram bin count, or ``None`` for seaborn's default.
    :param color_target: Target color; defaults to green.
    :param color_predictions: Colors for results with stored predictions.
    :param xlabel: X-axis label; defaults to ``"# Instance"``.
    :param ylabel: Y-axis label; defaults to ``"target"``.
    :param plot_title: Plot title; defaults to one based on enabled plots.
    :param figure_path: Directory to save figures when a file name is specified.
    :param figure_name: Optional file name for saving the main plot.
    :param figure_name_hist: Optional file name for saving the residual histogram.
    :param save: Save figures when ``True``; set to ``False`` to disable file output.
    :param save_only: Save without displaying; omitted file names are generated automatically.
    :param prevent_plotting_drifts: Do not mark drifts from the first result.
    :raises ValueError: If no results are supplied, a result has no regression
        metrics, ground truth is unavailable, or no plot type is enabled.
    """
    first_result = next(iter(results), None)
    if first_result is None:
        raise ValueError("At least one result is required.")

    # check if the results are prequential regression results
    for result in results:
        if "rmse" not in result["cumulative"]:
            raise ValueError(
                "Cannot process results that do not include regression results."
            )

    # Get the ground truth array
    if ground_truth is None:
        if "y_true" not in first_result:
            raise ValueError(
                "`ground_truth` is `None` but `results` does not contain `y_true`."
            )
        ground_truth_ = np.array(first_result["y_true"])
    elif isinstance(ground_truth, np.ndarray):
        ground_truth_ = ground_truth
    elif isinstance(ground_truth, list):
        ground_truth_ = np.array(ground_truth)
    else:
        raise ValueError(
            "`ground_truth` must be either `None`, `np.ndarray`, or `list[float]`."
        )

    # Check for plotting interval
    start = max(start, 0)
    end = len(ground_truth_) if end is None else min(end, len(ground_truth_))

    # Get ground truth
    targets = ground_truth_[start:end]

    predictions = []
    prediction_results = []
    residuals = []
    if absolute_residuals:
        absolute_values = []
    for result in results:
        y_pred = result.get("y_pred")
        if y_pred is None:
            continue

        prediction = np.array(y_pred[start:end])
        predictions.append(prediction)
        prediction_results.append(result)
        residual = prediction - targets
        residuals.append(residual)
        if absolute_residuals:
            absolute_values.append(np.abs(residual))

    # Create a figure
    plt.figure(figsize=((end - start) / 10, 6))
    # x-axis
    instance_numbers = list(range(start, end))

    # get default colors from matplotlib for further possible use
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot targets
    if plot_target:
        plt.plot(
            instance_numbers,
            targets,
            label="targets",
            linewidth=1,
            color=color_target if color_target is not None else "g",
        )
    if add_target_markers:
        plt.scatter(
            instance_numbers,
            targets,
            label="targets",
            marker=target_marker,
            s=20,
            color=color_target if color_target is not None else "g",
        )

    # plot predictions
    if plot_predictions:
        for i, prediction in enumerate(predictions):
            if predictions_type == "line":
                plt.plot(
                    instance_numbers,
                    prediction,
                    label=prediction_results[i]["learner"] + " predictions",
                    color=color_predictions[i]
                    if color_predictions is not None
                    else default_colors[i],
                    linewidth=1,
                    linestyle="--",
                    alpha=0.5,
                )
            elif predictions_type == "dots":
                plt.scatter(
                    instance_numbers,
                    prediction,
                    label=prediction_results[i]["learner"] + " predictions",
                    color=color_predictions[i]
                    if color_predictions is not None
                    else default_colors[i],
                    marker=predictions_marker,
                    s=20,
                )
            else:
                raise ValueError("Predictions_type must be 'line' or 'dots'.")

    if predictions_type == "dots":
        if len(results) > 2:
            plot_residuals = False

        for i in range(len(instance_numbers)):
            values = [predictions[x][i] for x in range(len(predictions))]
            values.append(targets[i])
            values = np.array(values)
            plt.vlines(
                x=instance_numbers[i],
                ymin=min(values),
                ymax=max(values),
                linestyles="dashed",
                colors="grey",
                linewidth=0.5,
            )

    # plot residuals
    if plot_residuals:
        for i, residual in enumerate(residuals):
            plt.bar(
                instance_numbers,
                residual if not absolute_residuals else absolute_values[i],
                label=prediction_results[i]["learner"] + " residuals"
                if not absolute_residuals
                else " absolute residuals",
                color=color_predictions[i]
                if color_predictions is not None
                else default_colors[i],
                alpha=0.5,
            )

    if not prevent_plotting_drifts:
        _plot_drifts(first_result, start, end)

    output_name = "target"

    prepared_title = ""
    fragments = []
    if plot_predictions:
        fragments.append("Predictions")
    if plot_target:
        fragments.append("Targets")
    if plot_residuals:
        fragments.append(
            "Residuals" if not absolute_residuals else " Absolute Residuals"
        )
    if len(fragments) > 0:
        for i, s in enumerate(fragments):
            prepared_title += s
            if i < len(fragments) - 1:
                prepared_title += " vs. "
    else:
        raise ValueError("Nothing to plot")

    # Add labels and title
    sns.set_style("darkgrid")
    plt.xlabel(xlabel if xlabel else "# Instance")
    plt.ylabel(ylabel if ylabel else output_name)
    plt.title(plot_title if plot_title else prepared_title)
    plt.legend()

    # Save when requested, then display unless save_only was selected.
    if save:
        if figure_name is None and save_only:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            figure_name = f"sequential_regression_results_{current_time}.pdf"
        if figure_name is not None:
            plt.savefig(figure_path + figure_name)
    if not save_only:
        plt.show()

    # plot bar plot for residuals
    if plot_hist_residuals:
        plt.figure(figsize=(8, 6))
        for i, residual in enumerate(residuals):
            sns.histplot(
                residual,
                kde=kde_residuals,
                bins="auto" if hist_bins is None else hist_bins,
                label=prediction_results[i]["learner"],
                color=color_predictions[i]
                if color_predictions is not None
                else default_colors[i],
                alpha=0.5,
            )

        sns.set_style("darkgrid")
        plt.title("Residuals Histogram Plot")
        plt.legend()

        # Save when requested, then display unless save_only was selected.
        if save:
            if figure_name_hist is None and save_only:
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                figure_name_hist = f"histogram_for_residuals_{current_time}.pdf"
            if figure_name_hist is not None:
                plt.savefig(figure_path + figure_name_hist)
        if not save_only:
            plt.show()
