"""Plots for the results of :func:`capymoa.regressor.evaluate_regressor`.

The results need ``store_y=True`` and ``store_predictions=True``.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from capymoa.evaluation.plot import _plot_drifts


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
    # cope with data
    *results,  # results value from regression models
    ground_truth=None,  # stored ground truths
    start=0,  # the start point of plotting
    end=1e10,  # the end
    # options for users
    plot_target=True,
    plot_predictions=True,
    plot_residuals=True,
    # target_type='line',  # line or dots
    add_target_markers=True,
    target_marker="*",  # can be any markers supported by matplotlib
    predictions_type="dots",  # line or dots
    predictions_marker=".",  # can be any markers supported by matplotlib
    absolute_residuals=False,
    plot_hist_residuals=False,
    kde_residuals=False,
    hist_bins=None,
    # color options
    color_target=None,
    color_predictions=None,  # if specified, MUST have same amount with *results
    # label settings
    xlabel=None,
    ylabel=None,
    # cope with file
    plot_title=None,
    figure_path="./",
    figure_name=None,
    figure_name_hist=None,
    save_only=False,
    prevent_plotting_drifts=False,
):
    # check if the results are prequential regression results
    for result in results:
        if "rmse" not in result:
            raise ValueError(
                "Cannot process results that do not include regression results."
            )

    # Check if the ground_truth is stored in the first result
    if ground_truth is None and results and results[0].get("y_true") is not None:
        ground_truth = results[0]["y_true"]

    # Check if ground_truth is none
    if ground_truth is None:
        raise ValueError("Ground truth y is None.")

    # Check for plotting interval
    start = max(start, 0)
    end = min(end, len(ground_truth))

    # Get ground truth
    targets = ground_truth[start:end]

    predictions = []
    residuals = []
    if absolute_residuals:
        absolute_values = []
    for i, result in enumerate(results):
        if result.get("y_pred") is not None:
            predictions.append(np.array(result["y_pred"][start:end]))
            residuals.append(
                np.array(np.array(result["y_pred"][start:end]) - np.array(targets))
            )
            if absolute_residuals:
                absolute_values.append(
                    np.abs(
                        np.array(
                            np.array(result["y_pred"][start:end]) - np.array(targets)
                        )
                    )
                )

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
                    label=results[i]["learner"] + " predictions",
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
                    label=results[i]["learner"] + " predictions",
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
                label=results[i]["learner"] + " residuals"
                if not absolute_residuals
                else " absolute residuals",
                color=color_predictions[i]
                if color_predictions is not None
                else default_colors[i],
                alpha=0.5,
            )

    if not prevent_plotting_drifts:
        _plot_drifts(results[0], start, end)

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

    # Show the plot or save it to the specified path
    if not save_only:
        plt.show()
    elif figure_path:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        figure_name = (
            figure_name
            if figure_name is not None
            else f"sequential_regression_results_{current_time}.pdf"
        )
        plt.savefig(figure_path + figure_name)

    # plot bar plot for residuals
    if plot_hist_residuals:
        plt.figure(figsize=(8, 6))
        for i, residual in enumerate(residuals):
            sns.histplot(
                residual,
                kde=kde_residuals,
                bins="auto" if hist_bins is None else hist_bins,
                label=results[i]["learner"],
                color=color_predictions[i]
                if color_predictions is not None
                else default_colors[i],
                alpha=0.5,
            )

        sns.set_style("darkgrid")
        plt.title("Residuals Histogram Plot")
        plt.legend()

        # Show the plot or save it to the specified path
        if not save_only:
            plt.show()
        elif figure_path:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            figure_name_hist = (
                figure_name_hist
                if figure_name_hist is not None
                else f"histogram_for_residuals_{current_time}.pdf"
            )
            plt.savefig(figure_path + figure_name_hist)
