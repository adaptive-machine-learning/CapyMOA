"""Plots for the results of :func:`capymoa.uncertainty.evaluate_uncertainty`.

The results need ``store_y=True`` and ``store_predictions=True``.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from capymoa.evaluation.plot import _plot_drifts
from capymoa.uncertainty.evaluate import UncertaintyResults


def plot_prediction_interval(
    *results: UncertaintyResults,
    ground_truth: Sequence[float] | np.ndarray | None = None,
    start: int = 0,
    end: int | None = None,
    plot_truth: bool = True,
    plot_bounds: bool = True,
    plot_predictions: bool = True,
    colors: Sequence[Any] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    plot_title: str | None = None,
    figure_path: str = "./",
    figure_name: str | None = None,
    save_only: bool = False,
    save: bool = True,
    dynamic_switch: bool = True,
    prevent_plotting_drifts: bool = False,
) -> None:
    """Plot one prediction interval result or compare two results.

    Results must include stored ground truth and interval predictions unless
    ``ground_truth`` is provided. Each prediction is represented by its lower
    bound, point prediction, and upper bound.

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from capymoa.datasets import FriedTiny
    >>> from capymoa.uncertainty import MVE, evaluate_uncertainty
    >>> stream = FriedTiny()
    >>> result = evaluate_uncertainty(
    ...     stream,
    ...     MVE(stream.get_schema()),
    ...     max_instances=100,
    ...     store_predictions=True,
    ...     store_y=True,
    ... )
    >>> plot_prediction_interval(result, prevent_plotting_drifts=True)

    :param results: One or two prediction interval results to plot.
    :param ground_truth: Optional target values; otherwise use stored ``y_true``.
    :param start: First instance index to plot.
    :param end: Exclusive final instance index.
    :param plot_truth: Show ground-truth values and whether they are covered.
    :param plot_bounds: Show interval bounds and shaded interval regions.
    :param plot_predictions: Show the point predictions.
    :param colors: Optional color for each result.
    :param xlabel: X-axis label; defaults to ``"# Instance"``.
    :param ylabel: Y-axis label; defaults to ``"target"``.
    :param plot_title: Optional plot title.
    :param figure_path: Directory used when saving the plot.
    :param figure_name: Optional name for the saved plot.
    :param save_only: Save without displaying; generate a file name if omitted.
    :param save: Save the plot when ``True``; set to ``False`` to disable file output.
    :param dynamic_switch: Dynamically layer overlapping intervals when comparing two results.
    :param prevent_plotting_drifts: Do not mark drifts from the first result.
    :raises ValueError: If a result lacks interval metrics, more than two results are supplied, or ground truth is unavailable.
    """
    # check if the results are all prequential
    for result in results:
        if "coverage" not in result["cumulative"]:
            raise ValueError(
                "Cannot process results that do not include prediction interval results."
            )

    if len(results) > 2:
        raise ValueError("This function only supports up to 2 results currently.")

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(results) == 1:
        stored_targets = results[0].get("y_true")
        if stored_targets is not None:
            targets = stored_targets
        elif ground_truth is not None:
            targets = ground_truth
        else:
            raise ValueError("No ground truth Found.")

        intervals = results[0].get("y_pred")
        if intervals is None:
            raise ValueError("No stored interval predictions found.")

        start = max(start, 0)
        end = len(targets) if end is None else min(end, len(targets))

        instance_numbers = list(range(start, end))
        targets = targets[start:end]
        intervals = intervals[start:end]
        upper = []
        lower = []
        predictions = []
        for interval in intervals:
            upper.append(interval[2])
            lower.append(interval[0])
            predictions.append(interval[1])

        plt.figure(figsize=((end - start) / 10, 6))

        if plot_bounds:
            upper = np.array(upper)
            lower = np.array(lower)
            plt.plot(
                instance_numbers,
                upper,
                linewidth=0.1,
                alpha=0.2,
                color=colors[0] if colors is not None else default_colors[0],
            )
            plt.plot(
                instance_numbers,
                lower,
                linewidth=0.1,
                alpha=0.2,
                color=colors[0] if colors is not None else default_colors[0],
            )
            plt.fill_between(
                instance_numbers,
                upper,
                lower,
                color=colors[0] if colors is not None else default_colors[0],
                alpha=0.5,
                label=results[0]["learner"] + " interval",
            )
        if plot_predictions:
            plt.plot(
                instance_numbers,
                np.array(predictions),
                linewidth=1,
                linestyle="-",
                color=colors[0] if colors is not None else default_colors[0],
                label=results[0]["learner"] + " predictions",
            )
        if plot_truth:
            insideX = []
            insideY = []
            outsideX = []
            outsideY = []
            for i, v in enumerate(targets):
                if upper[i] >= v >= lower[i]:
                    insideX.append(instance_numbers[i])
                    insideY.append(v)
                else:
                    outsideX.append(instance_numbers[i])
                    outsideY.append(v)

            plt.scatter(
                np.array(insideX),
                np.array(insideY),
                marker="*",
                color="g",
                label="Ground Truth (inner)",
            )
            plt.scatter(
                np.array(outsideX),
                np.array(outsideY),
                marker="x",
                color="r",
                label="Ground Truth (outer)",
            )

        if not prevent_plotting_drifts:
            _plot_drifts(results[0], start, end)

        output_name = "target"

        # Add labels and title
        sns.set_style("darkgrid")
        plt.xlabel(xlabel if xlabel else "# Instance")
        plt.ylabel(ylabel if ylabel else output_name)
        plt.title(plot_title if plot_title else "Prediction Interval")
        plt.legend()
        # Save when requested, then display unless save_only was selected.
        if save:
            if figure_name is None and save_only:
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                figure_name = f"prediction_interval_over_time_{current_time}.pdf"
            if figure_name is not None:
                plt.savefig(figure_path + figure_name)
        if not save_only:
            plt.show()

    # Plots two regions from prediction interval learners for comparison
    elif len(results) == 2:
        stored_targets = results[0].get("y_true")
        if stored_targets is not None:
            targets = stored_targets
        elif ground_truth is not None:
            targets = ground_truth
        else:
            raise ValueError("No ground truth Found.")

        first_intervals = results[0].get("y_pred")
        second_intervals = results[1].get("y_pred")
        if first_intervals is None or second_intervals is None:
            raise ValueError("Both results must include stored interval predictions.")

        start = max(start, 0)
        end = len(targets) if end is None else min(end, len(targets))

        instance_numbers = list(range(start, end))
        targets = targets[start:end]

        intervals_first = first_intervals[start:end]
        intervals_second = second_intervals[start:end]

        upper_first = []
        lower_first = []
        upper_second = []
        lower_second = []
        predictions_first = []
        predictions_second = []

        for i in range(len(targets)):
            upper_first.append(intervals_first[i][2])
            lower_first.append(intervals_first[i][0])
            upper_second.append(intervals_second[i][2])
            lower_second.append(intervals_second[i][0])
            predictions_first.append(intervals_first[i][1])
            predictions_second.append(intervals_second[i][1])

        plt.figure(figsize=((end - start) / 10, 6))

        if plot_bounds:
            u_first = np.array(upper_first)
            l_first = np.array(lower_first)
            u_second = np.array(upper_second)
            l_second = np.array(lower_second)

            if not dynamic_switch:
                # Plot first area
                plt.plot(
                    instance_numbers,
                    u_first,
                    linewidth=0.1,
                    alpha=0.2,
                    color=colors[0] if colors is not None else default_colors[0],
                )
                plt.plot(
                    instance_numbers,
                    l_first,
                    linewidth=0.1,
                    alpha=0.2,
                    color=colors[0] if colors is not None else default_colors[0],
                )
                plt.fill_between(
                    instance_numbers,
                    u_first,
                    l_first,
                    color=colors[0] if colors is not None else default_colors[0],
                    alpha=0.2,
                    label=results[0]["learner"] + " interval",
                )

                # Plot second area
                plt.plot(
                    instance_numbers,
                    u_second,
                    linewidth=0.1,
                    alpha=0.5,
                    color=colors[1] if colors is not None else default_colors[1],
                )
                plt.plot(
                    instance_numbers,
                    l_second,
                    linewidth=0.1,
                    alpha=0.5,
                    color=colors[1] if colors is not None else default_colors[1],
                )
                plt.fill_between(
                    instance_numbers,
                    u_second,
                    l_second,
                    color=colors[1] if colors is not None else default_colors[1],
                    alpha=0.5,
                    label=results[1]["learner"] + " interval",
                )
            else:
                # define function for further dynamic plot
                def _plot_first(i, alpha):
                    plt.plot(
                        instance_numbers[switch_points[i] : switch_points[i + 1] + 1],
                        u_first[switch_points[i] : switch_points[i + 1] + 1],
                        linewidth=0.1,
                        alpha=alpha,
                        color=colors[0] if colors is not None else default_colors[0],
                    )
                    plt.plot(
                        instance_numbers[switch_points[i] : switch_points[i + 1] + 1],
                        l_first[switch_points[i] : switch_points[i + 1] + 1],
                        linewidth=0.1,
                        alpha=alpha,
                        color=colors[0] if colors is not None else default_colors[0],
                    )

                    plt.fill_between(
                        instance_numbers[switch_points[i] : switch_points[i + 1] + 1],
                        u_first[switch_points[i] : switch_points[i + 1] + 1],
                        l_first[switch_points[i] : switch_points[i + 1] + 1],
                        color=colors[0] if colors is not None else default_colors[0],
                        alpha=alpha,
                        label=results[0]["learner"] + " interval" if i == 0 else "",
                    )

                def _plot_second(i, alpha):
                    plt.plot(
                        instance_numbers[switch_points[i] : switch_points[i + 1] + 1],
                        u_second[switch_points[i] : switch_points[i + 1] + 1],
                        linewidth=0.1,
                        alpha=alpha,
                        color=colors[1] if colors is not None else default_colors[1],
                    )
                    plt.plot(
                        instance_numbers[switch_points[i] : switch_points[i + 1] + 1],
                        l_second[switch_points[i] : switch_points[i + 1] + 1],
                        linewidth=0.1,
                        alpha=alpha,
                        color=colors[1] if colors is not None else default_colors[1],
                    )

                    plt.fill_between(
                        instance_numbers[switch_points[i] : switch_points[i + 1] + 1],
                        u_second[switch_points[i] : switch_points[i + 1] + 1],
                        l_second[switch_points[i] : switch_points[i + 1] + 1],
                        color=colors[1] if colors is not None else default_colors[1],
                        alpha=alpha,
                        label=results[1]["learner"] + " interval" if i == 0 else "",
                    )

                # determine which on top first
                first_first = l_first[0] > l_second[0]
                # find the switch point
                larger = True
                switch_points = [0]
                for i in range(len(l_first)):
                    if larger:
                        if l_first[i] < l_second[i]:
                            switch_points.append(i)
                            larger = not larger
                    else:
                        if l_first[i] > l_second[i]:
                            switch_points.append(i)
                            larger = not larger
                switch_points.append(len(u_first) - 1)

                # Plot dynamic switching areas
                for i in range(len(switch_points) - 1):
                    if first_first:
                        if i % 2 == 0:
                            _plot_first(i, alpha=0.2)
                            _plot_second(i, alpha=0.5)
                        else:
                            _plot_second(i, alpha=0.2)
                            _plot_first(i, alpha=0.5)
                    else:
                        if i % 2 == 0:
                            _plot_second(i, alpha=0.2)
                            _plot_first(i, alpha=0.5)
                        else:
                            _plot_first(i, alpha=0.2)
                            _plot_second(i, alpha=0.5)

        #  Plot predictions
        if plot_predictions:
            plt.plot(
                instance_numbers,
                np.array(predictions_first),
                linewidth=1,
                linestyle="-",
                color=colors[0] if colors is not None else default_colors[0],
                label=results[0]["learner"] + " predictions",
            )
            plt.plot(
                instance_numbers,
                np.array(predictions_second),
                linewidth=1,
                linestyle="-",
                color=colors[1] if colors is not None else default_colors[1],
                label=results[1]["learner"] + " predictions",
            )

        if plot_truth:
            insideX = []
            insideY = []
            betweenX = []
            betweenY = []
            outsideX = []
            outsideY = []

            for i, v in enumerate(targets):
                _out = v >= max(upper_first[i], upper_second[i]) or v <= min(
                    lower_first[i], lower_second[i]
                )
                _in = (
                    min(upper_first[i], upper_second[i])
                    >= v
                    >= max(lower_first[i], lower_second[i])
                )
                if _out:
                    outsideX.append(instance_numbers[i])
                    outsideY.append(v)
                elif _in:
                    insideX.append(instance_numbers[i])
                    insideY.append(v)
                else:
                    betweenX.append(instance_numbers[i])
                    betweenY.append(v)

            plt.scatter(
                np.array(insideX),
                np.array(insideY),
                marker="*",
                color="g",
                label="Ground Truth (inner)",
            )
            plt.scatter(
                np.array(outsideX),
                np.array(outsideY),
                marker="x",
                color="r",
                label="Ground Truth (outer)",
            )
            plt.scatter(
                np.array(betweenX),
                np.array(betweenY),
                marker="+",
                color="orange",
                label="Ground Truth (interim)",
            )

        if not prevent_plotting_drifts:
            _plot_drifts(results[0], start, end)

        output_name = "target"

        # Add labels and title
        sns.set_style("darkgrid")
        plt.xlabel(xlabel if xlabel else "# Instance")
        plt.ylabel(ylabel if ylabel else output_name)
        plt.title(plot_title if plot_title else "Prediction Interval Comparison")
        plt.legend()
        # Save when requested, then display unless save_only was selected.
        if save:
            if figure_name is None and save_only:
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                figure_name = (
                    f"prediction_interval_over_time_comparison_{current_time}.pdf"
                )
            if figure_name is not None:
                plt.savefig(figure_path + figure_name)
        if not save_only:
            plt.show()
