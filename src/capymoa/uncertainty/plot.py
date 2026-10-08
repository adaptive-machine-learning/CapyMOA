"""Plots for the results of :func:`capymoa.uncertainty.evaluate_prediction_interval`.

The results need ``store_y=True`` and ``store_predictions=True``.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from capymoa.evaluation.plot import _plot_drifts


def plot_prediction_interval(
    *results,
    ground_truth=None,
    start=0,
    end=1e10,
    plot_truth=True,
    plot_bounds=True,
    plot_predictions=True,
    colors=None,
    xlabel=None,
    ylabel=None,
    plot_title=None,
    figure_path="./",
    figure_name=None,
    save_only=False,
    dynamic_switch=True,
    prevent_plotting_drifts=False,
):
    # check if the results are all prequential
    for result in results:
        if "coverage" not in result:
            raise ValueError(
                "Cannot process results that do not include prediction interval results."
            )

    if len(results) > 2:
        raise ValueError("This function only supports up to 2 results currently.")

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(results) == 1:
        if results[0].get("y_true") is not None:
            targets = results[0]["y_true"]
        elif ground_truth is not None:
            targets = ground_truth
        else:
            raise ValueError("No ground truth Found.")

        start = max(start, 0)
        end = min(end, len(targets))

        instance_numbers = list(range(start, end))
        targets = targets[start:end]
        intervals = results[0]["y_pred"][start:end]
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
        # Show the plot or save it to the specified path
        if not save_only:
            plt.show()
        elif figure_path:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            figure_name = (
                figure_name
                if figure_name
                else f"prediction_interval_over_time_{current_time}.pdf"
            )
            plt.savefig(figure_path + figure_name)

    # Plots two regions from prediction interval learners for comparison
    elif len(results) == 2:
        if results[0].get("y_true") is not None:
            targets = results[0]["y_true"]
        elif ground_truth is not None:
            targets = ground_truth
        else:
            raise ValueError("No ground truth Found.")

        start = max(start, 0)
        end = min(end, len(targets))

        instance_numbers = list(range(start, end))
        targets = targets[start:end]

        intervals_first = results[0]["y_pred"][start:end]
        intervals_second = results[1]["y_pred"][start:end]

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
        # Show the plot or save it to the specified path
        if not save_only:
            plt.show()
        elif figure_path:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            figure_name = (
                figure_name
                if figure_name
                else f"prediction_interval_over_time_comparison_{current_time}.pdf"
            )
            plt.savefig(figure_path + figure_name)
