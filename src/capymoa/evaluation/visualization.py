import matplotlib.pyplot as plt
from datetime import datetime
from capymoa.stream.drift import DriftStream
from com.yahoo.labs.samoa.instances import InstancesHeader
import numpy as np
import seaborn as sns
from capymoa.evaluation.results import PrequentialResults


def plot_windowed_results(
        *results,
        metric="classifications correct (percent)",
        plot_title=None,
        xlabel=None,
        ylabel=None,
        figure_path="./",
        figure_name=None,
        save_only=True,
        prevent_plotting_drifts=False,
        # ,
        # drift_locations=None, gradual_drift_window_lengths=None
):
    """
    Plot a comparison of values from multiple evaluators based on a selected column using line plots.
    It assumes the results contain windowed results ('windowed') which often originate from metrics_per_window()
    and the learner identification ('learner').

    If figure_path is provided, the figure will be saved at the specified path instead of displaying it.
    """
    dfs = []
    labels = []

    # check if the results are all prequential
    for result in results:
        if not isinstance(result, PrequentialResults):
            raise ValueError('only can process PrequantialResults class.')

    num_instances = results[0].max_instances
    stream = results[0]["stream"]

    if num_instances is not None:
        window_size = results[0].windowed.metrics_per_window()['classified instances'][0]
        num_windows = results[0].windowed.metrics_per_window().shape[0]
        x_values = []
        for i in range(1, num_windows + 1):
            x_values.append(i * window_size)
        # print(f'x_values: {x_values}')

    # Check if the given metric exists in all DataFrames
    for result in results:
        df = result.windowed.metrics_per_window()
        if metric not in df.columns:
            print(
                f"Column '{metric}' not found in metrics DataFrame for {result['learner']}. Skipping."
            )
        else:
            dfs.append(df)
            # if "experiment_id" in result:
            #     labels.append(result["experiment_id"])
            # else:
            labels.append(result["learner"])

    if not dfs:
        print("No valid DataFrames to plot.")
        return

    # Create a figure
    plt.figure(figsize=(12, 5))

    # Plot data from each DataFrame
    for i, df in enumerate(dfs):
        # print(f'df.index: {df.index}')
        if num_instances is not None:
            plt.plot(
                x_values,
                df[metric],
                label=labels[i],
                marker="o",
                linestyle="-",
                markersize=5,
            )
        else:
            plt.plot(
                df.index,
                df[metric],
                label=labels[i],
                marker="o",
                linestyle="-",
                markersize=5,
            )

    if stream is not None and isinstance(stream, DriftStream):
        if not prevent_plotting_drifts:
            drifts = stream.get_drifts()

            drift_locations = [drift.position for drift in drifts]
            gradual_drift_window_lengths = [drift.width for drift in drifts]

            # Add vertical lines at drift locations
            if drift_locations:
                for location in drift_locations:
                    plt.axvline(location, color="red", linestyle="-")

            # Add gradual drift windows as 70% transparent rectangles
            if gradual_drift_window_lengths:
                if not drift_locations:
                    print(
                        "Error: gradual_drift_window_lengths is provided, but drift_locations is not."
                    )
                    return

                if len(drift_locations) != len(gradual_drift_window_lengths):
                    print(
                        "Error: drift_locations and gradual_drift_window_lengths must have the same length."
                    )
                    return

                for i in range(len(drift_locations)):
                    location = drift_locations[i]
                    window_length = gradual_drift_window_lengths[i]

                    # Plot the 70% transparent rectangle
                    plt.axvspan(
                        location - window_length / 2,
                        location + window_length / 2,
                        alpha=0.2,
                        color="red",
                    )

    # Add labels and title
    xlabel = xlabel if xlabel is not None else "# Instances"
    plt.xlabel(xlabel)
    ylabel = ylabel if ylabel is not None else metric
    plt.ylabel(ylabel)
    plot_title = plot_title if plot_title is not None else metric
    plt.title(plot_title)

    # Add legend
    plt.legend()
    plt.grid(True)

    # Show the plot or save it to the specified path
    if save_only == False:
        plt.show()
    elif figure_path is not None:
        if figure_name is None:
            figure_name = result["learner"] + "_" + ylabel.replace(" ", "")
        plt.savefig(figure_path + figure_name)


# TODO: Update this function so that it works properly with DriftStreams
# TODO: Once Schema is updated to provide an easier access to the target name should remove direct access to MOA
def plot_predictions_vs_ground_truth(*results, ground_truth=None, plot_interval=None, plot_title=None,
                                     xlabel=None, ylabel=None, figure_path="./", figure_name=None, save_only=False
                                     ):
    """
    Plot predictions vs. ground truth for multiple results.

    If ground_truth is None, then the code should check if "ground_truth_y" is not None in the first result,
    i.e. results[0]["ground_truth_y"], and use it instead. If ground_truth is None and there is no data in
    results[0]["ground_truth_y"] (also None) then it raises an error stating that the ground truth y is None.

    The plot_interval parameter is a tuple (start, end) that determines when to start and stop plotting predictions.

    If save_only is True, then a figure will be saved at the specified path
    """

    # check if the results are all prequential
    for result in results:
        if not isinstance(result, PrequentialResults):
            raise ValueError('only can process PrequantialResults class.')

    # Determine ground truth y
    if ground_truth is None:
        if results and results[0].get_targets():
            ground_truth = results[0].get_targets()

    # Check if ground truth y is available
    if ground_truth is None:
        raise ValueError("Ground truth y is None.")

    # Create a figure
    plt.figure(figsize=(20, 6))

    # Determine indices to plot based on plot_interval
    start, end = plot_interval or (0, len(ground_truth))

    # Check if predictions have the same length as ground truth
    for i, result in enumerate(results):
        if result.get_predictions() is not None:
            predictions = result.get_predictions()[start:end]
            if len(predictions) != len(ground_truth[start:end]):
                raise ValueError(f"Length of predictions for result {i + 1} does not match ground truth.")

    # Plot ground truth y vs. predictions for each result within the specified interval
    instance_numbers = list(range(start, end))
    # for i, result in enumerate(results):
    #     if "predictions" in result:
    #         predictions = result["predictions"][start:end]
    for result in results:
        predictions = result.get_predictions()[start:end]
        plt.plot(instance_numbers, predictions, label=f"{result['learner']} predictions", alpha=0.7)

    # Plot ground truth y
    plt.scatter(instance_numbers, ground_truth[start:end], label="ground truth", marker='*', s=20, color='red')

    # TODO: Once Schema is updated to provide an easier access to the target name should remove direct access to MOA
    output_name = str(InstancesHeader.getClassNameString(results[0]['stream'].get_schema().get_moa_header()))
    output_name = output_name[output_name.find(":") + 1:-1]

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
        figure_name = figure_name if figure_name else f"predictions_vs_ground_truth_{current_time}.pdf"
        plt.savefig(figure_path + figure_name)


def plot_regression_results(
        # cope with data

        *results,  # results value from regression models
        ground_truth=None,  # stored ground truths
        start=0,  # the start point of plotting
        end=1E10,  # the end

        # options for users

        plot_target=True,
        plot_predictions=True,
        plot_residuals=True,

        # target_type='line',  # line or dots
        add_target_markers=True,
        target_marker='*',  # can be any markers supported by matplotlib

        predictions_type='dots',  # line or dots
        predictions_marker='.',  # can be any markers supported by matplotlib

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
    # check if the results are all prequential
    for result in results:
        if not isinstance(result, PrequentialResults):
            raise ValueError('only can process PrequantialResults class.')

    # Check if the ground_truth is stored in the first result
    if ground_truth is None:
        if results and results[0].get_targets():
            ground_truth = results[0].get_targets()

    # Check if ground_truth is none
    if ground_truth is None:
        raise ValueError("Ground truth y is None.")

    # Check for plotting interval
    start = max(start, 0)
    end = min(end, len(ground_truth))

    # Get stream
    stream = results[0]["stream"]

    # Get ground truth
    targets = ground_truth[start:end]

    predictions = []
    residuals = []
    if absolute_residuals:
        absolute_values = []
    for i, result in enumerate(results):
        if result.get_predictions() is not None:
            predictions.append(np.array(result.get_predictions()[start:end]))
            residuals.append(np.array(np.array(result.get_predictions()[start:end]) - np.array(targets)))
            if absolute_residuals:
                absolute_values.append(np.abs(np.array(np.array(result.get_predictions()[start:end]) - np.array(targets))))

    # Create a figure
    plt.figure(figsize=((end - start) / 10, 6))
    # x-axis
    instance_numbers = list(range(start, end))

    # get default colors from matplotlib for further possible use
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # plot targets
    if plot_target:
        plt.plot(instance_numbers, targets, label="targets", linewidth=1,
                     color=color_target if color_target is not None else "g")
    if add_target_markers:
        plt.scatter(instance_numbers, targets, label="targets", marker=target_marker, s=20,
                        color=color_target if color_target is not None else "g")

    # plot predictions
    if plot_predictions:
        for i, prediction in enumerate(predictions):
            if predictions_type == 'line':
                plt.plot(instance_numbers, predictions[i],
                         label=results[i]['learner'] + " predictions",
                         color=color_predictions[i] if color_predictions is not None else default_colors[i],
                         linewidth=1, linestyle="--", alpha=0.5)
            elif predictions_type == 'dots':
                plt.scatter(instance_numbers, predictions[i],
                            label=results[i]['learner'] + " predictions",
                            color=color_predictions[i] if color_predictions is not None else default_colors[i],
                            marker=predictions_marker, s=20)
            else:
                raise ValueError("Predictions_type must be 'line' or 'dots'.")

    if predictions_type == 'dots':
        if len(results) > 2 :
            plot_residuals = False

        for i in range(len(instance_numbers)):
            values = [predictions[x][i] for x in range(len(predictions))]
            values.append(targets[i])
            values = np.array(values)
            plt.vlines(x=instance_numbers[i], ymin=min(values), ymax=max(values), linestyles='dashed', colors='grey',
                       linewidth=0.5)

    # plot residuals
    if plot_residuals:
        for i, residual in enumerate(residuals):
            plt.bar(instance_numbers, residuals[i] if not absolute_residuals else absolute_values[i],
                    label=results[i]['learner'] + " residuals" if not absolute_residuals else " absolute residuals",
                    color=color_predictions[i] if color_predictions is not None else default_colors[i], alpha=0.5)

    if stream is not None and isinstance(stream, DriftStream):
        if not prevent_plotting_drifts:
            drifts = stream.get_drifts()

            drift_locations = [drift.position for drift in drifts]
            gradual_drift_window_lengths = [drift.width for drift in drifts]

            # Add vertical lines at drift locations
            if drift_locations:
                for location in drift_locations:
                    if start < location < end:
                        plt.axvline(location, color="red", linestyle="-")

            # Add gradual drift windows as 70% transparent rectangles
            if gradual_drift_window_lengths:
                if not drift_locations:
                    print(
                        "Error: gradual_drift_window_lengths is provided, but drift_locations is not."
                    )
                    return

                if len(drift_locations) != len(gradual_drift_window_lengths):
                    print(
                        "Error: drift_locations and gradual_drift_window_lengths must have the same length."
                    )
                    return

                for i in range(len(drift_locations)):
                    location = drift_locations[i]
                    window_length = gradual_drift_window_lengths[i]

                    # Plot the 70% transparent rectangle
                    if start < location < end:
                        plt.axvspan(
                            max(location - window_length / 2, start),
                            min(location + window_length / 2, end),
                            alpha=0.2,
                            color="red",
                        )

    output_name = str(InstancesHeader.getClassNameString(results[0]['stream'].get_schema().get_moa_header()))
    output_name = output_name[output_name.find(":") + 1:-1]

    prepared_title = ''
    fragments = []
    if plot_predictions:
        fragments.append("Predictions")
    if plot_target:
        fragments.append("Targets")
    if plot_residuals:
        fragments.append("Residuals" if not absolute_residuals else " Absolute Residuals")
    if len(fragments) > 0:
        for i, s in enumerate(fragments):
            prepared_title += s
            if i < len(fragments) - 1:
                prepared_title += " vs. "
    else:
        raise ValueError("Nothing to plot")

    # Add labels and title
    sns.set_style('darkgrid')
    plt.xlabel(xlabel if xlabel else "# Instance")
    plt.ylabel(ylabel if ylabel else output_name)
    plt.title(plot_title if plot_title else prepared_title)
    plt.legend()

    # Show the plot or save it to the specified path
    if not save_only:
        plt.show()
    elif figure_path:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        figure_name = figure_name if figure_name is not None else f"sequential_regression_results_{current_time}.pdf"
        plt.savefig(figure_path + figure_name)

    # plot bar plot for residuals
    if plot_hist_residuals:
        plt.figure(figsize=(8, 6))
        for i, residual in enumerate(residuals):
            sns.histplot(residual, kde=kde_residuals, bins='auto' if hist_bins is None else hist_bins,
                         label=results[i]['learner'],
                         color=color_predictions[i] if color_predictions is not None else default_colors[i],
                         alpha=0.5)

        sns.set_style("darkgrid")
        plt.title("Residuals Histogram Plot")
        plt.legend()

        # Show the plot or save it to the specified path
        if not save_only:
            plt.show()
        elif figure_path:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            figure_name_hist = figure_name_hist if figure_name_hist is not None else f"histogram_for_residuals_{current_time}.pdf"
            plt.savefig(figure_path + figure_name_hist)


def plot_prediction_interval(
        *results, ground_truth=None,
        start=0, end=1E10,

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
        if not isinstance(result, PrequentialResults):
            raise ValueError('only can process PrequantialResults class.')

    if len(results) > 2:
        raise ValueError('this function only supports up to 2 results currently.')

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    stream = results[0]["stream"]

    if len(results) == 1:
        if results[0].get_targets() is not None:
            targets = results[0].get_targets()
        elif ground_truth is not None:
            targets = ground_truth
        else:
            raise ValueError("No ground truth Found.")

        start = max(start, 0)
        end = min(end, len(targets))

        instance_numbers = list(range(start, end))
        targets = targets[start:end]
        intervals = results[0].get_predictions()[start:end]
        upper = []
        lower = []
        predictions = []
        for interval in intervals:
            upper.append(interval[2])
            lower.append(interval[0])
            predictions.append(interval[1])

        plt.figure(figsize=((end - start) / 10, 6))

        if plot_bounds:
            u = np.array(upper)
            l = np.array(lower)
            plt.plot(instance_numbers, u, linewidth=0.1, alpha=0.2,
                     color=colors[0] if colors is not None else default_colors[0])
            plt.plot(instance_numbers, l, linewidth=0.1, alpha=0.2,
                     color=colors[0] if colors is not None else default_colors[0])
            plt.fill_between(instance_numbers, u, l, color=colors[0] if colors is not None else default_colors[0],
                             alpha=0.5, label=results[0]["learner"] + " interval")
        if plot_predictions:
            plt.plot(instance_numbers, np.array(predictions), linewidth=1, linestyle='-',
                     color=colors[0] if colors is not None else default_colors[0],
                     label=results[0]["learner"] + " predictions")
        if plot_truth:
            insideX = []
            insideY = []
            outsideX = []
            outsideY = []
            for i, v in enumerate(targets):
                if u[i] >= v >= l[i]:
                    insideX.append(instance_numbers[i])
                    insideY.append(v)
                else:
                    outsideX.append(instance_numbers[i])
                    outsideY.append(v)

            plt.scatter(np.array(insideX), np.array(insideY), marker='*',
                        color='g', label='Ground Truth (inner)')
            plt.scatter(np.array(outsideX), np.array(outsideY), marker='x',
                        color='r', label='Ground Truth (outer)')

        if stream is not None and isinstance(stream, DriftStream):
            if not prevent_plotting_drifts:
                drifts = stream.get_drifts()

                drift_locations = [drift.position for drift in drifts]
                gradual_drift_window_lengths = [drift.width for drift in drifts]

                # Add vertical lines at drift locations
                if drift_locations:
                    for location in drift_locations:
                        if start < location < end:
                            plt.axvline(location, color="red", linestyle="-")

                # Add gradual drift windows as 70% transparent rectangles
                if gradual_drift_window_lengths:
                    if not drift_locations:
                        print(
                            "Error: gradual_drift_window_lengths is provided, but drift_locations is not."
                        )
                        return

                    if len(drift_locations) != len(gradual_drift_window_lengths):
                        print(
                            "Error: drift_locations and gradual_drift_window_lengths must have the same length."
                        )
                        return

                    for i in range(len(drift_locations)):
                        location = drift_locations[i]
                        window_length = gradual_drift_window_lengths[i]

                        # Plot the 70% transparent rectangle
                        if start < location < end:
                            plt.axvspan(
                                max(location - window_length / 2, start),
                                min(location + window_length / 2, end),
                                alpha=0.2,
                                color="red",
                            )

        output_name = str(InstancesHeader.getClassNameString(results[0]['stream'].get_schema().get_moa_header()))
        output_name = output_name[output_name.find(":") + 1:-1]

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
            figure_name = figure_name if figure_name else f"prediction_interval_over_time_{current_time}.pdf"
            plt.savefig(figure_path + figure_name)

    # Plots two regions from prediction interval learners for comparison
    elif len(results) == 2:
        if results[0].get_targets() is not None:
            targets = results[0].get_targets()
        elif ground_truth is not None:
            targets = ground_truth
        else:
            raise ValueError("No ground truth Found.")

        start = max(start, 0)
        end = min(end, len(targets))

        instance_numbers = list(range(start, end))
        targets = targets[start:end]

        intervals_first = results[0].get_predictions()[start:end]
        intervals_second = results[1].get_predictions()[start:end]

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
                plt.plot(instance_numbers, u_first, linewidth=0.1, alpha=0.2,
                         color=colors[0] if colors is not None else default_colors[0])
                plt.plot(instance_numbers, l_first, linewidth=0.1, alpha=0.2,
                         color=colors[0] if colors is not None else default_colors[0])
                plt.fill_between(instance_numbers, u_first, l_first,
                                 color=colors[0] if colors is not None else default_colors[0],
                                 alpha=0.2, label=results[0]["learner"] + " interval")

                # Plot second area
                plt.plot(instance_numbers, u_second, linewidth=0.1, alpha=0.5,
                         color=colors[1] if colors is not None else default_colors[1])
                plt.plot(instance_numbers, l_second, linewidth=0.1, alpha=0.5,
                         color=colors[1] if colors is not None else default_colors[1])
                plt.fill_between(instance_numbers, u_second, l_second,
                                 color=colors[1] if colors is not None else default_colors[1],
                                 alpha=0.5, label=results[1]["learner"] + " interval")
            else:
                # define function for further dynamic plot
                def _plot_first(i, alpha):
                    plt.plot(instance_numbers[switch_points[i]:switch_points[i+1]+1],
                             u_first[switch_points[i]:switch_points[i + 1]+1], linewidth=0.1, alpha=alpha,
                             color=colors[0] if colors is not None else default_colors[0])
                    plt.plot(instance_numbers[switch_points[i]:switch_points[i + 1]+1],
                             l_first[switch_points[i]:switch_points[i + 1]+1], linewidth=0.1, alpha=alpha,
                             color=colors[0] if colors is not None else default_colors[0])

                    plt.fill_between(instance_numbers[switch_points[i]:switch_points[i + 1]+1],
                                     u_first[switch_points[i]:switch_points[i + 1]+1],
                                     l_first[switch_points[i]:switch_points[i + 1]+1],
                                     color=colors[0] if colors is not None else default_colors[0],
                                     alpha=alpha, label=results[0]["learner"] + " interval" if i == 0 else "")

                def _plot_second(i, alpha):
                    plt.plot(instance_numbers[switch_points[i]:switch_points[i + 1]+1],
                             u_second[switch_points[i]:switch_points[i + 1]+1], linewidth=0.1, alpha=alpha,
                             color=colors[1] if colors is not None else default_colors[1])
                    plt.plot(instance_numbers[switch_points[i]:switch_points[i + 1]+1],
                             l_second[switch_points[i]:switch_points[i + 1]+1], linewidth=0.1, alpha=alpha,
                             color=colors[1] if colors is not None else default_colors[1])

                    plt.fill_between(instance_numbers[switch_points[i ]:switch_points[i + 1]+1],
                                     u_second[switch_points[i]:switch_points[i + 1]+1],
                                     l_second[switch_points[i]:switch_points[i + 1]+1],
                                     color=colors[1] if colors is not None else default_colors[1],
                                     alpha=alpha, label=results[1]["learner"] + " interval" if i == 0 else "")

                # determine which on top first
                first_first = l_first[0] > l_second[0]
                # find the switch point
                larger = True
                switch_points = [0]
                for i in range(0, len(l_first)):
                    if larger:
                        if l_first[i] < l_second[i]:
                            switch_points.append(i)
                            larger = not larger
                    else:
                        if l_first[i] > l_second[i]:
                            switch_points.append(i)
                            larger = not larger
                switch_points.append(len(u_first) -1)

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
            plt.plot(instance_numbers, np.array(predictions_first), linewidth=1, linestyle='-',
                     color=colors[0] if colors is not None else default_colors[0],
                     label=results[0]["learner"] + " predictions")
            plt.plot(instance_numbers, np.array(predictions_second), linewidth=1, linestyle='-',
                     color=colors[1] if colors is not None else default_colors[1],
                     label=results[1]["learner"] + " predictions")

        if plot_truth:
            insideX = []
            insideY = []
            betweenX = []
            betweenY = []
            outsideX = []
            outsideY = []

            for i, v in enumerate(targets):
                _out = v >= max(upper_first[i], upper_second[i]) or v <= min(lower_first[i], lower_second[i])
                _in = min(upper_first[i], upper_second[i]) >= v >= max(lower_first[i], lower_second[i])
                if _out:
                    outsideX.append(instance_numbers[i])
                    outsideY.append(v)
                elif _in:
                    insideX.append(instance_numbers[i])
                    insideY.append(v)
                else:
                    betweenX.append(instance_numbers[i])
                    betweenY.append(v)

            plt.scatter(np.array(insideX), np.array(insideY), marker='*',
                        color='g', label='Ground Truth (inner)')
            plt.scatter(np.array(outsideX), np.array(outsideY), marker='x',
                        color='r', label='Ground Truth (outer)')
            plt.scatter(np.array(betweenX), np.array(betweenY), marker='+',
                        color='orange', label='Ground Truth (interim)')

        if stream is not None and isinstance(stream, DriftStream):
            if not prevent_plotting_drifts:
                drifts = stream.get_drifts()

                drift_locations = [drift.position for drift in drifts]
                gradual_drift_window_lengths = [drift.width for drift in drifts]

                # Add vertical lines at drift locations
                if drift_locations:
                    for location in drift_locations:
                        if start < location < end:
                            plt.axvline(location, color="red", linestyle="-")

                # Add gradual drift windows as 70% transparent rectangles
                if gradual_drift_window_lengths:
                    if not drift_locations:
                        print(
                            "Error: gradual_drift_window_lengths is provided, but drift_locations is not."
                        )
                        return

                    if len(drift_locations) != len(gradual_drift_window_lengths):
                        print(
                            "Error: drift_locations and gradual_drift_window_lengths must have the same length."
                        )
                        return

                    for i in range(len(drift_locations)):
                        location = drift_locations[i]
                        window_length = gradual_drift_window_lengths[i]

                        # Plot the 70% transparent rectangle
                        if start < location < end:
                            plt.axvspan(
                                max(location - window_length / 2, start),
                                min(location + window_length / 2, end),
                                alpha=0.2,
                                color="red",
                            )

        output_name = str(InstancesHeader.getClassNameString(results[0]['stream'].get_schema().get_moa_header()))
        output_name = output_name[output_name.find(":") + 1:-1]

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
            figure_name = figure_name if figure_name else f"prediction_interval_over_time_comparison_{current_time}.pdf"
            plt.savefig(figure_path + figure_name)
