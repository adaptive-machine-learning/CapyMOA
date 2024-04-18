import matplotlib.pyplot as plt
from datetime import datetime
from capymoa.stream.drift import DriftStream
from com.yahoo.labs.samoa.instances import InstancesHeader


def plot_windowed_results(
    *results,
    metric="classifications correct (percent)",
    plot_title=None,
    xlabel=None,
    ylabel=None,
    figure_path="./",
    figure_name=None,
    save_only=True,
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

    num_instances = results[0].get("max_instances", None)
    stream = results[0].get("stream", None)

    if num_instances is not None:
        window_size = results[0]["windowed"].window_size
        num_windows = results[0]["windowed"].metrics_per_window().shape[0]
        x_values = []
        for i in range(1, num_windows + 1):
            x_values.append(i * window_size)
        # print(f'x_values: {x_values}')

    # Check if the given metric exists in all DataFrames
    for result in results:
        df = result["windowed"].metrics_per_window()
        if metric not in df.columns:
            print(
                f"Column '{metric}' not found in metrics DataFrame for {result['learner']}. Skipping."
            )
        else:
            dfs.append(df)
            if "experiment_id" in result:
                labels.append(result["experiment_id"])
            else:
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
    # Determine ground truth y
    if ground_truth is None:
        if results and "ground_truth_y" in results[0]:
            ground_truth = results[0]["ground_truth_y"]

    # Check if ground truth y is available
    if ground_truth is None:
        raise ValueError("Ground truth y is None.")

    # Create a figure
    plt.figure(figsize=(20, 6))

    # Determine indices to plot based on plot_interval
    start, end = plot_interval or (0, len(ground_truth))

    # Check if predictions have the same length as ground truth
    for i, result in enumerate(results):
        if "predictions" in result:
            predictions = result["predictions"][start:end]
            if len(predictions) != len(ground_truth[start:end]):
                raise ValueError(f"Length of predictions for result {i + 1} does not match ground truth.")

    # Plot ground truth y vs. predictions for each result within the specified interval
    instance_numbers = list(range(start, end))
    for i, result in enumerate(results):
        if "predictions" in result:
            predictions = result["predictions"][start:end]
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

