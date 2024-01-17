import matplotlib.pyplot as plt
import os
from capymoa.stream.stream import DriftStream

def plot_windowed_results(*results, metric="classifications correct (percent)", 
                            plot_title=None, xlabel=None, ylabel=None,
                            figure_path="./", figure_name=None, save_only=True
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
    
    num_instances = results[0].get('max_instances', None)
    stream = results[0].get('stream', None)

    if num_instances is not None:
        window_size = results[0]['windowed'].window_size
        num_windows = results[0]['windowed'].metrics_per_window().shape[0]
        x_values = []
        for i in range(1, num_windows+1):
            x_values.append(i * window_size)
        # print(f'x_values: {x_values}')

    # Check if the given metric exists in all DataFrames
    for result in results:
        df = result['windowed'].metrics_per_window()
        if metric not in df.columns:
            print(f"Column '{metric}' not found in metrics DataFrame for {result['learner']}. Skipping.")
        else:
            dfs.append(df)
            if 'experiment_id' in result:
                labels.append(result['experiment_id'])
            else:
                labels.append(result['learner'])
    
    if not dfs:
        print("No valid DataFrames to plot.")
        return
    
    # Create a figure
    plt.figure(figsize=(12, 5))

    # Plot data from each DataFrame
    for i, df in enumerate(dfs):
        # print(f'df.index: {df.index}')
        if num_instances is not None:
            plt.plot(x_values, df[metric], label=labels[i], marker='o', linestyle='-', markersize=5)
        else:
            plt.plot(df.index, df[metric], label=labels[i], marker='o', linestyle='-', markersize=5)


    if stream is not None and isinstance(stream, DriftStream):
        drifts = stream.get_drifts()

        drift_locations = [drift.position for drift in drifts]
        gradual_drift_window_lengths = [drift.width for drift in drifts]

        # Add vertical lines at drift locations
        if drift_locations:
            for location in drift_locations:
                plt.axvline(location, color='red', linestyle='-')
        
        # Add gradual drift windows as 70% transparent rectangles
        if gradual_drift_window_lengths:
            if not drift_locations:
                print("Error: gradual_drift_window_lengths is provided, but drift_locations is not.")
                return
            
            if len(drift_locations) != len(gradual_drift_window_lengths):
                print("Error: drift_locations and gradual_drift_window_lengths must have the same length.")
                return
            
            for i in range(len(drift_locations)):
                location = drift_locations[i]
                window_length = gradual_drift_window_lengths[i]
                
                # Plot the 70% transparent rectangle
                plt.axvspan(location - window_length / 2, location + window_length / 2, alpha=0.2, color='red')
    
    # Add labels and title
    xlabel = xlabel if xlabel is not None else '# Instances'
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
            figure_name = result['learner'] + "_" + ylabel.replace(' ', '')
        plt.savefig(figure_path + figure_name)
