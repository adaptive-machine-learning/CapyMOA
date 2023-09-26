import matplotlib.pyplot as plt


def plot_windowed_results(*results, metric="classifications correct (percent)"):
    """
    Plot a comparison of values from multiple evaluators based on a selected column using line plots.
    It assumes the results contain windowed results ('windowed') which often originate from metrics_per_window()
    and the learner identification ('learner'). 
    """
    dfs = []
    labels = []
    
    # Check if the given metric exists in all DataFrames
    for result in results:
        df = result['windowed'].metrics_per_window()
        if metric not in df.columns:
            print(f"Column '{metric}' not found in DataFrame for {result['learner']}. Skipping.")
        else:
            dfs.append(df)
            labels.append(result['learner'])
    
    if not dfs:
        print("No valid DataFrames to plot.")
        return
    
    # Create a figure
    plt.figure(figsize=(10, 5))
    
    # Plot data from each DataFrame
    for i, df in enumerate(dfs):
        plt.plot(df.index, df[metric], label=labels[i], marker='o', linestyle='-', markersize=5)
    
    # Add labels and title
    plt.xlabel('# Instances (thousands)')
    plt.ylabel(metric)
    plt.title(f'{metric}')
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.grid(True)
    plt.show()