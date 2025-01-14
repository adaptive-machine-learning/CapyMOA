# Python imports
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# river imports
from river import stream as stream_river, metrics
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier, ExtremelyFastDecisionTreeClassifier
from river.neighbors import KNNClassifier, LazySearch
from river.forest import ARFClassifier

# Library imports
from capymoa.evaluation.evaluation import (
    test_then_train_evaluation,
    start_time_measuring,
    stop_time_measuring,
)
from capymoa.datasets import RTG_2abrupt
from capymoa.classifier import (
    NaiveBayes,
    HoeffdingTree,
    EFDT,
    KNN,
    AdaptiveRandomForestClassifier,
)

# Globals
DT_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_OUTPUT_CSV = f"./benchmark_{DT_stamp}.csv"
RAW_RESULTS_OUTPUT_CSV = f"./benchmark_{DT_stamp}_raw.csv"

# To check if the benchmark is running appropriately until the end, you might want to set this to a lower value.
MAX_INSTANCES = 100

# Data path for river (csv file), can be downloaded using the cli tool
# See python -m capymoa.datasets --help, python -m capymoa.datasets -d RTG_2abrupt -o "../data/"
csv_RTG_2abrupt_path = "../data/RTG_2abrupt.csv"


# Save combined results to output file
def checkpoint_results(results, new_result, output_file):
    results = pd.concat([results, new_result], ignore_index=True)
    results.to_csv(output_file, index=False)
    return results


def capymoa_experiment(
    dataset_name, learner_name, stream, learner, hyperparameters={}, repetitions=1
):
    date_time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
    print(f"[{date_time_stamp}][capymoa] Executing {learner_name} on {dataset_name}")

    results = []
    raw_results = []  # Store raw results for each repetition

    repetition = 1
    for _ in range(repetitions):
        print(f"[{date_time_stamp}][capymoa]\trepetition {repetition}")

        stream.restart()
        result = test_then_train_evaluation(
            stream=stream,
            learner=learner(**hyperparameters, schema=stream.get_schema()),
            max_instances=MAX_INSTANCES,
            sample_frequency=MAX_INSTANCES,
        )
        results.append(result)

        # Append raw result to list
        raw_results.append(
            {
                "library": "capymoa",
                "repetition": repetition,
                "dataset": dataset_name,
                "learner": learner_name,
                "hyperparameters": str(hyperparameters),
                "accuracy": result["cumulative"].accuracy(),
                "wallclock": result["wallclock"],
                "cpu_time": result["cpu_time"],
            }
        )
        repetition += 1

    # Calculate average and std for accuracy, wallclock, and cpu_time
    avg_accuracy = (
        sum(result["cumulative"].accuracy() for result in results) / repetitions
    )
    std_accuracy = pd.Series(
        [result["cumulative"].accuracy() for result in results]
    ).std()
    avg_wallclock = sum(result["wallclock"] for result in results) / repetitions
    std_wallclock = pd.Series([result["wallclock"] for result in results]).std()
    avg_cpu_time = sum(result["cpu_time"] for result in results) / repetitions
    std_cpu_time = pd.Series([result["cpu_time"] for result in results]).std()

    # Create DataFrame for aggregated results
    df = pd.DataFrame(
        {
            "library": "capymoa",
            "dataset": dataset_name,
            "learner": learner_name,
            "hyperparameters": str(hyperparameters),
            "repetitions": repetitions,
            "avg_accuracy": avg_accuracy,
            "std_accuracy": std_accuracy,
            "avg_wallclock": avg_wallclock,
            "std_wallclock": std_wallclock,
            "avg_cpu_time": avg_cpu_time,
            "std_cpu_time": std_cpu_time,
        },
        index=[0],
    )  # Single row

    # Create DataFrame for raw results
    raw_df = pd.DataFrame(raw_results)

    return df, raw_df


def test_then_train_river(stream_data, model, max_instances=MAX_INSTANCES):
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    instancesProcessed = 1
    accuracy = metrics.Accuracy()

    X, Y = stream_data[:, :-1], stream_data[:, -1]

    data = []
    performance_names = ["Classified instances", "accuracy"]

    ds = stream_river.iter_array(X, Y)

    for x, y in ds:
        if instancesProcessed > max_instances:
            break
        yp = model.predict_one(x)
        accuracy.update(y, yp)
        model.learn_one(x, y)

        instancesProcessed += 1

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    return (
        accuracy.get(),
        elapsed_wallclock_time,
        elapsed_cpu_time,
        pd.DataFrame(data, columns=performance_names),
    )


def river_experiment(
    dataset_name,
    learner_name,
    stream_path_csv,
    learner,
    hyperparameters={},
    repetitions=1,
):
    date_time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
    print(f"[{date_time_stamp}][river] Executing {learner_name} on {dataset_name}")

    raw_results = []  # Store raw results for each repetition

    repetition = 1
    for _ in range(repetitions):
        print(f"[{date_time_stamp}][river]\trepetition {repetition}")
        stream_data = pd.read_csv(stream_path_csv).to_numpy()

        model_instance = learner(**hyperparameters)
        acc, wallclock, cpu_time, df_raw = test_then_train_river(
            stream_data=stream_data, model=model_instance
        )

        # Append raw result to list
        raw_results.append(
            {
                "library": "river",
                "repetition": repetition,
                "dataset": dataset_name,
                "learner": learner_name,
                "hyperparameters": str(hyperparameters).replace("\n", ""),
                "accuracy": acc,
                "wallclock": wallclock,
                "cpu_time": cpu_time,
            }
        )
        repetition += 1

    # Calculate average and std for accuracy, wallclock, and cpu_time
    avg_accuracy = sum(result["accuracy"] for result in raw_results) / repetitions
    std_accuracy = pd.Series([result["accuracy"] for result in raw_results]).std()
    avg_wallclock = sum(result["wallclock"] for result in raw_results) / repetitions
    std_wallclock = pd.Series([result["wallclock"] for result in raw_results]).std()
    avg_cpu_time = sum(result["cpu_time"] for result in raw_results) / repetitions
    std_cpu_time = pd.Series([result["cpu_time"] for result in raw_results]).std()

    # Create DataFrame for aggregated results
    df_aggregated = pd.DataFrame(
        {
            "library": "river",
            "dataset": dataset_name,
            "learner": learner_name,
            "hyperparameters": str(hyperparameters),
            "repetitions": repetitions,
            "avg_accuracy": avg_accuracy * 100,  # changing the range to 0 to 100
            "std_accuracy": std_accuracy * 100,  # changing the range to 0 to 100
            "avg_wallclock": avg_wallclock,
            "std_wallclock": std_wallclock,
            "avg_cpu_time": avg_cpu_time,
            "std_cpu_time": std_cpu_time,
        },
        index=[0],
    )  # Single row

    # Create DataFrame for raw results
    df_raw = pd.DataFrame(raw_results)

    return df_aggregated, df_raw


def benchmark_classifiers_capymoa(
    intermediary_results,
    raw_intermediary_results,
    data,
    dataset_names,
    results_output_csv,
    raw_results_output_csv,
):
    # Run experiment 1
    result_capyNB, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="NaiveBayes",
        stream=data,
        learner=NaiveBayes,
        hyperparameters={},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_capyNB, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_capymoa, raw_results_output_csv
    )

    # Run experiment 2
    result_capyHT, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="HT",
        stream=data,
        learner=HoeffdingTree,
        hyperparameters={},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_capyHT, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_capymoa, raw_results_output_csv
    )

    # Run experiment 3
    result_capyEFDT, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="EFDT",
        stream=data,
        learner=EFDT,
        hyperparameters={},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_capyEFDT, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_capymoa, raw_results_output_csv
    )

    # Run experiment 4
    result_capyKNN, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="kNN",
        stream=data,
        learner=KNN,
        hyperparameters={"window_size": 1000, "k": 3},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_capyKNN, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_capymoa, raw_results_output_csv
    )

    # Run experiment 5
    result_capyARF5, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="ARF5",
        stream=data,
        learner=AdaptiveRandomForestClassifier,
        hyperparameters={"ensemble_size": 5, "max_features": 0.6},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_capyARF5, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_capymoa, raw_results_output_csv
    )

    # Run experiment 6
    result_capyARF10, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="ARF10",
        stream=data,
        learner=AdaptiveRandomForestClassifier,
        hyperparameters={"ensemble_size": 10, "max_features": 0.6},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_capyARF10, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_capymoa, raw_results_output_csv
    )

    # Run experiment 7
    result_capyARF30, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="ARF30",
        stream=data,
        learner=AdaptiveRandomForestClassifier,
        hyperparameters={"ensemble_size": 30, "max_features": 0.6},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_capyARF30, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_capymoa, raw_results_output_csv
    )

    # Run experiment 8
    result_capyARF100, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="ARF100",
        stream=data,
        learner=AdaptiveRandomForestClassifier,
        hyperparameters={"ensemble_size": 100, "max_features": 0.6},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_capyARF100, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_capymoa, raw_results_output_csv
    )

    # Run experiment 9
    result_capyARF100j4, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="ARF100j4",
        stream=data,
        learner=AdaptiveRandomForestClassifier,
        hyperparameters={
            "ensemble_size": 100,
            "max_features": 0.6,
            "number_of_jobs": 4,
        },
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_capyARF100j4, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_capymoa, raw_results_output_csv
    )

    return intermediary_results, raw_intermediary_results


def benchmark_classifiers_river(
    intermediary_results,
    raw_intermediary_results,
    stream_path_csv,
    dataset_names,
    results_output_csv,
    raw_results_output_csv,
):
    # Run experiment 1
    result_riverNB, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="NaiveBayes",
        stream_path_csv=stream_path_csv,
        learner=GaussianNB,
        hyperparameters={},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_riverNB, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_river, raw_results_output_csv
    )

    # Run experiment 2
    result_riverHT, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="HT",
        stream_path_csv=stream_path_csv,
        learner=HoeffdingTreeClassifier,
        hyperparameters={},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_riverHT, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_river, raw_results_output_csv
    )

    # Run experiment 3
    result_riverEFDT, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="EFDT",
        stream_path_csv=stream_path_csv,
        learner=ExtremelyFastDecisionTreeClassifier,
        hyperparameters={},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_riverEFDT, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_river, raw_results_output_csv
    )

    # Run experiment 4
    result_riverKNN, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="KNN",
        stream_path_csv=stream_path_csv,
        learner=KNNClassifier,
        hyperparameters={"engine": LazySearch(window_size=1000), "n_neighbors": 3},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_riverKNN, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_river, raw_results_output_csv
    )

    # Run experiment 5
    result_riverARF5, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="ARF5",
        stream_path_csv=stream_path_csv,
        learner=ARFClassifier,
        hyperparameters={"n_models": 5, "max_features": 0.60},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_riverARF5, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_river, raw_results_output_csv
    )

    # Run experiment 6
    result_riverARF10, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="ARF10",
        stream_path_csv=stream_path_csv,
        learner=ARFClassifier,
        hyperparameters={"n_models": 10, "max_features": 0.60},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_riverARF10, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_river, raw_results_output_csv
    )

    # Run experiment 7
    result_riverARF30, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="ARF30",
        stream_path_csv=stream_path_csv,
        learner=ARFClassifier,
        hyperparameters={"n_models": 30, "max_features": 0.60},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_riverARF30, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_river, raw_results_output_csv
    )

    # Run experiment 8
    result_riverARF100, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="ARF100",
        stream_path_csv=stream_path_csv,
        learner=ARFClassifier,
        hyperparameters={"n_models": 100, "max_features": 0.60},
        repetitions=5,
    )

    intermediary_results = checkpoint_results(
        intermediary_results, result_riverARF100, results_output_csv
    )
    raw_intermediary_results = checkpoint_results(
        raw_intermediary_results, raw_river, raw_results_output_csv
    )

    return intermediary_results, raw_intermediary_results


def plot_performance(df, plot_prefix):
    # Step 1: Filter and reorder data
    ordered_algorithms = [
        "NaiveBayes",
        "HT",
        "EFDT",
        "kNN",
        "ARF5",
        "ARF10",
        "ARF30",
        "ARF100",
        "ARF100j4",
    ]
    df["learner"] = pd.Categorical(df["learner"], ordered_algorithms)
    df = df.sort_values("learner")

    # Step 2: Check if there are results for both libraries
    capymoa_data = df[df["library"] == "capymoa"]
    river_data = df[df["library"] == "river"]

    if len(capymoa_data) == 0 or len(river_data) == 0:
        print("Results not available for both libraries.")
        return

    # Step 3: Get common algorithms
    common_algorithms = set(capymoa_data["learner"]).intersection(
        set(river_data["learner"])
    )

    # Step 4: Plot each measure
    measures = ["accuracy", "wallclock", "cpu_time"]
    for measure in measures:
        plt.figure(figsize=(10, 6))
        plt.title(f"{plot_prefix}_{measure}")
        plt.xlabel("Algorithm")
        plt.ylabel(measure.capitalize())
        plt.xticks(rotation=45)

        # Capymoa bars
        capymoa_means = capymoa_data[capymoa_data["learner"].isin(common_algorithms)][
            f"avg_{measure}"
        ]
        capymoa_std = capymoa_data[capymoa_data["learner"].isin(common_algorithms)][
            f"std_{measure}"
        ]
        capymoa_colors = ["green" for _ in range(len(capymoa_means))]
        capymoa_positions = np.arange(len(capymoa_means))
        plt.bar(
            capymoa_positions - 0.2,
            capymoa_means,
            yerr=capymoa_std,
            width=0.4,
            color=capymoa_colors,
            label="capymoa",
        )

        # River bars
        river_means = river_data[river_data["learner"].isin(common_algorithms)][
            f"avg_{measure}"
        ]
        river_std = river_data[river_data["learner"].isin(common_algorithms)][
            f"std_{measure}"
        ]
        river_colors = ["red" for _ in range(len(river_means))]
        river_positions = np.arange(len(river_means))
        plt.bar(
            river_positions + 0.2,
            river_means,
            yerr=river_std,
            width=0.4,
            color=river_colors,
            label="river",
        )

        # X-axis labels
        algorithm_names = capymoa_data[capymoa_data["learner"].isin(common_algorithms)][
            "learner"
        ]
        plt.xticks(range(len(algorithm_names)), algorithm_names)

        # Step 5: Customize plot
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_{measure}.png")
        # plt.show()


if __name__ == "__main__":
    # Initialize an empty DataFrame for combined_results
    combined_results = pd.DataFrame()
    raw_results = pd.DataFrame()

    rtg2abrupt_stream = RTG_2abrupt()

    combined_results, raw_results = benchmark_classifiers_capymoa(
        intermediary_results=combined_results,
        raw_intermediary_results=raw_results,
        data=rtg2abrupt_stream,
        dataset_names="RTG2Abrupt",
        results_output_csv=RESULTS_OUTPUT_CSV,
        raw_results_output_csv=RAW_RESULTS_OUTPUT_CSV,
    )

    combined_results, raw_results = benchmark_classifiers_river(
        intermediary_results=combined_results,
        raw_intermediary_results=raw_results,
        stream_path_csv=csv_RTG_2abrupt_path,
        dataset_names="RTG2Abrupt",
        results_output_csv=RESULTS_OUTPUT_CSV,
        raw_results_output_csv=RAW_RESULTS_OUTPUT_CSV,
    )

    plot_performance(combined_results, f"benchmark_{DT_stamp}_performance_plot")

    print(combined_results)
