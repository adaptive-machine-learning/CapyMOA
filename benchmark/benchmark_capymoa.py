from typing import Union

import pandas as pd
from datetime import datetime
from capymoa.evaluation import prequential_evaluation


# Function to execute a single experiment using CapyMOA
def capymoa_experiment(
        # task_type: Literal["classification", "regression", "anomaly_detection"],
        dataset_name,
        learner_name,
        stream,
        learner,
        window_size=1000,
        hyperparameters={},
        repetitions=1,
        max_instances=1000000,
        show_progress=True,
        **kwargs,

):
    date_time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")

    if show_progress:
        print(f"[{date_time_stamp}][capymoa] Executing {learner_name} on {dataset_name}")


    results = []
    raw_results = []  # Store raw results for each repetition

    repetition = 1
    for _ in range(repetitions):
        if show_progress:
            print(f"[{date_time_stamp}][capymoa]\trepetition {repetition}")

        stream.restart()
        if ['random_seed'] not in list(hyperparameters.keys()):
            hyperparameters['random_seed'] = repetition
        result = prequential_evaluation(
            stream=stream,
            learner=learner(**hyperparameters, schema=stream.get_schema()),
            window_size=window_size,
            max_instances=max_instances,
            store_y=False,
            store_predictions=False,
            optimise=True,
        )
        results.append(result)

        metric_names = result.cumulative.metrics_header()

        # Append raw result to list
        raw_result_dict = {
            "library": "capymoa",
            "repetition": repetition,
            "dataset": dataset_name,
            "learner": learner_name,
            "hyperparameters": str(hyperparameters),
            "wallclock": result['wallclock'],
            "cpu_time": result['cpu_time']
        }

        # raw_result_dict['instances'] = result.cumulative[metric_names[0]]
        for name in metric_names:
            func = getattr(result['cumulative'], name)
            raw_result_dict[name] = func()

        raw_results.append(raw_result_dict)
        repetition += 1

    classified_instance = results[0][metric_names[0]]

    avg_wallclock = sum(result['wallclock'] for result in results) / repetitions
    std_wallclock = pd.Series([result['wallclock'] for result in results]).std()
    avg_cpu_time = sum(result['cpu_time'] for result in results) / repetitions
    std_cpu_time = pd.Series([result['cpu_time'] for result in results]).std()

    avg_metrics = {name: 'avg'+name for name in metric_names[1:]}
    std_metrics = {name: 'std'+name for name in metric_names[1:]}
    for name in metric_names[1:]:
        avg_metrics[name] = pd.Series([getattr(result['cumulative'], name)() for result in results]).mean()
        std_metrics[name] = pd.Series([getattr(result['cumulative'], name)() for result in results]).std()

    # Remove the random seed hyperparameter for the aggregated results
    hyperparameters.pop('random_seed', hyperparameters)
    # Create DataFrame for aggregated results

    aggregated_result_dict = {
        "library": "capymoa",
        "dataset": dataset_name,
        "learner": learner_name,
        "hyperparameters": str(hyperparameters),
        "repetitions": repetitions,
        "instances": classified_instance,
        "avg_wallclock": avg_wallclock,
        "std_wallclock": std_wallclock,
        "avg_cpu_time": avg_cpu_time,
        "std_cpu_time": std_cpu_time,
    }
    for name in metric_names[1:]:
        aggregated_result_dict[f"avg_{name}"] = avg_metrics[name]
        aggregated_result_dict[f"std_{name}"] = std_metrics[name]

    df = pd.DataFrame(
        aggregated_result_dict
    , index=[0])  # Single row

    # Create DataFrame for raw results
    raw_df = pd.DataFrame(raw_results)

    return df, raw_df
