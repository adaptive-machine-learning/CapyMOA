# Python imports
import pandas as pd
from datetime import datetime
from typing import Literal, Union
import numpy as np

# river imports
from river import stream as stream_river, metrics

from capymoa.evaluation.evaluation import start_time_measuring, stop_time_measuring

_cla_metric_names = ['Accuracy', 'CohenKappa', 'Recall', 'Precision', 'F1']
_reg_metric_names = ['RMSE', 'MAE', 'R2']
_anomaly_metric_names = ['ROCAUC']

_capymoa_river_name_converter_dict = {
    "Accuracy": "accuracy",
    "CohenKappa": "kappa",
    "Recall": "recall",
    "Precision": "precision",
    "F1": "f1_score",
    "RMSE": "rmse",
    "MAE": "mae",
    "R2": "r2",
    "ROCAUC": "auc"
}


def _test_then_train_river(
        task_type,
        stream_data,
        model,
        max_instances=1000000,
):
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    instancesProcessed = 1

    metric_names = (
        _cla_metric_names if task_type == "classification" else
        _reg_metric_names if task_type == "regression" else
        _anomaly_metric_names if task_type == "anomaly_detection" else
        None
    )
    functions = {_capymoa_river_name_converter_dict[name]: name for name in metric_names }

    for name in metric_names:
        func = getattr(metrics, name)
        functions[_capymoa_river_name_converter_dict[name]] = func()
    # accuracy = metrics.Accuracy()

    X, Y = stream_data[:, :-1], stream_data[:, -1]

    data = []
    performance_names = ["Classified instances"] + list(functions.values())
    performance_values = []

    ds = stream_river.iter_array(X, Y)

    for x, y in ds:
        if max_instances is not None and instancesProcessed > max_instances:
            break
        yp = model.predict_one(x)
        for name in metric_names:
            func = getattr(functions[_capymoa_river_name_converter_dict[name]], 'update')
            func(y, yp)
        # accuracy.update(y, yp)
        model.learn_one(x, y)

        instancesProcessed += 1

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    return (
        {name: getattr(functions[_capymoa_river_name_converter_dict[name]], 'get')() for name in metric_names},
        elapsed_wallclock_time,
        elapsed_cpu_time,
        pd.DataFrame(data, columns=performance_names),
    )


def river_experiment(
        task_type: Literal["classification", "regression", "anomaly_detection"],
        dataset_name,
        learner_name,
        stream_path_csv,
        learner,
        hyperparameters={},
        repetitions=1,
        max_instances=1000000,
        show_progress=True,
        **kwargs,
):
    date_time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
    if show_progress:
        print(f"[{date_time_stamp}][river] Executing {learner_name} on {dataset_name}")

    raw_results = []  # Store raw results for each repetition

    metric_names = (
        _cla_metric_names if task_type == "classification" else
        _reg_metric_names if task_type == "regression" else
        _anomaly_metric_names if task_type == "anomaly_detection" else
        None
    )

    repetition = 1
    for _ in range(repetitions):
        if show_progress:
            print(f"[{date_time_stamp}][river]\trepetition {repetition}")
        stream_data = pd.read_csv(stream_path_csv).to_numpy()

        if 'seed' not in list(hyperparameters.keys()):
            hyperparameters['seed'] = repetition
        model_instance = learner(**hyperparameters)
        results, wallclock, cpu_time, df_raw = _test_then_train_river(
            stream_data=stream_data,
            model=model_instance,
            max_instances=max_instances,
            task_type=task_type,
        )

        # Append raw result to list
        raw_result_dict = {
            "library": "river",
            "repetition": repetition,
            "dataset": dataset_name,
            "learner": learner_name,
            "hyperparameters": str(hyperparameters).replace("\n", ""),
            "wallclock": wallclock,
            "cpu_time": cpu_time
        }
        for name, value in results.items():
            raw_result_dict[_capymoa_river_name_converter_dict[name]] = value
        raw_results.append(raw_result_dict)
        repetition += 1

    # Calculate average and std for accuracy, wallclock, and cpu_time
    avg_metrics = {name: 'avg'+_capymoa_river_name_converter_dict[name] for name in metric_names}
    std_metrics = {name: 'std'+_capymoa_river_name_converter_dict[name] for name in metric_names}
    for name in metric_names:
        avg_metrics[name] = pd.Series([result[_capymoa_river_name_converter_dict[name]] for result in raw_results]).mean()
        std_metrics[name] = pd.Series([result[_capymoa_river_name_converter_dict[name]] for result in raw_results]).std()
    avg_wallclock = sum(result['wallclock'] for result in raw_results) / repetitions
    std_wallclock = pd.Series([result['wallclock'] for result in raw_results]).std()
    avg_cpu_time = sum(result['cpu_time'] for result in raw_results) / repetitions
    std_cpu_time = pd.Series([result['cpu_time'] for result in raw_results]).std()

    # Remove the random seed hyperparameter for the aggregated results
    hyperparameters.pop('seed', hyperparameters)
    # Create DataFrame for aggregated results
    aggregated_result_dict = {
        "library": "river",
        "dataset": dataset_name,
        "learner": learner_name,
        "hyperparameters": str(hyperparameters),
        "repetitions": repetitions,
        "avg_wallclock": avg_wallclock,
        "std_wallclock": std_wallclock,
        "avg_cpu_time": avg_cpu_time,
        "std_cpu_time": std_cpu_time
    }
    for name in metric_names:
        aggregated_result_dict[f"avg_{_capymoa_river_name_converter_dict[name]}"] = avg_metrics[name]
        aggregated_result_dict[f"std_{_capymoa_river_name_converter_dict[name]}"] = std_metrics[name]

    df_aggregated = pd.DataFrame(aggregated_result_dict, index=[0])  # Single row

    # Create DataFrame for raw results
    df_raw = pd.DataFrame(raw_results)

    return df_aggregated, df_raw