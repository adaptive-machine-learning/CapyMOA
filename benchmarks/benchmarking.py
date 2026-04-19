# Python imports
import argparse
from datetime import datetime
import os
from pathlib import Path
import platform
import sys
import time

import matplotlib
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# river imports
from river import stream as stream_river, metrics
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier, ExtremelyFastDecisionTreeClassifier
from river.neighbors import KNNClassifier, LazySearch
from river.forest import ARFClassifier

# Library imports
import capymoa.datasets as capymoa_datasets
from capymoa.evaluation.evaluation import (
    prequential_evaluation,
    start_time_measuring,
    stop_time_measuring,
)
from capymoa.datasets import download_unpacked
from capymoa.datasets._source_list import SOURCE_LIST
from capymoa.datasets._utils import infer_unpacked_path
from capymoa.classifier import (
    NaiveBayes,
    HoeffdingTree,
    EFDT,
    KNN,
    AdaptiveRandomForestClassifier,
)

# Globals
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# To check if the benchmark is running appropriately until the end, you might want to set this to a lower value.
DEFAULT_MAX_INSTANCES = 100000
DEFAULT_REPETITIONS = 5

SUPPORTED_DATASETS = {
    "rtg_2abrupt": {"dataset_name": "RTG_2abrupt", "class_name": "RTG_2abrupt"},
    "hyper100k": {"dataset_name": "Hyper100k", "class_name": "Hyper100k"},
    "rbfm_100k": {"dataset_name": "RBFm_100k", "class_name": "RBFm_100k"},
    "electricity": {"dataset_name": "Electricity", "class_name": "Electricity"},
    "covtfd": {"dataset_name": "CovtFD", "class_name": "CovtFD"},
    "covtype_norm": {"dataset_name": "CovtypeNorm", "class_name": "CovtypeNorm"},
    "sensor": {"dataset_name": "Sensor", "class_name": "Sensor"},
}

ALGORITHM_ORDER = [
    "NaiveBayes",
    "HT",
    "EFDT",
    "KNN",
    "ARF5",
    "ARF10",
    "ARF30",
    "ARF100",
    "ARF100j4",
]


def format_instance_count(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def build_default_output_prefix(dataset_name: str, max_instances: int) -> str:
    return experiment_id(dataset_name, max_instances)


def build_default_plot_title(dataset_name: str, max_instances: int) -> str:
    return f"{dataset_name} {format_instance_count(max_instances)}"


def ensure_dataset_assets(dataset_key: str):
    """Ensure the selected CapyMOA dataset and its CSV companion exist under repo-local data/."""
    dataset_config = SUPPORTED_DATASETS[dataset_key]
    dataset_name = dataset_config["dataset_name"]
    DATA_DIR.mkdir(exist_ok=True)

    dataset_class = getattr(capymoa_datasets, dataset_config["class_name"])
    stream = dataset_class(directory=DATA_DIR)
    csv_url = SOURCE_LIST[dataset_name].csv
    if csv_url is None:
        raise RuntimeError(f"{dataset_name} does not define a CSV download source.")
    csv_path = infer_unpacked_path(csv_url, DATA_DIR)

    if not csv_path.exists():
        print(f"Downloading {dataset_name} CSV to {csv_path}")
        download_unpacked(csv_url, DATA_DIR)

    return stream, csv_path, dataset_name


def write_machine_info(output_file):
    """Write lightweight host details that help interpret benchmark results."""
    try:
        machine_info = {
            "platform": platform.system() or "unknown",
            "platform_release": platform.release() or "unknown",
            "platform_version": platform.version() or "unknown",
            "architecture": platform.machine() or "unknown",
            "processor": platform.processor() or "unknown",
            "python_version": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "machine_info_status": "ok",
        }
    except Exception as exc:
        machine_info = {
            "machine_info_status": "unavailable",
            "message": (
                "Machine details could not be generated for this run. "
                f"Reason: {type(exc).__name__}: {exc}"
            ),
            "python_version": sys.version.split()[0],
        }

    pd.Series(machine_info).to_json(output_file, indent=2)
    return machine_info


# Save combined results to output file
def checkpoint_results(results, new_result, output_file):
    results = pd.concat([results, new_result], ignore_index=True)
    results.to_csv(output_file, index=False)
    return results


def append_raw_result(raw_result, output_file):
    raw_df = pd.DataFrame([raw_result])
    header = not output_file.exists()
    raw_df.to_csv(output_file, mode="a", header=header, index=False)


class PulseRecorder:
    def __init__(
        self,
        *,
        output_file: Path,
        platform_name: str,
        algorithm_name: str,
        repetition: int,
        total_instances: int,
        pulse_percent: float,
    ):
        self.output_file = output_file
        self.platform_name = platform_name
        self.algorithm_name = algorithm_name
        self.repetition = repetition
        self.total_instances = max(int(total_instances), 1)
        self.pulse_percent = pulse_percent
        if pulse_percent <= 0:
            self.pulse_interval = self.total_instances
        else:
            self.pulse_interval = max(
                1, int(np.ceil(self.total_instances * (pulse_percent / 100.0)))
            )
        self.last_timestamp = None
        self.next_pulse_at = self.pulse_interval
        self.record(0)

    def record(self, processed_instances: int):
        now = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        delta = 0.0 if self.last_timestamp is None else now - self.last_timestamp
        pulse_row = {
            "platform": self.platform_name,
            "algorithm": self.algorithm_name,
            "repetition": self.repetition,
            "processed_instances": int(processed_instances),
            "percent_processed": float(processed_instances) / self.total_instances,
            "timestamp": timestamp,
            "delta_s": delta,
        }
        pulse_df = pd.DataFrame([pulse_row])
        header = not self.output_file.exists()
        pulse_df.to_csv(self.output_file, mode="a", header=header, index=False)
        self.last_timestamp = now

    def maybe_record(self, processed_instances: int):
        processed_instances = int(processed_instances)
        while processed_instances >= self.next_pulse_at and self.next_pulse_at <= self.total_instances:
            self.record(self.next_pulse_at)
            self.next_pulse_at += self.pulse_interval

    def finish(self, processed_instances: int):
        processed_instances = min(int(processed_instances), self.total_instances)
        if processed_instances > 0 and (
            self.last_timestamp is None or processed_instances != self.total_instances
        ):
            self.record(processed_instances)
        elif processed_instances == self.total_instances and (
            self.last_timestamp is None or self.next_pulse_at <= self.total_instances
        ):
            self.record(processed_instances)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CapyMOA and River on a supported CapyMOA dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_DATASETS.keys()),
        default="rtg_2abrupt",
        help="CapyMOA dataset to benchmark.",
    )
    parser.add_argument(
        "--library",
        choices=["both", "capymoa", "river"],
        default="both",
        help="Choose whether to run CapyMOA, River, or both benchmarks.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Base name for all output artifacts written under benchmarks/results/. "
            "When omitted, a dataset-aware timestamped prefix is used."
        ),
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=DEFAULT_MAX_INSTANCES,
        help="Maximum number of stream instances to evaluate.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Number of repetitions for each learner.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Regenerate plots from an existing results CSV identified by "
            "--output-prefix, without rerunning the benchmark."
        ),
    )
    parser.add_argument(
        "--dark-theme",
        action="store_true",
        help="Render plots with a dark background theme.",
    )
    parser.add_argument(
        "--plot-title",
        default=None,
        help=(
            "Optional title prefix for plots. When omitted, each plot uses the "
            "dataset name and instance count."
        ),
    )
    parser.add_argument(
        "--skip-threaded-arf",
        action="store_true",
        help="Exclude the CapyMOA-only threaded ARF100j4 learner from the benchmark.",
    )
    parser.add_argument(
        "--pulse-percent",
        type=float,
        default=5.0,
        help="Write a pulse entry every N percent of the dataset. Defaults to 5.",
    )
    parser.add_argument(
        "--algorithms",
        default=None,
        help=(
            "Comma-separated learner names to run, for example "
            "'HT,EFDT,ARF30'. Defaults to all supported learners."
        ),
    )
    return parser.parse_args()


def resolve_output_paths(output_prefix, experiment_dir: Path, experiment_name: str):
    return {
        "results_csv": experiment_dir / f"{output_prefix}.csv",
        "raw_results_csv": experiment_dir / f"{output_prefix}_raw.csv",
        "pulse_csv": experiment_dir / f"pulse_{output_prefix}.csv",
        "pulse_dir": experiment_dir / "pulse",
        "machine_info_json": experiment_dir / f"{output_prefix}_machine.json",
        "experiment_md": experiment_dir / f"{experiment_name}_experiment.md",
        "configurations_md": experiment_dir / f"{experiment_name}_configurations.md",
        "plot_prefix": experiment_dir / f"{output_prefix}_performance_plot",
    }


def dataset_docs_url(dataset_class_name: str) -> str:
    return (
        "https://capymoa.org/api/modules/"
        f"capymoa.datasets.{dataset_class_name}.html"
        f"#capymoa.datasets.{dataset_class_name}"
    )


def classifier_docs_url(class_name: str) -> str:
    return (
        "https://capymoa.org/api/modules/"
        f"capymoa.classifier.{class_name}.html"
        f"#capymoa.classifier.{class_name}"
    )


def benchmark_cli_string() -> str:
    return "python " + " ".join(sys.argv)


def experiment_id(dataset_name: str, max_instances: int) -> str:
    return f"{dataset_name}_{max_instances}"


def experiment_algorithms(
    libraries: list[str],
    include_threaded_arf: bool = True,
    selected_algorithms=None,
) -> list[str]:
    algorithms = [
        f"[NaiveBayes]({classifier_docs_url('NaiveBayes')})",
        f"[HT]({classifier_docs_url('HoeffdingTree')})",
        f"[EFDT]({classifier_docs_url('EFDT')})",
        f"[KNN]({classifier_docs_url('KNN')})",
        f"[ARF5]({classifier_docs_url('AdaptiveRandomForestClassifier')})",
        f"[ARF10]({classifier_docs_url('AdaptiveRandomForestClassifier')})",
        f"[ARF30]({classifier_docs_url('AdaptiveRandomForestClassifier')})",
        f"[ARF100]({classifier_docs_url('AdaptiveRandomForestClassifier')})",
    ]
    if "capymoa" in libraries and include_threaded_arf:
        algorithms.append(
            f"[ARF100j4]({classifier_docs_url('AdaptiveRandomForestClassifier')}) [capymoa-only]"
        )
    if selected_algorithms is None:
        return algorithms
    return [
        algorithm
        for algorithm, learner_name in zip(algorithms, ALGORITHM_ORDER)
        if learner_name in selected_algorithms
    ]


def parse_selected_algorithms(raw_value, *, include_threaded_arf: bool):
    allowed = set(ALGORITHM_ORDER if include_threaded_arf else ALGORITHM_ORDER[:-1])
    if raw_value is None:
        return [name for name in ALGORITHM_ORDER if name in allowed]
    selected = [part.strip() for part in raw_value.split(",") if part.strip()]
    invalid = [name for name in selected if name not in allowed]
    if invalid:
        raise ValueError(
            "Unknown or unavailable algorithms requested: "
            f"{', '.join(invalid)}. Allowed values: {', '.join(sorted(allowed))}"
        )
    return selected


def write_configurations_summary(
    output_file: Path,
    *,
    dataset_name: str,
    max_instances: int,
    include_threaded_arf: bool = True,
    selected_algorithms=None,
):
    lines = [
        f"Experiment ID: `{experiment_id(dataset_name, max_instances)}`",
        "",
        "Configurations:",
        "",
    ]
    selected_set = set(selected_algorithms or ALGORITHM_ORDER)
    config_lines = {
        "NaiveBayes": "- NaiveBayes (`default`)",
        "HT": "- HT (`default`)",
        "EFDT": "- EFDT (`default`)",
        "KNN": "- KNN (`{'window_size': 1000, 'k': 3}`)",
        "ARF5": "- ARF5 (`{'ensemble_size': 5, 'max_features': 0.6}` for CapyMOA; `{'n_models': 5, 'max_features': 0.6}` for River)",
        "ARF10": "- ARF10 (`{'ensemble_size': 10, 'max_features': 0.6}` for CapyMOA; `{'n_models': 10, 'max_features': 0.6}` for River)",
        "ARF30": "- ARF30 (`{'ensemble_size': 30, 'max_features': 0.6}` for CapyMOA; `{'n_models': 30, 'max_features': 0.6}` for River)",
        "ARF100": "- ARF100 (`{'ensemble_size': 100, 'max_features': 0.6}` for CapyMOA; `{'n_models': 100, 'max_features': 0.6}` for River)",
        "ARF100j4": "- ARF100j4 (`{'ensemble_size': 100, 'max_features': 0.6, 'number_of_jobs': 4}`) [capymoa-only]",
    }
    for learner_name in ALGORITHM_ORDER:
        if learner_name == "ARF100j4" and not include_threaded_arf:
            continue
        if learner_name in selected_set:
            lines.append(config_lines[learner_name])
    if include_threaded_arf:
        if "ARF100j4" in selected_set:
            lines.append(
                "- ARF100j4 is a CapyMOA-only threaded configuration and has no River counterpart."
            )
    lines.append("")
    output_file.write_text("\n".join(lines), encoding="utf-8")


def write_experiment_summary(
    output_file: Path,
    *,
    dataset_name: str,
    dataset_class_name: str,
    task: str,
    max_instances: int,
    repetitions: int,
    stream,
    libraries: list[str],
    experiment_date: str,
    elapsed_seconds: float,
    machine_info: dict,
    include_threaded_arf: bool = True,
    selected_algorithms=None,
):
    schema = stream.get_schema()
    total_instances = len(stream)
    feature_count = schema.get_num_attributes()
    class_count = schema.get_num_classes()
    algorithms = experiment_algorithms(
        libraries, include_threaded_arf, selected_algorithms
    )
    machine_info_json = pd.Series(machine_info).to_json(indent=2)

    lines = [
        f"Experiment ID: `{experiment_id(dataset_name, max_instances)}`",
        "",
        f"Experiment date: `{experiment_date}`",
        "",
        f"Experiment duration: `{elapsed_seconds:.2f}` seconds",
        "",
        f"Task: `{task}`",
        "",
        f"Benchmarking CLI: `{benchmark_cli_string()}`",
        "",
        "Dataset:",
        f"- Name: `{dataset_name}`",
        f"- Instances used in benchmark: `{max_instances}`",
        f"- Total dataset instances: `{total_instances}`",
        f"- Features: `{feature_count}`",
        f"- Classes: `{class_count}`",
        f"- Docs: [{dataset_class_name}]({dataset_docs_url(dataset_class_name)})",
        "",
        "Algorithms:",
    ]

    lines.extend([f"- {algorithm}" for algorithm in algorithms])

    lines.extend(
        [
            "",
            "Libraries:",
            *[f"- `{library}`" for library in libraries],
            "",
            "Benchmark Settings:",
            f"- Repetitions: `{repetitions}`",
            f"- Max instances: `{max_instances}`",
            "",
            "Machine details:",
            "```json",
            machine_info_json,
            "```",
            "",
        ]
    )

    output_file.write_text("\n".join(lines), encoding="utf-8")


def capymoa_experiment(
    dataset_name,
    learner_name,
    stream,
    learner,
    hyperparameters={},
    repetitions=1,
    max_instances=DEFAULT_MAX_INSTANCES,
    raw_results_output_csv=None,
    pulse_output_csv=None,
    pulse_percent=5.0,
):
    date_time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
    print(f"[{date_time_stamp}][capymoa] Executing {learner_name} on {dataset_name}")

    results = []
    raw_results = []  # Store raw results for each repetition

    total_instances = min(len(stream), max_instances)

    repetition = 1
    for _ in range(repetitions):
        print(f"[{date_time_stamp}][capymoa]\trepetition {repetition}")
        learner_instance = learner(**hyperparameters, schema=stream.get_schema())
        pulse_recorder = None
        if pulse_output_csv is not None:
            pulse_recorder = PulseRecorder(
                output_file=pulse_output_csv,
                platform_name="capymoa",
                algorithm_name=learner_name,
                repetition=repetition,
                total_instances=total_instances,
                pulse_percent=pulse_percent,
            )
        stream.restart()
        processed_instances = 0
        total_correct = 0.0
        total_wallclock = 0.0
        total_cpu_time = 0.0
        chunk_results = []
        pulse_interval = (
            total_instances
            if pulse_percent <= 0
            else max(1, int(np.ceil(total_instances * (pulse_percent / 100.0))))
        )

        while processed_instances < total_instances:
            remaining = total_instances - processed_instances
            chunk_size = min(pulse_interval, remaining)
            result = prequential_evaluation(
                stream=stream,
                learner=learner_instance,
                max_instances=chunk_size,
                window_size=chunk_size,
                restart_stream=False,
            )
            chunk_results.append(result)
            processed_instances += chunk_size
            total_correct += (result["cumulative"].accuracy() / 100.0) * chunk_size
            total_wallclock += result["wallclock"]
            total_cpu_time += result["cpu_time"]
            if pulse_recorder is not None:
                pulse_recorder.maybe_record(processed_instances)

        if pulse_recorder is not None:
            pulse_recorder.finish(processed_instances)

        aggregated_accuracy = (total_correct / total_instances) * 100.0
        repetition_result = {
            "library": "capymoa",
            "repetition": repetition,
            "dataset": dataset_name,
            "learner": learner_name,
            "hyperparameters": str(hyperparameters),
            "accuracy": aggregated_accuracy,
            "wallclock": total_wallclock,
            "cpu_time": total_cpu_time,
        }
        raw_results.append(repetition_result)
        if raw_results_output_csv is not None:
            append_raw_result(repetition_result, raw_results_output_csv)
        repetition += 1

    # Calculate average and std for accuracy, wallclock, and cpu_time
    avg_accuracy = sum(result["accuracy"] for result in raw_results) / repetitions
    std_accuracy = pd.Series([result["accuracy"] for result in raw_results]).std()
    avg_wallclock = sum(result["wallclock"] for result in raw_results) / repetitions
    std_wallclock = pd.Series([result["wallclock"] for result in raw_results]).std()
    avg_cpu_time = sum(result["cpu_time"] for result in raw_results) / repetitions
    std_cpu_time = pd.Series([result["cpu_time"] for result in raw_results]).std()

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


def test_then_train_river(
    stream_data,
    model,
    max_instances=DEFAULT_MAX_INSTANCES,
    pulse_recorder=None,
):
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    instancesProcessed = 0
    accuracy = metrics.Accuracy()

    X, Y = stream_data[:, :-1], stream_data[:, -1]

    data = []
    performance_names = ["Classified instances", "accuracy"]

    ds = stream_river.iter_array(X, Y)

    for x, y in ds:
        if instancesProcessed >= max_instances:
            break
        yp = model.predict_one(x)
        accuracy.update(y, yp)
        model.learn_one(x, y)
        instancesProcessed += 1
        if pulse_recorder is not None:
            pulse_recorder.maybe_record(instancesProcessed)

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )
    if pulse_recorder is not None:
        pulse_recorder.finish(instancesProcessed)

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
    max_instances=DEFAULT_MAX_INSTANCES,
    raw_results_output_csv=None,
    pulse_output_csv=None,
    pulse_percent=5.0,
):
    date_time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
    print(f"[{date_time_stamp}][river] Executing {learner_name} on {dataset_name}")

    raw_results = []  # Store raw results for each repetition

    stream_data = pd.read_csv(stream_path_csv).to_numpy()
    total_instances = min(len(stream_data), max_instances)

    repetition = 1
    for _ in range(repetitions):
        print(f"[{date_time_stamp}][river]\trepetition {repetition}")
        pulse_recorder = None
        if pulse_output_csv is not None:
            pulse_recorder = PulseRecorder(
                output_file=pulse_output_csv,
                platform_name="river",
                algorithm_name=learner_name,
                repetition=repetition,
                total_instances=total_instances,
                pulse_percent=pulse_percent,
            )
        model_instance = learner(**hyperparameters)
        acc, wallclock, cpu_time, df_raw = test_then_train_river(
            stream_data=stream_data,
            model=model_instance,
            max_instances=max_instances,
            pulse_recorder=pulse_recorder,
        )

        # Append raw result to list
        raw_result = {
            "library": "river",
            "repetition": repetition,
            "dataset": dataset_name,
            "learner": learner_name,
            "hyperparameters": str(hyperparameters).replace("\n", ""),
            "accuracy": acc,
            "wallclock": wallclock,
            "cpu_time": cpu_time,
        }
        raw_results.append(raw_result)
        if raw_results_output_csv is not None:
            append_raw_result(raw_result, raw_results_output_csv)
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
    pulse_output_csv,
    max_instances,
    repetitions,
    include_threaded_arf=True,
    pulse_percent=5.0,
    selected_algorithms=None,
):
    selected_algorithms = set(selected_algorithms or ALGORITHM_ORDER)
    # Run experiment 1
    if "NaiveBayes" in selected_algorithms:
        result_capyNB, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="NaiveBayes",
        stream=data,
        learner=NaiveBayes,
        hyperparameters={},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_capyNB, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_capymoa], ignore_index=True
        )

    # Run experiment 2
    if "HT" in selected_algorithms:
        result_capyHT, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="HT",
        stream=data,
        learner=HoeffdingTree,
        hyperparameters={},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_capyHT, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_capymoa], ignore_index=True
        )

    # Run experiment 3
    if "EFDT" in selected_algorithms:
        result_capyEFDT, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="EFDT",
        stream=data,
        learner=EFDT,
        hyperparameters={},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_capyEFDT, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_capymoa], ignore_index=True
        )

    # Run experiment 4
    if "KNN" in selected_algorithms:
        result_capyKNN, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="KNN",
        stream=data,
        learner=KNN,
        hyperparameters={"window_size": 1000, "k": 3},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_capyKNN, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_capymoa], ignore_index=True
        )

    # Run experiment 5
    if "ARF5" in selected_algorithms:
        result_capyARF5, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="ARF5",
        stream=data,
        learner=AdaptiveRandomForestClassifier,
        hyperparameters={"ensemble_size": 5, "max_features": 0.6},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_capyARF5, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_capymoa], ignore_index=True
        )

    # Run experiment 6
    if "ARF10" in selected_algorithms:
        result_capyARF10, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="ARF10",
        stream=data,
        learner=AdaptiveRandomForestClassifier,
        hyperparameters={"ensemble_size": 10, "max_features": 0.6},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_capyARF10, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_capymoa], ignore_index=True
        )

    # Run experiment 7
    if "ARF30" in selected_algorithms:
        result_capyARF30, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="ARF30",
        stream=data,
        learner=AdaptiveRandomForestClassifier,
        hyperparameters={"ensemble_size": 30, "max_features": 0.6},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_capyARF30, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_capymoa], ignore_index=True
        )

    # Run experiment 8
    if "ARF100" in selected_algorithms:
        result_capyARF100, raw_capymoa = capymoa_experiment(
        dataset_name=dataset_names,
        learner_name="ARF100",
        stream=data,
        learner=AdaptiveRandomForestClassifier,
        hyperparameters={"ensemble_size": 100, "max_features": 0.6},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_capyARF100, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_capymoa], ignore_index=True
        )

    if include_threaded_arf and "ARF100j4" in selected_algorithms:
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
            repetitions=repetitions,
            max_instances=max_instances,
            raw_results_output_csv=raw_results_output_csv,
            pulse_output_csv=pulse_output_csv,
            pulse_percent=pulse_percent,
        )

        intermediary_results = checkpoint_results(
            intermediary_results, result_capyARF100j4, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_capymoa], ignore_index=True
        )

    return intermediary_results, raw_intermediary_results


def benchmark_classifiers_river(
    intermediary_results,
    raw_intermediary_results,
    stream_path_csv,
    dataset_names,
    results_output_csv,
    raw_results_output_csv,
    pulse_output_csv,
    max_instances,
    repetitions,
    pulse_percent,
    selected_algorithms=None,
):
    selected_algorithms = set(selected_algorithms or ALGORITHM_ORDER)
    # Run experiment 1
    if "NaiveBayes" in selected_algorithms:
        result_riverNB, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="NaiveBayes",
        stream_path_csv=stream_path_csv,
        learner=GaussianNB,
        hyperparameters={},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_riverNB, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_river], ignore_index=True
        )

    # Run experiment 2
    if "HT" in selected_algorithms:
        result_riverHT, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="HT",
        stream_path_csv=stream_path_csv,
        learner=HoeffdingTreeClassifier,
        hyperparameters={},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_riverHT, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_river], ignore_index=True
        )

    # Run experiment 3
    if "EFDT" in selected_algorithms:
        result_riverEFDT, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="EFDT",
        stream_path_csv=stream_path_csv,
        learner=ExtremelyFastDecisionTreeClassifier,
        hyperparameters={},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_riverEFDT, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_river], ignore_index=True
        )

    # Run experiment 4
    if "KNN" in selected_algorithms:
        result_riverKNN, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="KNN",
        stream_path_csv=stream_path_csv,
        learner=KNNClassifier,
        hyperparameters={"engine": LazySearch(window_size=1000), "n_neighbors": 3},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_riverKNN, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_river], ignore_index=True
        )

    # Run experiment 5
    if "ARF5" in selected_algorithms:
        result_riverARF5, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="ARF5",
        stream_path_csv=stream_path_csv,
        learner=ARFClassifier,
        hyperparameters={"n_models": 5, "max_features": 0.60},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_riverARF5, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_river], ignore_index=True
        )

    # Run experiment 6
    if "ARF10" in selected_algorithms:
        result_riverARF10, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="ARF10",
        stream_path_csv=stream_path_csv,
        learner=ARFClassifier,
        hyperparameters={"n_models": 10, "max_features": 0.60},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_riverARF10, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_river], ignore_index=True
        )

    # Run experiment 7
    if "ARF30" in selected_algorithms:
        result_riverARF30, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="ARF30",
        stream_path_csv=stream_path_csv,
        learner=ARFClassifier,
        hyperparameters={"n_models": 30, "max_features": 0.60},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_riverARF30, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_river], ignore_index=True
        )

    # Run experiment 8
    if "ARF100" in selected_algorithms:
        result_riverARF100, raw_river = river_experiment(
        dataset_name=dataset_names,
        learner_name="ARF100",
        stream_path_csv=stream_path_csv,
        learner=ARFClassifier,
        hyperparameters={"n_models": 100, "max_features": 0.60},
        repetitions=repetitions,
        max_instances=max_instances,
        raw_results_output_csv=raw_results_output_csv,
        pulse_output_csv=pulse_output_csv,
        pulse_percent=pulse_percent,
    )

        intermediary_results = checkpoint_results(
            intermediary_results, result_riverARF100, results_output_csv
        )
        raw_intermediary_results = pd.concat(
            [raw_intermediary_results, raw_river], ignore_index=True
        )

    return intermediary_results, raw_intermediary_results


def plot_performance(
    df,
    plot_prefix,
    dark_theme=False,
    plot_title=None,
    dataset_name=None,
    max_instances=None,
):
    # Step 1: Filter and reorder data
    ordered_algorithms = [name for name in ALGORITHM_ORDER if name in set(df["learner"])]
    df = df.copy()
    df["learner"] = pd.Categorical(df["learner"], ordered_algorithms, ordered=True)
    df = df.sort_values("learner")

    libraries = [library for library in ["capymoa", "river"] if library in set(df["library"])]
    if len(libraries) == 0:
        print("No benchmark results available to plot.")
        return

    plot_df = df[df["learner"].notna()].set_index(["learner", "library"])

    # Step 2: Plot each measure
    measures = ["accuracy", "wallclock", "cpu_time"]
    for measure in measures:
        fig, ax = plt.subplots(figsize=(10, 6))
        if dark_theme:
            fig.patch.set_facecolor("#101418")
            ax.set_facecolor("#101418")
            text_color = "#f3f5f7"
            grid_color = "#3a444d"
            error_bar_color = "#f3f5f7"
        else:
            text_color = "black"
            grid_color = "#d0d7de"
            error_bar_color = "black"

        metric_title = measure.replace("_", " ").title()
        title_base = plot_title
        if title_base is None:
            inferred_dataset_name = dataset_name
            if inferred_dataset_name is None and "dataset" in df.columns and len(df) > 0:
                inferred_dataset_name = str(df["dataset"].iloc[0])

            if inferred_dataset_name is not None and max_instances is not None:
                title_base = build_default_plot_title(
                    inferred_dataset_name, max_instances
                )
        if plot_title:
            ax.set_title(f"{plot_title}: {metric_title}", color=text_color)
        elif title_base:
            ax.set_title(f"{title_base} ({metric_title})", color=text_color)
        else:
            ax.set_title(metric_title, color=text_color)
        ax.set_xlabel("Algorithm", color=text_color)
        ax.set_ylabel(measure.capitalize(), color=text_color)
        ax.tick_params(axis="x", colors=text_color, rotation=45)
        ax.tick_params(axis="y", colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.grid(axis="y", color=grid_color, alpha=0.35)
        ax.set_axisbelow(True)

        x_positions = np.arange(len(ordered_algorithms))
        width = 0.8 / max(len(libraries), 1)
        colors = {"capymoa": "#44d17a", "river": "#ff6b6b"} if dark_theme else {"capymoa": "green", "river": "red"}
        paired_learners = {
            learner
            for learner in ordered_algorithms
            if all((learner, library) in plot_df.index for library in libraries)
        }
        capymoa_only_color = "#6aa9ff" if dark_theme else "#4a7dff"

        for idx, library in enumerate(libraries):
            means = []
            stds = []
            bar_colors = []
            for learner in ordered_algorithms:
                if (learner, library) in plot_df.index:
                    row = plot_df.loc[(learner, library)]
                    means.append(row[f"avg_{measure}"])
                    stds.append(row[f"std_{measure}"])
                    if library == "capymoa" and learner not in paired_learners:
                        bar_colors.append(capymoa_only_color)
                    else:
                        bar_colors.append(colors.get(library, "gray"))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                    bar_colors.append(colors.get(library, "gray"))

            means_series = pd.Series(means, index=ordered_algorithms)
            stds_series = pd.Series(stds, index=ordered_algorithms)
            valid_mask = means_series.notna()
            offset = (idx - (len(libraries) - 1) / 2) * width
            positions = x_positions + offset

            ax.bar(
                positions[valid_mask.to_numpy()],
                means_series[valid_mask],
                yerr=stds_series[valid_mask],
                width=width,
                color=pd.Series(bar_colors, index=ordered_algorithms)[valid_mask],
                ecolor=error_bar_color,
                capsize=4,
            )

        ax.set_xticks(x_positions, ordered_algorithms)

        # Step 3: Customize plot
        legend_handles = []
        if "capymoa" in libraries:
            legend_handles.append(Patch(color=colors["capymoa"], label="capymoa"))
        if any(
            learner not in paired_learners
            for learner in ordered_algorithms
            if ("capymoa" in libraries and (learner, "capymoa") in plot_df.index)
        ):
            legend_handles.append(
                Patch(color=capymoa_only_color, label="capymoa-only")
            )
        if "river" in libraries:
            legend_handles.append(Patch(color=colors["river"], label="river"))

        legend = ax.legend(handles=legend_handles)
        if dark_theme:
            legend.get_frame().set_facecolor("#101418")
            legend.get_frame().set_edgecolor("#3a444d")
            for text in legend.get_texts():
                text.set_color(text_color)
        fig.tight_layout()
        fig.savefig(f"{plot_prefix}_{measure}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
        # plt.show()


def sanitize_filename(value: str) -> str:
    allowed = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "plot"


def write_pulse_plots(pulse_csv: Path, pulse_dir: Path, *, dark_theme: bool = False):
    if not pulse_csv.exists():
        return

    pulse_df = pd.read_csv(pulse_csv)
    if pulse_df.empty:
        return

    pulse_dir.mkdir(parents=True, exist_ok=True)

    algorithms = sorted(pulse_df["algorithm"].dropna().unique())
    for algorithm in algorithms:
        algorithm_df = pulse_df[pulse_df["algorithm"] == algorithm].copy()
        if algorithm_df.empty:
            continue

        algorithm_csv = pulse_dir / f"{sanitize_filename(algorithm)}_pulse.csv"
        algorithm_df.to_csv(algorithm_csv, index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        if dark_theme:
            fig.patch.set_facecolor("#101418")
            ax.set_facecolor("#101418")
            text_color = "#f3f5f7"
            grid_color = "#3a444d"
            colors = {"capymoa": "#44d17a", "river": "#ff6b6b"}
        else:
            text_color = "black"
            grid_color = "#d0d7de"
            colors = {"capymoa": "green", "river": "red"}

        ax.set_title(f"{algorithm} Pulse", color=text_color)
        ax.set_xlabel("Processed Instances", color=text_color)
        ax.set_ylabel("Delta (s)", color=text_color)
        ax.tick_params(axis="x", colors=text_color)
        ax.tick_params(axis="y", colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.grid(axis="y", color=grid_color, alpha=0.35)
        ax.set_axisbelow(True)

        for platform_name in ["capymoa", "river"]:
            platform_df = algorithm_df[algorithm_df["platform"] == platform_name].copy()
            if platform_df.empty:
                continue
            grouped = (
                platform_df.groupby("processed_instances", as_index=False)
                .agg(
                    mean_delta_s=("delta_s", "mean"),
                    std_delta_s=("delta_s", "std"),
                    percent_processed=("percent_processed", "mean"),
                )
                .sort_values("processed_instances")
            )
            grouped["std_delta_s"] = grouped["std_delta_s"].fillna(0.0)
            color = colors.get(platform_name, "gray")
            ax.plot(
                grouped["processed_instances"],
                grouped["mean_delta_s"],
                label=platform_name,
                color=color,
            )
            ax.fill_between(
                grouped["processed_instances"],
                grouped["mean_delta_s"] - grouped["std_delta_s"],
                grouped["mean_delta_s"] + grouped["std_delta_s"],
                color=color,
                alpha=0.2,
            )

        legend = ax.legend()
        if legend is not None and dark_theme:
            legend.get_frame().set_facecolor("#101418")
            legend.get_frame().set_edgecolor("#3a444d")
            for text in legend.get_texts():
                text.set_color(text_color)

        fig.tight_layout()
        fig.savefig(
            pulse_dir / f"{sanitize_filename(algorithm)}_pulse.png",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)


if __name__ == "__main__":
    overall_start_time = time.time()
    args = parse_args()
    if args.plot_only and args.output_prefix is None:
        raise ValueError("--plot-only requires an explicit --output-prefix.")
    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dataset_config = SUPPORTED_DATASETS[args.dataset]
    dataset_name = dataset_config["dataset_name"]
    selected_algorithms = parse_selected_algorithms(
        args.algorithms, include_threaded_arf=not args.skip_threaded_arf
    )
    experiment_name = experiment_id(dataset_name, args.max_instances)
    experiment_dir = RESULTS_DIR / experiment_name
    output_prefix = args.output_prefix or build_default_output_prefix(
        dataset_name, args.max_instances
    )
    output_paths = resolve_output_paths(output_prefix, experiment_dir, experiment_name)

    # Initialize an empty DataFrame for combined_results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    combined_results = pd.DataFrame()
    raw_results = pd.DataFrame()

    if args.plot_only:
        if not output_paths["results_csv"].exists():
            raise FileNotFoundError(
                "Could not find existing results CSV for plot regeneration: "
                f"{output_paths['results_csv']}"
            )
        combined_results = pd.read_csv(output_paths["results_csv"])
        plot_performance(
            combined_results,
            output_paths["plot_prefix"],
            dark_theme=args.dark_theme,
            plot_title=args.plot_title,
            dataset_name=None,
            max_instances=args.max_instances,
        )
        print(f"Regenerated plots from {output_paths['results_csv']}")
        sys.exit(0)

    for output_key in ("results_csv", "raw_results_csv", "pulse_csv"):
        output_paths[output_key].unlink(missing_ok=True)

    dataset_stream, dataset_csv_path, dataset_label = ensure_dataset_assets(args.dataset)
    machine_info = write_machine_info(output_paths["machine_info_json"])
    active_libraries = (
        ["capymoa", "river"]
        if args.library == "both"
        else [args.library]
    )

    if args.library in {"both", "capymoa"}:
        combined_results, raw_results = benchmark_classifiers_capymoa(
            intermediary_results=combined_results,
            raw_intermediary_results=raw_results,
            data=dataset_stream,
            dataset_names=dataset_label,
            results_output_csv=output_paths["results_csv"],
            raw_results_output_csv=output_paths["raw_results_csv"],
            pulse_output_csv=output_paths["pulse_csv"],
            max_instances=args.max_instances,
            repetitions=args.repetitions,
            include_threaded_arf=not args.skip_threaded_arf,
            pulse_percent=args.pulse_percent,
            selected_algorithms=selected_algorithms,
        )

    if args.library in {"both", "river"}:
        combined_results, raw_results = benchmark_classifiers_river(
            intermediary_results=combined_results,
            raw_intermediary_results=raw_results,
            stream_path_csv=dataset_csv_path,
            dataset_names=dataset_label,
            results_output_csv=output_paths["results_csv"],
            raw_results_output_csv=output_paths["raw_results_csv"],
            pulse_output_csv=output_paths["pulse_csv"],
            max_instances=args.max_instances,
            repetitions=args.repetitions,
            pulse_percent=args.pulse_percent,
            selected_algorithms=selected_algorithms,
        )

    plot_performance(
        combined_results,
        output_paths["plot_prefix"],
        dark_theme=args.dark_theme,
        plot_title=args.plot_title,
        dataset_name=dataset_label,
        max_instances=args.max_instances,
    )
    write_pulse_plots(
        output_paths["pulse_csv"],
        output_paths["pulse_dir"],
        dark_theme=args.dark_theme,
    )

    elapsed_seconds = time.time() - overall_start_time
    write_experiment_summary(
        output_paths["experiment_md"],
        dataset_name=dataset_label,
        dataset_class_name=dataset_config["class_name"],
        task="Classification",
        max_instances=args.max_instances,
        repetitions=args.repetitions,
        stream=dataset_stream,
        libraries=active_libraries,
        experiment_date=run_started_at,
        elapsed_seconds=elapsed_seconds,
        machine_info=machine_info,
        include_threaded_arf=not args.skip_threaded_arf,
        selected_algorithms=selected_algorithms,
    )
    write_configurations_summary(
        output_paths["configurations_md"],
        dataset_name=dataset_label,
        max_instances=args.max_instances,
        include_threaded_arf=not args.skip_threaded_arf,
        selected_algorithms=selected_algorithms,
    )

    print(combined_results)
