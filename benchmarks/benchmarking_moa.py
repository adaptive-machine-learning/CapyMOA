import argparse
from datetime import datetime
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time

import matplotlib
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import capymoa.datasets as capymoa_datasets


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MOA_JAR = REPO_ROOT / "src" / "capymoa" / "jar" / "moa.jar"

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


MOA_LEARNERS = [
    {
        "name": "NaiveBayes",
        "classifier_cli": "bayes.NaiveBayes",
        "hyperparameters": {},
        "config_summary": "default",
    },
    {
        "name": "HT",
        "classifier_cli": "trees.HoeffdingTree",
        "hyperparameters": {},
        "config_summary": "default",
    },
    {
        "name": "EFDT",
        "classifier_cli": "trees.EFDT",
        "hyperparameters": {},
        "config_summary": "default",
    },
    {
        "name": "KNN",
        "classifier_cli": "lazy.kNN -k 3 -w 1000",
        "hyperparameters": {"window_size": 1000, "k": 3},
        "config_summary": "{'window_size': 1000, 'k': 3}",
    },
    {
        "name": "ARF5",
        "classifier_cli": "meta.AdaptiveRandomForest -l (ARFHoeffdingTree -e 2000000 -g 50 -c 0.01) -s 5 -o (Percentage (M * (m / 100))) -m 60 -a 6.0 -x (ADWINChangeDetector -a 1.0E-3) -p (ADWINChangeDetector -a 1.0E-2) -j 1",
        "hyperparameters": {"ensemble_size": 5, "max_features": 0.6},
        "config_summary": "{'ensemble_size': 5, 'max_features': 0.6}",
    },
    {
        "name": "ARF10",
        "classifier_cli": "meta.AdaptiveRandomForest -l (ARFHoeffdingTree -e 2000000 -g 50 -c 0.01) -s 10 -o (Percentage (M * (m / 100))) -m 60 -a 6.0 -x (ADWINChangeDetector -a 1.0E-3) -p (ADWINChangeDetector -a 1.0E-2) -j 1",
        "hyperparameters": {"ensemble_size": 10, "max_features": 0.6},
        "config_summary": "{'ensemble_size': 10, 'max_features': 0.6}",
    },
    {
        "name": "ARF30",
        "classifier_cli": "meta.AdaptiveRandomForest -l (ARFHoeffdingTree -e 2000000 -g 50 -c 0.01) -s 30 -o (Percentage (M * (m / 100))) -m 60 -a 6.0 -x (ADWINChangeDetector -a 1.0E-3) -p (ADWINChangeDetector -a 1.0E-2) -j 1",
        "hyperparameters": {"ensemble_size": 30, "max_features": 0.6},
        "config_summary": "{'ensemble_size': 30, 'max_features': 0.6}",
    },
    {
        "name": "ARF100",
        "classifier_cli": "meta.AdaptiveRandomForest -l (ARFHoeffdingTree -e 2000000 -g 50 -c 0.01) -s 100 -o (Percentage (M * (m / 100))) -m 60 -a 6.0 -x (ADWINChangeDetector -a 1.0E-3) -p (ADWINChangeDetector -a 1.0E-2) -j 1",
        "hyperparameters": {"ensemble_size": 100, "max_features": 0.6},
        "config_summary": "{'ensemble_size': 100, 'max_features': 0.6}",
    },
    {
        "name": "ARF100j4",
        "classifier_cli": "meta.AdaptiveRandomForest -l (ARFHoeffdingTree -e 2000000 -g 50 -c 0.01) -s 100 -o (Percentage (M * (m / 100))) -m 60 -a 6.0 -x (ADWINChangeDetector -a 1.0E-3) -p (ADWINChangeDetector -a 1.0E-2) -j 4",
        "hyperparameters": {
            "ensemble_size": 100,
            "max_features": 0.6,
            "number_of_jobs": 4,
        },
        "config_summary": "{'ensemble_size': 100, 'max_features': 0.6, 'number_of_jobs': 4}",
    },
]


def format_instance_count(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def experiment_id(dataset_name: str, max_instances: int) -> str:
    return f"{dataset_name}_{max_instances}"


def build_default_output_prefix(dataset_name: str, max_instances: int) -> str:
    return f"moa_{experiment_id(dataset_name, max_instances)}"


def build_default_plot_title(dataset_name: str, max_instances: int) -> str:
    return f"{dataset_name} {format_instance_count(max_instances)}"


def dataset_docs_url(dataset_class_name: str) -> str:
    return (
        "https://capymoa.org/api/modules/"
        f"capymoa.datasets.{dataset_class_name}.html"
        f"#capymoa.datasets.{dataset_class_name}"
    )


def benchmark_cli_string() -> str:
    return "python " + " ".join(sys.argv)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark MOA CLI on a supported CapyMOA dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_DATASETS.keys()),
        default="rtg_2abrupt",
        help="CapyMOA dataset to benchmark through MOA CLI.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Base name for all output artifacts written under benchmarks/results/.",
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
        help="Regenerate plots from an existing results CSV identified by --output-prefix.",
    )
    parser.add_argument(
        "--dark-theme",
        action="store_true",
        help="Render plots with a dark background theme.",
    )
    parser.add_argument(
        "--plot-title",
        default=None,
        help="Optional title prefix for plots.",
    )
    parser.add_argument(
        "--java-bin",
        default="java",
        help="Java executable used to run MOA.",
    )
    parser.add_argument(
        "--java-args",
        default="",
        help="Extra JVM arguments, for example '-Xmx8g -Xss50M'.",
    )
    parser.add_argument(
        "--skip-threaded-arf",
        action="store_true",
        help="Exclude the threaded ARF100j4 learner from the MOA benchmark.",
    )
    return parser.parse_args()


def resolve_output_paths(output_prefix: str, experiment_dir: Path):
    return {
        "results_csv": experiment_dir / f"{output_prefix}.csv",
        "raw_results_csv": experiment_dir / f"{output_prefix}_raw.csv",
        "machine_info_json": experiment_dir / f"{output_prefix}_machine.json",
        "experiment_md": experiment_dir / f"{output_prefix}_experiment.md",
        "configurations_md": experiment_dir / f"{output_prefix}_configurations.md",
        "plot_prefix": experiment_dir / f"{output_prefix}_performance_plot",
    }


def ensure_dataset_assets(dataset_key: str):
    dataset_config = SUPPORTED_DATASETS[dataset_key]
    dataset_class = getattr(capymoa_datasets, dataset_config["class_name"])
    stream = dataset_class(directory=DATA_DIR)
    return stream, Path(stream.path), dataset_config["dataset_name"]


def write_machine_info(output_file: Path):
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


def checkpoint_results(results: pd.DataFrame, new_result: pd.DataFrame, output_file: Path):
    results = pd.concat([results, new_result], ignore_index=True)
    results.to_csv(output_file, index=False)
    return results


def selected_learners(include_threaded_arf: bool = True):
    if include_threaded_arf:
        return MOA_LEARNERS
    return [learner for learner in MOA_LEARNERS if learner["name"] != "ARF100j4"]


def learner_names(include_threaded_arf: bool = True):
    return [learner["name"] for learner in selected_learners(include_threaded_arf)]


def write_configurations_summary(
    output_file: Path,
    *,
    dataset_name: str,
    max_instances: int,
    include_threaded_arf: bool = True,
):
    lines = [
        f"Experiment ID: `{experiment_id(dataset_name, max_instances)}`",
        "",
        "Configurations:",
        "",
    ]
    for learner in selected_learners(include_threaded_arf):
        lines.append(f"- {learner['name']} (`{learner['config_summary']}`)")
    lines.append("")
    output_file.write_text("\n".join(lines), encoding="utf-8")


def write_experiment_summary(
    output_file: Path,
    *,
    dataset_name: str,
    dataset_class_name: str,
    max_instances: int,
    repetitions: int,
    stream,
    experiment_date: str,
    elapsed_seconds: float,
    machine_info: dict,
    include_threaded_arf: bool = True,
):
    schema = stream.get_schema()
    total_instances = len(stream)
    feature_count = schema.get_num_attributes()
    class_count = schema.get_num_classes()
    machine_info_json = pd.Series(machine_info).to_json(indent=2)

    lines = [
        f"Experiment ID: `{experiment_id(dataset_name, max_instances)}`",
        "",
        f"Experiment date: `{experiment_date}`",
        "",
        f"Experiment duration: `{elapsed_seconds:.2f}` seconds",
        "",
        "Task: `Classification`",
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
    lines.extend([f"- `{name}`" for name in learner_names(include_threaded_arf)])
    lines.extend(
        [
            "",
            "Libraries:",
            "- `moa`",
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


def build_task_string(*, learner_cli: str, arff_path: Path, max_instances: int, dump_file: Path):
    stream_cli = f'ArffFileStream -f "{arff_path}" -c -1'
    evaluator_cli = "BasicClassificationPerformanceEvaluator"
    sample_frequency = max_instances
    return (
        "EvaluatePrequential "
        f"-l ({learner_cli}) "
        f"-e ({evaluator_cli}) "
        f"-i {max_instances} "
        f"-f {sample_frequency} "
        f'-d "{dump_file}" '
        f"-s ({stream_cli})"
    )


def extract_final_metrics(dump_file: Path):
    dump_df = pd.read_csv(dump_file)
    if dump_df.empty:
        raise RuntimeError(f"MOA dump file was empty: {dump_file}")
    final_row = dump_df.iloc[-1]

    accuracy_column = None
    cpu_time_column = None
    for column in dump_df.columns:
        normalized = column.strip().lower()
        if normalized == "classifications correct (percent)":
            accuracy_column = column
        if normalized.startswith("evaluation time"):
            cpu_time_column = column

    if accuracy_column is None:
        raise RuntimeError(
            f"Could not find MOA accuracy column in dump file: {list(dump_df.columns)}"
        )
    if cpu_time_column is None:
        raise RuntimeError(
            f"Could not find MOA evaluation-time column in dump file: {list(dump_df.columns)}"
        )

    return float(final_row[accuracy_column]), float(final_row[cpu_time_column])


def run_moa_task(task_string: str, *, java_bin: str, java_args: str):
    cmd = [java_bin]
    if java_args.strip():
        cmd.extend(shlex.split(java_args))
    cmd.extend(["-cp", str(MOA_JAR), "moa.DoTask", task_string])

    start_wallclock = time.time()
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_wallclock = time.time() - start_wallclock

    if completed.returncode != 0:
        raise RuntimeError(
            "MOA CLI task failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return elapsed_wallclock, completed.stdout, completed.stderr


def moa_experiment(
    dataset_name: str,
    learner_spec: dict,
    arff_path: Path,
    *,
    repetitions: int,
    max_instances: int,
    java_bin: str,
    java_args: str,
):
    date_time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
    print(f"{date_time_stamp}[moa] Executing {learner_spec['name']} on {dataset_name}")

    raw_results = []

    for repetition in range(1, repetitions + 1):
        print(f"{date_time_stamp}[moa]\trepetition {repetition}")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, dir=RESULTS_DIR
        ) as temp_file:
            dump_file = Path(temp_file.name)

        try:
            task_string = build_task_string(
                learner_cli=learner_spec["classifier_cli"],
                arff_path=arff_path,
                max_instances=max_instances,
                dump_file=dump_file,
            )
            wallclock, _, _ = run_moa_task(
                task_string, java_bin=java_bin, java_args=java_args
            )
            accuracy, cpu_time = extract_final_metrics(dump_file)
            raw_results.append(
                {
                    "library": "moa",
                    "repetition": repetition,
                    "dataset": dataset_name,
                    "learner": learner_spec["name"],
                    "hyperparameters": str(learner_spec["hyperparameters"]),
                    "accuracy": accuracy,
                    "wallclock": wallclock,
                    "cpu_time": cpu_time,
                }
            )
        finally:
            dump_file.unlink(missing_ok=True)

    raw_df = pd.DataFrame(raw_results)
    aggregated_df = pd.DataFrame(
        {
            "library": "moa",
            "dataset": dataset_name,
            "learner": learner_spec["name"],
            "hyperparameters": str(learner_spec["hyperparameters"]),
            "repetitions": repetitions,
            "avg_accuracy": raw_df["accuracy"].mean(),
            "std_accuracy": raw_df["accuracy"].std(),
            "avg_wallclock": raw_df["wallclock"].mean(),
            "std_wallclock": raw_df["wallclock"].std(),
            "avg_cpu_time": raw_df["cpu_time"].mean(),
            "std_cpu_time": raw_df["cpu_time"].std(),
        },
        index=[0],
    )
    return aggregated_df, raw_df


def benchmark_moa(
    intermediary_results: pd.DataFrame,
    raw_intermediary_results: pd.DataFrame,
    *,
    dataset_name: str,
    arff_path: Path,
    results_output_csv: Path,
    raw_results_output_csv: Path,
    max_instances: int,
    repetitions: int,
    java_bin: str,
    java_args: str,
    include_threaded_arf: bool = True,
):
    for learner_spec in selected_learners(include_threaded_arf):
        result_df, raw_df = moa_experiment(
            dataset_name=dataset_name,
            learner_spec=learner_spec,
            arff_path=arff_path,
            repetitions=repetitions,
            max_instances=max_instances,
            java_bin=java_bin,
            java_args=java_args,
        )
        intermediary_results = checkpoint_results(
            intermediary_results, result_df, results_output_csv
        )
        raw_intermediary_results = checkpoint_results(
            raw_intermediary_results, raw_df, raw_results_output_csv
        )
    return intermediary_results, raw_intermediary_results


def plot_performance(
    df: pd.DataFrame,
    plot_prefix: Path,
    *,
    dark_theme: bool = False,
    plot_title: str | None = None,
    dataset_name: str | None = None,
    max_instances: int | None = None,
    include_threaded_arf: bool = True,
):
    ordered_algorithms = learner_names(include_threaded_arf)
    df = df.copy()
    df["learner"] = pd.Categorical(df["learner"], ordered_algorithms, ordered=True)
    df = df.sort_values("learner")
    if df.empty:
        print("No benchmark results available to plot.")
        return

    measures = ["accuracy", "wallclock", "cpu_time"]
    for measure in measures:
        fig, ax = plt.subplots(figsize=(10, 6))
        if dark_theme:
            fig.patch.set_facecolor("#101418")
            ax.set_facecolor("#101418")
            text_color = "#f3f5f7"
            grid_color = "#3a444d"
            bar_color = "#f5a623"
            error_bar_color = "#f3f5f7"
        else:
            text_color = "black"
            grid_color = "#d0d7de"
            bar_color = "#d97706"
            error_bar_color = "black"

        metric_title = measure.replace("_", " ").title()
        title_base = plot_title
        if title_base is None and dataset_name is not None and max_instances is not None:
            title_base = build_default_plot_title(dataset_name, max_instances)

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

        means = [df.loc[df["learner"] == learner, f"avg_{measure}"].iloc[0] if any(df["learner"] == learner) else np.nan for learner in ordered_algorithms]
        stds = [df.loc[df["learner"] == learner, f"std_{measure}"].iloc[0] if any(df["learner"] == learner) else np.nan for learner in ordered_algorithms]
        x_positions = np.arange(len(ordered_algorithms))
        valid_mask = pd.Series(means).notna().to_numpy()

        ax.bar(
            x_positions[valid_mask],
            pd.Series(means)[valid_mask],
            yerr=pd.Series(stds)[valid_mask],
            width=0.8,
            color=bar_color,
            ecolor=error_bar_color,
            capsize=4,
        )
        ax.set_xticks(x_positions, ordered_algorithms)
        legend = ax.legend(handles=[Patch(color=bar_color, label="moa")])
        if dark_theme:
            legend.get_frame().set_facecolor("#101418")
            legend.get_frame().set_edgecolor("#3a444d")
            for text in legend.get_texts():
                text.set_color(text_color)
        fig.tight_layout()
        fig.savefig(f"{plot_prefix}_{measure}.png", facecolor=fig.get_facecolor())
        plt.close(fig)


if __name__ == "__main__":
    overall_start_time = time.time()
    args = parse_args()
    if args.plot_only and args.output_prefix is None:
        raise ValueError("--plot-only requires an explicit --output-prefix.")

    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_config = SUPPORTED_DATASETS[args.dataset]
    dataset_name = dataset_config["dataset_name"]
    experiment_name = experiment_id(dataset_name, args.max_instances)
    experiment_dir = RESULTS_DIR / experiment_name
    output_prefix = args.output_prefix or build_default_output_prefix(
        dataset_name, args.max_instances
    )
    output_paths = resolve_output_paths(output_prefix, experiment_dir)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        combined_results = pd.read_csv(output_paths["results_csv"])
        plot_performance(
            combined_results,
            output_paths["plot_prefix"],
            dark_theme=args.dark_theme,
            plot_title=args.plot_title,
            dataset_name=None,
            max_instances=args.max_instances,
            include_threaded_arf=not args.skip_threaded_arf,
        )
        print(f"Regenerated plots from {output_paths['results_csv']}")
        sys.exit(0)

    dataset_stream, dataset_arff_path, dataset_label = ensure_dataset_assets(args.dataset)
    machine_info = write_machine_info(output_paths["machine_info_json"])

    combined_results = pd.DataFrame()
    raw_results = pd.DataFrame()

    combined_results, raw_results = benchmark_moa(
        combined_results,
        raw_results,
        dataset_name=dataset_label,
        arff_path=dataset_arff_path,
        results_output_csv=output_paths["results_csv"],
        raw_results_output_csv=output_paths["raw_results_csv"],
        max_instances=args.max_instances,
        repetitions=args.repetitions,
        java_bin=args.java_bin,
        java_args=args.java_args,
        include_threaded_arf=not args.skip_threaded_arf,
    )

    plot_performance(
        combined_results,
        output_paths["plot_prefix"],
        dark_theme=args.dark_theme,
        plot_title=args.plot_title,
        dataset_name=dataset_label,
        max_instances=args.max_instances,
        include_threaded_arf=not args.skip_threaded_arf,
    )

    elapsed_seconds = time.time() - overall_start_time
    write_experiment_summary(
        output_paths["experiment_md"],
        dataset_name=dataset_label,
        dataset_class_name=dataset_config["class_name"],
        max_instances=args.max_instances,
        repetitions=args.repetitions,
        stream=dataset_stream,
        experiment_date=run_started_at,
        elapsed_seconds=elapsed_seconds,
        machine_info=machine_info,
        include_threaded_arf=not args.skip_threaded_arf,
    )
    write_configurations_summary(
        output_paths["configurations_md"],
        dataset_name=dataset_label,
        max_instances=args.max_instances,
        include_threaded_arf=not args.skip_threaded_arf,
    )

    print(combined_results)
