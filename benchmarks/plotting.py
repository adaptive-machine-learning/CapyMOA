from pathlib import Path

import matplotlib
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def format_instance_count(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def build_default_plot_title(dataset_name: str, max_instances: int) -> str:
    return f"{dataset_name} {format_instance_count(max_instances)}"


def sanitize_filename(value: str) -> str:
    allowed = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "plot"


def plot_performance(
    df,
    plot_prefix,
    *,
    dark_theme=False,
    plot_title=None,
    dataset_name=None,
    max_instances=None,
    algorithm_order=None,
    library_order=None,
    library_colors=None,
    unpaired_library=None,
    unpaired_label=None,
    unpaired_color_light="#4a7dff",
    unpaired_color_dark="#6aa9ff",
):
    ordered_algorithms = list(algorithm_order or [])
    if not ordered_algorithms:
        ordered_algorithms = sorted(set(df["learner"]))
    else:
        ordered_algorithms = [name for name in ordered_algorithms if name in set(df["learner"])]

    df = df.copy()
    df["learner"] = pd.Categorical(df["learner"], ordered_algorithms, ordered=True)
    df = df.sort_values("learner")

    available_libraries = set(df["library"])
    libraries = list(library_order or [])
    if libraries:
        libraries = [library for library in libraries if library in available_libraries]
    else:
        libraries = sorted(available_libraries)

    if len(libraries) == 0:
        print("No benchmark results available to plot.")
        return

    plot_df = df[df["learner"].notna()].set_index(["learner", "library"])

    if library_colors is None:
        library_colors = (
            {"capymoa": "#44d17a", "river": "#ff6b6b", "moa": "#f5a623"}
            if dark_theme
            else {"capymoa": "green", "river": "red", "moa": "#d97706"}
        )

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
                title_base = build_default_plot_title(inferred_dataset_name, max_instances)

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
        paired_learners = {
            learner
            for learner in ordered_algorithms
            if all((learner, library) in plot_df.index for library in libraries)
        }
        unpaired_color = unpaired_color_dark if dark_theme else unpaired_color_light

        for idx, library in enumerate(libraries):
            means = []
            stds = []
            bar_colors = []
            for learner in ordered_algorithms:
                if (learner, library) in plot_df.index:
                    row = plot_df.loc[(learner, library)]
                    means.append(row[f"avg_{measure}"])
                    stds.append(row[f"std_{measure}"])
                    if library == unpaired_library and learner not in paired_learners:
                        bar_colors.append(unpaired_color)
                    else:
                        bar_colors.append(library_colors.get(library, "gray"))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                    bar_colors.append(library_colors.get(library, "gray"))

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

        legend_handles = []
        for library in libraries:
            legend_handles.append(
                Patch(color=library_colors.get(library, "gray"), label=library)
            )
        if unpaired_library in libraries and unpaired_label is not None:
            if any(
                learner not in paired_learners
                for learner in ordered_algorithms
                if (learner, unpaired_library) in plot_df.index
            ):
                legend_handles.insert(
                    libraries.index(unpaired_library) + 1,
                    Patch(color=unpaired_color, label=unpaired_label),
                )

        legend = ax.legend(handles=legend_handles)
        if dark_theme and legend is not None:
            legend.get_frame().set_facecolor("#101418")
            legend.get_frame().set_edgecolor("#3a444d")
            for text in legend.get_texts():
                text.set_color(text_color)

        fig.tight_layout()
        fig.savefig(f"{plot_prefix}_{measure}.png", facecolor=fig.get_facecolor())
        plt.close(fig)


def write_pulse_plots(
    pulse_csv: Path,
    pulse_dir: Path,
    *,
    dark_theme: bool = False,
    library_colors=None,
):
    if not pulse_csv.exists():
        return

    pulse_df = pd.read_csv(pulse_csv)
    if pulse_df.empty:
        return

    pulse_dir.mkdir(parents=True, exist_ok=True)

    if library_colors is None:
        library_colors = (
            {"capymoa": "#44d17a", "river": "#ff6b6b", "moa": "#f5a623"}
            if dark_theme
            else {"capymoa": "green", "river": "red", "moa": "#d97706"}
        )

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
        else:
            text_color = "black"
            grid_color = "#d0d7de"

        ax.set_title(f"{algorithm} Pulse", color=text_color)
        ax.set_xlabel("Processed Instances", color=text_color)
        ax.set_ylabel("Delta (s)", color=text_color)
        ax.tick_params(axis="x", colors=text_color)
        ax.tick_params(axis="y", colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.grid(axis="y", color=grid_color, alpha=0.35)
        ax.set_axisbelow(True)

        for platform_name in sorted(algorithm_df["platform"].dropna().unique()):
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
            color = library_colors.get(platform_name, "gray")
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
