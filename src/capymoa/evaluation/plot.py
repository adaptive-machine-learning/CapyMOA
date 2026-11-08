"""Plots that work on the results of any domain."""

from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from capymoa.evaluation.results import RunInfo


def _plot_drifts(
    result: RunInfo,
    start: float = 0,
    end: float = float("inf"),
    concepts_y: float | None = None,
) -> None:
    """Draw the drifts of a result between ``start`` and ``end``.

    A drift is a red line. A gradual drift also gets a red band as wide as its
    window. If ``concepts_y`` is given, each concept is a dashed line at that
    height.
    """
    drifts = result.get("drifts") or []
    widths = result.get("drift_widths") or [0] * len(drifts)
    for position, width in zip(drifts, widths):
        if start < position < end:
            plt.axvline(position, color="red", linestyle="-")
            if width > 1:
                plt.axvspan(
                    position - width / 2, position + width / 2, alpha=0.2, color="red"
                )

    if concepts_y is None:
        return
    colours: dict[str, int] = {}
    for concept in result.get("concepts") or []:
        label = None
        if concept["id"] not in colours:
            colours[concept["id"]] = len(colours)
            label = concept["id"]
        plt.hlines(
            y=concepts_y,
            xmin=concept["start"],
            xmax=concept["end"],
            color=plt.cm.tab10(colours[concept["id"]]),
            linestyle="--",
            linewidth=2,
            label=label,
        )


def plot_windowed_results(
    *results: RunInfo,
    metric: str,
    plot_title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figure_path: str = "./",
    figure_name: str | None = None,
    save_only: bool = True,
    prevent_plotting_drifts: bool = False,
    ymin: float | None = None,
    ymax: float | None = None,
):
    """Plot a windowed metric of one or more results as lines.

    Results that have no ``windowed`` table or lack the metric are skipped.
    Drifts of the first result are drawn in red, and its concepts (if any) as
    dashed lines.

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from capymoa.classifier import HoeffdingTree, NaiveBayes, evaluate_classifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation.plot import plot_windowed_results
    >>> stream = ElectricityTiny()
    >>> results = [
    ...     evaluate_classifier(stream, learner, max_instances=1000)
    ...     for learner in (
    ...         HoeffdingTree(stream.get_schema()), NaiveBayes(stream.get_schema())
    ...     )
    ... ]
    >>> plot_windowed_results(*results, metric="accuracy", save_only=False)

    :param results: The results to plot.
    :param metric: The column of ``windowed`` to plot, such as ``"accuracy"``.
    :param plot_title: Title of the plot, defaults to the metric.
    :param xlabel: Label of the x axis, defaults to ``"# Instances"``.
    :param ylabel: Label of the y axis, defaults to the metric.
    :param figure_path: Directory to save the figure in.
    :param figure_name: File name of the figure. Defaults to one made from the
        metric and the learners.
    :param save_only: Save the figure to a file, else show it.
    :param prevent_plotting_drifts: Do not draw the drifts.
    :param ymin: Lower limit of the y axis, defaults to the lowest value.
    :param ymax: Upper limit of the y axis, defaults to the highest value.
    """
    plotted = []
    for result in results:
        windowed = result.get("windowed")
        if windowed is None or metric not in windowed:
            print(
                f"Column '{metric}' not found in the windowed results of "
                f"{result['learner']}. Skipping."
            )
        else:
            plotted.append(result)

    if not plotted:
        print("No valid results to plot.")
        return

    # Calculate ymin and ymax if not provided
    if ymin is None or ymax is None:
        all_values = np.concatenate([r["windowed"][metric] for r in plotted])
        if ymin is None:
            ymin = np.nanmin(all_values)
        if ymax is None:
            ymax = np.nanmax(all_values)

    # Add padding to ymin and ymax to prevent clipping
    padding = 0.05 * (ymax - ymin)
    ymin -= 2 * padding
    ymax += 2 * padding

    plt.figure(figsize=(12, 5))
    for result in plotted:
        windowed = result["windowed"]
        y_values: np.ndarray = windowed[metric]
        plt.plot(
            windowed["instances"],
            y_values,
            label=result["learner"],
            marker="o",
            linestyle="-",
            markersize=5,
        )
        if np.isnan(y_values).any():
            warn(f"Results for '{result['learner']}' contains NaNs.")

    if not prevent_plotting_drifts:
        _plot_drifts(plotted[0], concepts_y=ymin + padding)

    plt.ylim(ymin, ymax)
    xlabel = xlabel if xlabel is not None else "# Instances"
    plt.xlabel(xlabel)
    ylabel = ylabel if ylabel is not None else metric
    plt.ylabel(ylabel)
    plt.title(plot_title if plot_title is not None else metric)
    plt.legend()
    plt.grid(True)

    # Show the plot or save it to the specified path
    if not save_only:
        plt.show()
    elif figure_path is not None:
        if figure_name is None:
            learner_names = "_".join(result["learner"] for result in plotted)
            figure_name = ylabel.replace(" ", "") + "_" + learner_names + ".pdf"
        plt.savefig(figure_path + figure_name)
