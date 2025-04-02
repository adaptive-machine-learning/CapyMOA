import matplotlib.pyplot as plt
from capymoa.cluster.base import ClusteringResult, Cluster
from datetime import datetime
import os
import shutil
from PIL import Image
import glob
import numpy as np

import itertools

def _plot_clustering_state(
    clusterer_name,
    macro: ClusteringResult,
    micro: ClusteringResult,
    figure_path="./",
    figure_name=None,
    show_fig=True,
    save_fig=False,
    make_gif=False,
    show_ids=False,
):
    """
    Internal function to plot the current state of a clustering algorithm.
    This should not be used directly by the user.
    """
    fig, ax = plt.subplots()
    ma_centers = macro.get_centers()
    ma_weights = macro.get_weights()
    ma_radii = macro.get_radii()
    ma_ids = macro.get_ids()
    max_radius = 0

    # Macro-clustering visualization
    if len(ma_centers) > 0:
        if ma_weights is not None:
            scatter = ax.scatter(
                *zip(*ma_centers),
                c=ma_weights,
                cmap="copper",
                label="Centers",
                s=50,
                edgecolor="k",
                linewidths=0.4,
            )
            cbar = fig.colorbar(scatter)
            cbar.set_label("Macro cluster Weights")

        # Add circles representing the radius of each center
        # keep the largest radius for the plot
        if ma_radii is not None:
            for (x, y), radius in zip(ma_centers, ma_radii):
                if radius > max_radius:
                    max_radius = radius
                circle = plt.Circle((x, y), radius, color="red", fill=False, lw=0.4)
                ax.add_patch(circle)

        # Annotate the centers with cluster IDs
        if show_ids:
            if ma_ids is not None:
                for (x, y), cluster_id in zip(ma_centers, ma_ids):
                    ax.text(
                        x,
                        y,
                        str(cluster_id),
                        fontsize=7,
                        ha="center",
                        va="center",
                        color="white",
                    )
            else:
                for (x, y), cluster_id in zip(ma_centers, range(len(ma_centers))):
                    ax.text(
                        x,
                        y,
                        str(cluster_id),
                        fontsize=7,
                        ha="center",
                        va="center",
                        color="white",
                    )

    # Micro-clustering visualization
    mi_centers = micro.get_centers()
    mi_weights = micro.get_weights()
    mi_radii = micro.get_radii()
    if len(mi_centers) > 0:
        if mi_weights is not None:
            scatter_mi = ax.scatter(
                *zip(*mi_centers),
                c=mi_weights,
                cmap="winter",
                s=10,
                edgecolor="k",
                linewidths=0.2,
            )
            cbar = fig.colorbar(scatter_mi)
            cbar.set_label("Micro cluster Weights")
        # Add circles representing the radius of each center
        for (x, y), radius in zip(mi_centers, mi_radii):
            if radius > max_radius:
                max_radius = radius
            circle = plt.Circle((x, y), radius, color="blue", fill=False, lw=0.2)
            ax.add_patch(circle)
        # # Annotate the centers with cluster IDs
        # for (x, y), cluster_id in zip(mi_centers, range(len(mi_centers))):
        #     ax.text(x, y, str(cluster_id), fontsize=7, ha='center', va='center', color='white')

    # Add labels and title
    output_name = f"Clustering from {clusterer_name}"
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_title(output_name)
    # # Create a proxy artist for the minimum weight and add to the legend
    # proxy_artist = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.copper(0), markersize=10, label=f'Centers')
    # ax.legend(handles=[proxy_artist])
    ax.axis("equal")  # Ensure that the circles are not distorted
    # Show the plot or save it to the specified path
    if show_fig:
        plt.show()
    else:
        plt.close(fig)
    if make_gif:
        ax.set_title(output_name + f" {figure_name}")
        # include the max radius into the limits to make sure the circles are not cut off on all images
        minx = ax.get_xlim()[0] - max_radius
        maxx = ax.get_xlim()[1] + max_radius
        miny = ax.get_ylim()[0] - max_radius
        maxy = ax.get_ylim()[1] + max_radius
        return fig, minx, maxx, miny, maxy
    elif save_fig:
        # not a gif, use timestamp
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        figure_name = (
            figure_name if figure_name else f"clustering_result_{current_time}"
        )
        fig.savefig(figure_path + figure_name + ".png", dpi=300)


def plot_clustering_state(
    clusterer: Cluster,
    figure_path="./",
    figure_name=None,
    show_fig=True,
    save_fig=False,
    make_gif=False,
):
    """
    Plots the current state of a clustering algorithm.

    :param clusterer: Clusterer object
    :param figure_path: str, path to the directory where the figure is stored. Defaults to "./"
    :param figure_name: str, name of the figure. Defaults to None, in which case the name is going to be `clustering_result_<timestamp>`
    :param show_fig: bool, whether to show the figure. Defaults to True
    :param save_fig: bool, whether to save the figure. Defaults to False
    :param make_gif: bool, whether to make a gif. Defaults to False. This parameter is only used by `plot_clustering_evolution`
    """
    macro = clusterer.get_clustering_result()
    micro = clusterer.get_micro_clustering_result()
    _plot_clustering_state(
        str(clusterer),
        macro,
        micro,
        figure_path,
        figure_name,
        show_fig,
        save_fig,
        make_gif,
    )


def plot_clustering_evolution(
    clusteringResults,
    clean_up=True,
    filename=None,
    intermediate_directory=None,
    dpi=300,
    frame_duration=500,
    loops=0,
):
    """
    Plots the evolution of the clustering process as a gif.

    :param clusteringResults: ClusteringEvaluator object
    :param clean_up: bool, whether to remove the intermediate files after creating the gif
    :param filename: str, name of the gif file. Defaults to None, in which case the filename is going to be `<clusteringResults.clusterer_name>_clustering_evolution.gif`
    :param intermediate_directory: str, path to the directory where the intermediate files are stored. Defaults to None, in which case the files are stored in `./gifmaker/`
    :param dpi: int, resolution of the images. Defaults to 300
    :param frame_duration: int, duration of each frame in milliseconds. Defaults to 500
    :param loops: int, number of loops. Defaults to 0 (infinite loop)
    """
    macros = clusteringResults["macros"]
    micros = clusteringResults["micros"]
    gif_path = (
        "./gifmaker/" if intermediate_directory is None else intermediate_directory
    )

    os.makedirs(gif_path, exist_ok=True)
    figs = []
    # calculate the number of trailing zeroes needed for the image names
    num_images = len(macros) if len(macros) > len(micros) else len(micros)
    num_digits = len(str(num_images))
    maxx, maxy, minx, miny = -np.inf, -np.inf, np.inf, np.inf
    for i, (macro, micro) in enumerate(
        itertools.zip_longest(
            macros, micros, fillvalue=ClusteringResult([], [], [], [])
        )
    ):
        fig, e_minx, e_maxx, e_miny, e_maxy = _plot_clustering_state(
            clusteringResults.learner,
            macro,
            micro,
            figure_path=gif_path,
            figure_name=str(i).zfill(num_digits),
            show_fig=False,
            save_fig=True,
            make_gif=True,
        )
        if e_minx < minx:
            minx = e_minx
        if e_maxx > maxx:
            maxx = e_maxx
        if e_miny < miny:
            miny = e_miny
        if e_maxy > maxy:
            maxy = e_maxy
        figs.append(fig)

    # make the images with shared x and y lim
    for f in figs:
        f.gca().set_xlim([minx, maxx])
        f.gca().set_ylim([miny, maxy])
        f.savefig(f"{gif_path}{f.gca().get_title()}.png", dpi=dpi)
        plt.close(f)

    # Create a GIF from the images
    images = [Image.open(img) for img in sorted(glob.glob(gif_path + "*.png"))]

    images[0].save(
        f"{filename}"
        if filename is not None
        else f"{str(clusteringResults.learner).replace(' ', '_')}_clustering_evolution.gif", 
        save_all=True,
        append_images=images[1:],
        duration=frame_duration,  # Duration of each frame in milliseconds
        loop=loops,  # 0 means loop forever; set to 1 for single loop
    )
    print(f"GIF saved at {filename}" if filename is not None else f"GIF saved at {str(clusteringResults.learner).replace(' ', '_')}_clustering_evolution.gif")
    # clean up after making the gif
    if clean_up:
        shutil.rmtree(gif_path)
