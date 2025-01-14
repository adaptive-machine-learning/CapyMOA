from capymoa.base import MOAClusterer
import typing
from moa.clusterers.clustream import WithKmeans as _MOA_Clustream_WKM
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
# import numpy as np


class Clustream_with_kmeans(MOAClusterer):
    """
    Clustream clustering algorithm without Macro-clustering.
    """

    def __init__(
        self,
        schema: typing.Union[Schema, None] = None,
        time_window: int = 1000,
        max_num_kernels: int = 100,
        kernel_radi_factor: float = 2,
        k_option: int = 5,
    ):
        """Clustream clusterer with K-means offline clustering.

        :param schema: The schema of the stream.
        :param time_window: The size of the time window.
        :param max_num_kernels: Maximum number of micro kernels to use.
        :param kernel_radi_factor: Multiplier for the kernel radius
        :param k_option: Number of clusters to use in the k-means offline step
        """

        mapping = {
            "time_window": "-h",
            "max_num_kernels": "-m",
            "kernel_radi_factor": "-t",
            "k_option": "-k",
        }

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        self.moa_learner = _MOA_Clustream_WKM()
        super(Clustream_with_kmeans, self).__init__(
            schema=schema, CLI=config_str, moa_learner=self.moa_learner
        )

    def implements_micro_clusters(self) -> bool:
        return True

    def implements_macro_clusters(self) -> bool:
        return True

    # def predict(self, X):
    #     clusters = self.get_micro_clustering_result()
    #     min_dist = np.inf
    #     closest_center = None
    #     for center in clusters.get_centers():
    #         if np.linalg.norm(center - X) < min_dist:
    #             min_dist = np.linalg.norm(center - X)
    #             closest_center = center
    #     print(closest_center)
    #     return closest_center

    def __str__(self):
        return "Clustream with KMeans"
