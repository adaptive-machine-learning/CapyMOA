from capymoa.base import MOAClusterer
import typing
from moa.clusterers.denstream import WithDBSCAN as _MOA_denstream_with_dbscan
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
# import numpy as np


class Denstream_with_dbscan(MOAClusterer):
    """
    Denstream clustering algorithm with DBSCAN Macro-clustering.
    """

    def __init__(
        self,
        schema: typing.Union[Schema, None] = None,
        horizon: int = 1000,
        epsilon: float = 0.02,
        beta: float = 0.2,
        mu: int = 1,
        init_points: int = 1000,
        offline_option: float = 2,
        lambda_option: float = 0.25,
        speed: int = 100,
    ):
        """Clustream clusterer.

        :param schema: The schema of the stream
        :param horizon: The size of the time window
        :param epsilon: The epsilon neighborhood
                :param beta: The beta parameter
                :param mu: The mu parameter
                :param init_points: The number of initial points
                :param offline_option: The offline multiplier for epsilon
                :param lambda_option: The lambda option
                :param speed: Number of incoming data points per time unit
        """

        mapping = {
            "horizon": "-h",
            "epsilon": "-e",
            "beta": "-b",
            "mu": "-m",
            "init_points": "-i",
            "offline_option": "-o",
            "lambda_option": "-l",
            "speed": "-s",
        }

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        self.moa_learner = _MOA_denstream_with_dbscan()
        super(Denstream_with_dbscan, self).__init__(
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
        return "Denstream with DBSCAN"
