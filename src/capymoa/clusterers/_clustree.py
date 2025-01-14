from capymoa.base import MOAClusterer
import typing
from moa.clusterers.clustree import ClusTree as _MOA_ClusTree
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
# import numpy as np


class ClusTree(MOAClusterer):
    """
    ClusTree clustering algorithm without Macro-clustering.
    """

    def __init__(
        self,
        schema: typing.Union[Schema, None] = None,
        horizon: int = 1000,
        max_height: int = 8,
        breadth_first_strategy: bool = False,
    ):
        """Clustream clusterer.

        :param schema: The schema of the stream
        :param horizon: The size of the time window
        :param max_height: The maximum height of the tree
        :param breadth_first_strategy: Whether to use breadth-first strategy
        """

        mapping = {"horizon": "-h", "max_height": "-H", "breadth_first_strategy": "-B"}

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        self.moa_learner = _MOA_ClusTree()
        super(ClusTree, self).__init__(
            schema=schema, CLI=config_str, moa_learner=self.moa_learner
        )

    def implements_micro_clusters(self) -> bool:
        return True

    def implements_macro_clusters(self) -> bool:
        return False

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
