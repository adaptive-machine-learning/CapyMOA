import typing
from moa.clusterers import CobWeb as _MOA_CobWeb
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.cluster.base import MOACluster


class CobWeb(MOACluster):
    """
    CobWeb clustering algorithm without Macro-clustering.
    """

    def __init__(
        self,
        schema: typing.Union[Schema, None] = None,
        acuity: int = 1,
        cutoff: float = 0.002,
        randomSeed: int = 1,
    ):
        """CobWeb clusterer.

        :param schema: The schema of the stream
        :param acuity: The acuity parameter for CobWeb
        :param cutoff: The cutoff parameter for CobWeb
		:param randomSeed: The random seed for reproducibility
        """
        
        mapping = {
            "acuity": "-a", 
            "cutoff": "-c", 
            "randomSeed": "-r",
		}

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        self.moa_learner = _MOA_CobWeb()
        super(CobWeb, self).__init__(
            schema=schema, CLI=config_str, moa_learner=self.moa_learner
        )

    def implements_micro_clusters(self) -> bool:
        return False

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

    def _is_visualization_supported(self):
        return False
