import typing
from moa.clusterers.kmeanspm import BICO as _MOA_BICO
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.cluster.base import MOACluster


class BICO(MOACluster):
	"""
	BICO clustering algorithm.
	Citation: Hendrik Fichtenberger, Marc Gillé, Melanie Schmidt,
	Chris Schwiegelshohn, Christian Sohler:
	BICO: BIRCH Meets Coresets for k-Means Clustering.
	ESA 2013: 481-492 (2013)
	http://ls2-www.cs.tu-dortmund.de/bico/
	"""

	def __init__(
			self,
			schema: typing.Union[Schema, None] = None,
			num_clusters: int = 5,
			num_dimensions: int = 10,
			max_num_cluster_features: int = 1000,
			num_projections: int = 10
		):
			"""BICO clustering algorithm.

			:param schema: The schema of the stream
			:param num_clusters: Number of desired centers
			:param num_dimensions: Number of the dimensions of the input points
			:param max_num_cluster_features: Maximum size of the coreset
			:param num_projections: Number of random projections used for the nearest neighbour search.
			"""

			mapping = {
				"num_clusters": "-k",
				"num_dimensions": "-d",
				"max_num_cluster_features": "-n",
				"num_projections": "-p",
			}

			config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
			self.moa_learner = _MOA_BICO()
			super(BICO, self).__init__(
				schema=schema, CLI=config_str, moa_learner=self.moa_learner
			)

	def implements_micro_clusters(self) -> bool:
		return True

	def implements_macro_clusters(self) -> bool:
		return True

	def _is_visualization_supported(self):
		return False
