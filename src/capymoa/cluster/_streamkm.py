import typing
from moa.clusterers.streamkm import StreamKM as _MOA_StreamKM
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.cluster.base import MOACluster

class StreamKM(MOACluster):
	"""
	StreamKM++ clustering algorithm.
	Citation: Marcel R. Ackermann, Christiane Lammersen, Marcus Märtens,
	Christoph Raupach, Christian Sohler, Kamil Swierkot: StreamKM++: A
	Clustering Algorithms for Data Streams. ALENEX 2010: 173-187	
	"""

	def __init__(
			self,
			schema: typing.Union[Schema, None] = None,
			coreset_size: int = 10000,
			num_clusters: int = 5,
			data_length: int = 100000,
			evaluate_final_only: bool = False,
			random_seed: int = 1
		):
			"""StreamKM++ clustering algorithm.

			:param schema: The schema of the stream
			:param coreset_size: Size of the coreset (m)
			:param num_clusters: Number of clusters to compute
			:param data_length: Length of the data stream (n)
			:param evaluate_final_only: If true, only the final clustering is evaluated
			:param random_seed: Seed for random behaviour of the classifier
			"""
			
			mapping = {
				"coreset_size": "-s",
				"num_clusters": "-k",
				"data_length": "-l",
				"evaluate_final_only": "-e",
				"random_seed": "-r",
			}

			config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
			self.moa_learner = _MOA_StreamKM()
			super(StreamKM, self).__init__(
				schema=schema, CLI=config_str, moa_learner=self.moa_learner
			)

	def implements_micro_clusters(self) -> bool:
		return False

	def implements_macro_clusters(self) -> bool:
		return True

	def _is_visualization_supported(self):
		return False
