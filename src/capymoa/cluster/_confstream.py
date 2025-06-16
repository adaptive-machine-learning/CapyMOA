import typing
from moa.clusterers.meta import ConfStream as _MOA_ConfStream
from capymoa.stream import Schema
# from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.cluster.base import MOACluster

class ConfStream(MOACluster):
	"""
	ConfStream clustering algorithm.
	Citation: 
	"""

	def __init__(
			self,
			schema: typing.Union[Schema, None] = None,
			filename: typing.Union[str, None] = None,
		):
			"""ConfStream clustering algorithm.

			:param schema: The schema of the stream
			"""
			
			# mapping = {
			# 	"num_clusters": "-k",
			# 	"num_dimensions": "-d",
			# 	"max_num_cluster_features": "-m",
			# 	"num_projections": "-p",
			# }

			# config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
			self.moa_learner = _MOA_ConfStream()
			if filename is None:
				raise ValueError("setting file is not specified, please provide a valid filename")

			self.moa_learner.fileOption.setValue(filename)
			super(ConfStream, self).__init__(
				schema=schema, 
				# CLI=config_str, 
				moa_learner=self.moa_learner,

			)

	def implements_micro_clusters(self) -> bool:
		return True

	def implements_macro_clusters(self) -> bool:
		return False

	def _is_visualization_supported(self):
		return False
