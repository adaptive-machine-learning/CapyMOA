import typing
from moa.clusterers.dstream import Dstream as _MOA_Dstream
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.cluster.base import MOACluster


class DStream(MOACluster):
	"""
	DStream clustering algorithm.
	"""

	def __init__(
		self,
		schema: typing.Union[Schema, None] = None,
		decayFactor: float = 0.998,
		cm: float = 3.0,
		cl: float = 0.8,
		beta: float = 0.3,
	):
		"""DStream clusterer.

		:param schema: The schema of the stream
		:param decayFactor: The decay factor (lambda) for DStream
		:param cm: The cm parameter for DStream
		:param cl: The cl parameter for DStream
		:param beta: The beta parameter for DStream
		"""
		
		mapping = {
			"decayFactor": "-d",
			"cm": "-m",
			"cl": "-l",
			"beta": "-b",
		}

		config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
		self.moa_learner = _MOA_Dstream()
		super(DStream, self).__init__(
			schema=schema, CLI=config_str, moa_learner=self.moa_learner
		)

	def implements_micro_clusters(self) -> bool:
		return False

	def implements_macro_clusters(self) -> bool:
		return True

	def _is_visualization_supported(self):
		return False
