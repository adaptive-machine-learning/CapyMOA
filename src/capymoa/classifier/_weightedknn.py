from capymoa.base import MOAClassifier
from moa.classifiers.lazy import WeightedkNN as _MOA_WeightedkNN
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals


class WeightedkNN(MOAClassifier):
    """WeightedKNN
    Reference:

    'Effective Weighted k-Nearest Neighbors for Dynamic Data Streams'
    Maroua Bahri
    IEEE International Conference on Big Data (Big Data), 2022
    <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10020652>
    Example usages:
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import WeightedkNN
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = WeightedkNN(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    74.7
    """

    def __init__(self, schema: Schema, k: int = 10, limit: int = 1000):
        """Weighted KNN Classifier
        :param schema: The schema of the stream.
        :param k: The number of neighbors.
        :param w: The maximum number of instances to store.
        """

        mapping = {
            "k": "-k",
            "limit": "-w",
        }

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        self.moa_learner = _MOA_WeightedkNN()
        super(WeightedkNN, self).__init__(
            schema=schema,
            CLI=config_str,
            moa_learner=self.moa_learner,
        )
