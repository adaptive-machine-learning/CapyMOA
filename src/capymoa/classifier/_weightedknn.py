from moa.classifiers.lazy import WeightedkNN as _MOA_WeightedkNN

from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.base import MOAClassifier
from capymoa.stream import Schema


class WeightedkNN(MOAClassifier):
    """Weighted k-Nearest Neighbour.

    Weighted k-Nearest Neighbour (SRP) [#0]_ is a lazy classifier.

    >>> from capymoa.classifier import WeightedkNN
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = WeightedkNN(stream.get_schema())
    >>> results = evaluate_classifier(stream, classifier, max_instances=1000)
    >>> print(f"{results['accuracy']:.1f}")
    74.7

    .. [#0] `Effective Weighted k-Nearest Neighbors for Dynamic Data Streams’ Maroua
             Bahri IEEE International Conference on Big Data (Big Data), 2022
             <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10020652>`_
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
        super().__init__(
            schema=schema,
            CLI=config_str,
            moa_learner=self.moa_learner,
        )
