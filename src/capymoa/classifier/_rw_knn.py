from capymoa.base import MOAClassifier
from moa.classifiers.lazy import RW_kNN as _MOA_RWkNN
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals


class RWkNN(MOAClassifier):
    """RwkNN

    Reference:
    
    'Incremental k-Nearest Neighbors Using Reservoir Sampling for Data Streams
    Maroua Bahri, Albert Bifet
    Discovery Science: 24th International Conference, 2021
    <https://link.springer.com/chapter/10.1007/978-3-030-88942-5_10>`_

    Example usages:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import RWkNN
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = RWkNN(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    81.8
    """


    def __init__(
        self,
        schema: Schema,
        k: int = 5,
        limitW: int = 500,
        limitR: int = 500,
        nearest_neighbor_search: str = "LinearNN" 
    ):
        
        """ RW KNN Classifier

        :param schema: The schema of the stream.
        :param k: The number of neighbors.
        :param limit_w: The maximum number of instances to store in the window.
        :param limit_r: The maximum number of instances to store in the reservoir.
        :param nearest_neighbor_search: Nearest Neighbour Search to use.
        """

        self.nearest_neighbor_search = nearest_neighbor_search
        if isinstance(self.nearest_neighbor_search, str) and (nearest_neighbor_search == "LinearNN" or nearest_neighbor_search == "KDTree"):
            self.nearest_neighbor_search = nearest_neighbor_search
        else:
            # Raise an exception with information about valid options for max_features
            raise ValueError("Invalid value for nearest_neighbor_search. Valid options: LinearNN, KDTree")


        mapping = {
            "k": "-k",
            "limitW": "-w",
            "limitR": "-r",
            "nearest_neighbor_search": "-n",
        }


        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        self.moa_learner = _MOA_RWkNN()
        super(RWkNN, self).__init__(
            schema=schema,
            CLI=config_str,
            moa_learner=self.moa_learner,
        )

