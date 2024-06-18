from .onlineitree import OnlineITree
from abc import ABC, abstractmethod
from capymoa.base import AnomalyDetector
from capymoa.instance import Instance, LabelIndex
from capymoa.stream._stream import Schema
from capymoa.type_alias import AnomalyScore
from multiprocessing import cpu_count
from numpy import ndarray
from typing import Optional


class OnlineIForest(AnomalyDetector):
    """ Online Isolation Forest

    This class implements the Online Isolation Forest (oIFOR) algorithm, which is
    an ensemble anomaly detector capable of adapting to concept drift.

    Reference:
    @inproceedings{Leveni2024,
    title        = {Online Isolation Forest},
    author       = {Leveni, Filippo and Weigert Cassales, Guilherme and Pfahringer, Bernhard and Bifet, Albert and Boracchi, Giacomo},
    booktitle    = {International Conference on Machine Learning (ICML)},
    year         = {2024},
    organization = {Proceedings of Machine Learning Research (PMLR)}}
    Example usage:
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import OnlineIForest
    >>> from capymoa.evaluation import AUCEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = OnlineIForest.create('boundedrandomprojectiononlineiforest', schema=schema)
    >>> evaluator = AUCEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.6
    """
    @staticmethod
    def create(iforest_type: str = 'boundedrandomprojectiononlineiforest', **kwargs) -> 'OnlineIForest':
        # TODO: Find an alternative solution to overcome circular imports
        from .BoundedRandomProjectionOnlineIForest import BoundedRandomProjectionOnlineIForest
        # Map iforest type to an iforest class
        iforest_type_to_iforest_map: dict = {'boundedrandomprojectiononlineiforest': BoundedRandomProjectionOnlineIForest}
        if iforest_type not in iforest_type_to_iforest_map:
            raise ValueError('Bad iforest type {}'.format(iforest_type))
        return iforest_type_to_iforest_map[iforest_type](**kwargs)

    def __init__(self, num_trees: int, window_size: int, branching_factor: int, max_leaf_samples: int, type: str,
                 subsample: float, n_jobs: int, schema: Schema = None, random_seed: int = 1):
        """Construct an Online Isolation Forest anomaly detector

        :param num_trees: Number of trees in the ensemble.
        :param window_size: The size of the window for each tree.
        :param branching_factor: Branching factor of each tree.
        :param max_leaf_samples: Maximum number of samples per leaf. When this number is reached, a split is performed.
        :param type: Type of split performed. If "adaptive", the max_leaf_samples grows with tree depth.
        :param subsample: Probability of learning a new sample in each tree.
        :param n_jobs: Number of parallel jobs.
        :param schema: The schema of the stream. If not provided, it will be inferred from the data.
        :param random_seed: Random seed for reproducibility.
        """
        super().__init__(schema=schema, random_seed=random_seed)
        self.num_trees: int = num_trees
        self.window_size: int = window_size
        self.branching_factor: int = branching_factor
        self.max_leaf_samples: int = max_leaf_samples
        self.type: str = type
        self.subsample: float = subsample
        self.trees: list[OnlineITree] = []
        self.data_window: list[ndarray] = []
        self.data_size: int = 0
        self.normalization_factor: float = 0
        self.n_jobs: int = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

    @abstractmethod
    def learn_batch(self, data: ndarray):
        pass

    @abstractmethod
    def score_batch(self, data: ndarray):
        pass

    @abstractmethod
    def predict_batch(self, data: ndarray):
        pass

    def train(self, instance: Instance):
        data: ndarray = instance.x.reshape((1, -1))
        return self.learn_batch(data)

    def predict(self, instance: Instance) -> Optional[LabelIndex]:
        pass

    def score_instance(self, instance: Instance) -> AnomalyScore:
        data: ndarray = instance.x.reshape((1, -1))
        return self.score_batch(data)
