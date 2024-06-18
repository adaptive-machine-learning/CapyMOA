from .onlineinode import OnlineINode
from abc import ABC, abstractmethod
from numpy import log, ndarray
from numpy.random import default_rng, Generator


class OnlineITree(ABC):
    @staticmethod
    def create(itree_type: str, **kwargs) -> 'OnlineITree':
        # TODO: Find an alternative solution to overcome circular imports
        from .BoundedRandomProjectionOnlineIForest import BoundedRandomProjectionOnlineITree
        # Map itree type to an itree class
        itree_type_to_itree_map: dict = {'boundedrandomprojectiononlineitree': BoundedRandomProjectionOnlineITree}
        if itree_type not in itree_type_to_itree_map:
            raise ValueError('Bad itree type {}'.format(itree_type))
        return itree_type_to_itree_map[itree_type](**kwargs)

    @staticmethod
    def get_random_path_length(branching_factor: int, max_leaf_samples: int, num_samples: float) -> float:
        if num_samples < max_leaf_samples:
            return 0
        else:
            return log(num_samples / max_leaf_samples) / log(2 * branching_factor)

    @staticmethod
    def get_multiplier(type: str, depth: int) -> int:
        # Compute the multiplier according to the type
        if type == 'fixed':
            return 1
        elif type == 'adaptive':
            return 2 ** depth
        else:
            raise ValueError('Bad type {}'.format(type))

    def __init__(self, max_leaf_samples: int, type: str, subsample: float, branching_factor: int, data_size: int,
                 random_seed: int = 1):
        self.max_leaf_samples: int = max_leaf_samples
        self.type: str = type
        self.subsample: float = subsample
        self.branching_factor: int = branching_factor
        self.data_size: int = data_size
        self.random_generator: Generator = default_rng(seed=random_seed)
        self.depth_limit: float = OnlineITree.get_random_path_length(self.branching_factor, self.max_leaf_samples,
                                                                     self.data_size * self.subsample)
        self.root: OnlineINode = None
        self.next_node_index: int = 0

    @abstractmethod
    def learn(self, data: ndarray) -> 'OnlineITree':
        pass

    @abstractmethod
    def recursive_learn(self, node: OnlineINode, data: ndarray, node_index: int) -> (int, OnlineINode):
        pass

    @abstractmethod
    def recursive_build(self, data: ndarray, depths: ndarray[float], node_index: int) -> (int, OnlineINode):
        pass

    @abstractmethod
    def unlearn(self, data: ndarray) -> 'OnlineITree':
        pass

    @abstractmethod
    def recursive_unlearn(self, node: OnlineINode, data: ndarray) -> OnlineINode:
        pass

    @abstractmethod
    def recursive_unbuild(self, node: OnlineINode) -> OnlineINode:
        pass

    @abstractmethod
    def predict(self, data: ndarray) -> ndarray[float]:
        pass

    @abstractmethod
    def recursive_depth_search(self, node: OnlineINode, data: ndarray, depths: ndarray[float]) -> ndarray[float]:
        pass

    @abstractmethod
    def split_data(self, data: ndarray, **kwargs) -> list[ndarray[int]]:
        pass
