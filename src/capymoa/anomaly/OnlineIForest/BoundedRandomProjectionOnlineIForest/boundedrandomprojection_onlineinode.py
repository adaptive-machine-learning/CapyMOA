from ..onlineinode import OnlineINode
from numpy import ndarray


class BoundedRandomProjectionOnlineINode(OnlineINode):
    def __init__(self, data_size: int, children: ndarray[OnlineINode], depth: int, node_index: int,
                 min_values: ndarray, max_values: ndarray, projection_vector: ndarray[float],
                 split_values: ndarray[float]):
        super().__init__(data_size, children, depth, node_index)
        self.min_values: ndarray = min_values
        self.max_values: ndarray = max_values
        self.projection_vector: ndarray[float] = projection_vector
        self.split_values: ndarray[float] = split_values
