from abc import ABC
from numpy import ndarray


class OnlineINode(ABC):
    @staticmethod
    def create(inode_type: str, **kwargs) -> 'OnlineINode':
        # TODO: Find an alternative solution to overcome circular imports
        from .BoundedRandomProjectionOnlineIForest import BoundedRandomProjectionOnlineINode
        # Map inode type to an inode class
        inode_type_to_inode_map: dict = {'boundedrandomprojectiononlineinode': BoundedRandomProjectionOnlineINode}
        if inode_type not in inode_type_to_inode_map:
            raise ValueError('Bad inode type {}'.format(inode_type))
        return inode_type_to_inode_map[inode_type](**kwargs)

    def __init__(self, data_size: int, children: ndarray['OnlineINode'], depth: int, node_index: int):
        self.data_size: int = data_size
        self.children: ndarray['OnlineINode'] = children
        self.depth: int = depth
        self.node_index: int = node_index
