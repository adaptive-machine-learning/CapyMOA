from ..onlineitree import OnlineITree
from ..onlineinode import OnlineINode
from numpy import argsort, empty, inf, ndarray, sort, split, vstack, zeros
from numpy.linalg import norm


class BoundedRandomProjectionOnlineITree(OnlineITree):
    def __init__(self, max_leaf_samples: int, type: str, subsample: float, branching_factor: int, data_size: int,
                 metric: str = 'axisparallel', random_seed: int = 1):
        super().__init__(max_leaf_samples, type, subsample, branching_factor, data_size, random_seed)
        self.metric: str = metric

    def learn(self, data: ndarray) -> 'BoundedRandomProjectionOnlineITree':
        # Subsample data in order to improve diversity among trees
        data: ndarray = data[self.random_generator.random(data.shape[0]) < self.subsample]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size += data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit: float = OnlineITree.get_random_path_length(self.branching_factor, self.max_leaf_samples,
                                                                         self.data_size)
            # Recursively update the tree
            self.next_node_index, self.root = self.recursive_build(data) if self.root is None else self.recursive_learn(self.root, data, self.next_node_index)
        return self

    def recursive_learn(self, node: OnlineINode, data: ndarray, node_index: int) -> (int, OnlineINode):
        # Update the number of data seen so far by the current node
        node.data_size += data.shape[0]
        # Update the vectors of minimum and maximum values seen so far by the current node
        node.min_values: ndarray = vstack([data, node.min_values]).min(axis=0)
        node.max_values: ndarray = vstack([data, node.max_values]).max(axis=0)
        # If the current node is a leaf, try to split it
        if node.children is None:
            # If there are enough samples to be split according to the max leaf samples and the depth limit has not been
            # reached yet, split the node
            if node.data_size >= self.max_leaf_samples*OnlineITree.get_multiplier(self.type, node.depth) and node.depth < self.depth_limit:
                # Sample data_size points uniformly at random within the bounding box defined by the vectors of minimum
                # and maximum values of data seen so far by the current node
                data_sampled: ndarray = self.random_generator.uniform(node.min_values, node.max_values, size=(node.data_size, data.shape[1]))
                return self.recursive_build(data_sampled, depth=node.depth, node_index=node_index)
            else:
                return node_index, node
        # If the current node is not a leaf, recursively update all its children
        else:
            # Partition data
            partition_indices: list[ndarray[int]] = self.split_data(data, node.projection_vector, node.split_values)
            # Recursively update children
            for i, indices in enumerate(partition_indices):
                node_index, node.children[i] = self.recursive_learn(node.children[i], data[indices], node_index)
            return node_index, node

    def recursive_build(self, data: ndarray, depth: int = 0, node_index: int = 0) -> (int, OnlineINode):
        # If there aren't enough samples to be split according to the max leaf samples or the depth limit has been
        # reached, build a leaf node
        if data.shape[0] < self.max_leaf_samples*OnlineITree.get_multiplier(self.type, depth) or depth >= self.depth_limit:
            return node_index + 1, OnlineINode.create('boundedrandomprojectiononlineinode',
                                                      data_size=data.shape[0], children=None, depth=depth,
                                                      node_index=node_index, min_values=data.min(axis=0, initial=inf),
                                                      max_values=data.max(axis=0, initial=-inf), projection_vector=None,
                                                      split_values=None)
        else:
            # Sample projection vector
            if self.metric == 'axisparallel':
                projection_vector: ndarray[float] = zeros(data.shape[1])
                projection_vector[self.random_generator.choice(projection_vector.shape[0])]: float = 1.0
            else:
                raise ValueError('Bad metric {}'.format(self.metric))
            projection_vector: ndarray[float] = projection_vector / norm(projection_vector)
            # Project sampled data using projection vector
            projected_data: ndarray = data @ projection_vector
            # Sample split values
            split_values: ndarray[float] = sort(self.random_generator.uniform(min(projected_data), max(projected_data),
                                                                              size=self.branching_factor - 1))
            # Partition sampled data
            partition_indices: list[ndarray[int]] = self.split_data(data, projection_vector, split_values)
            # Generate recursively children nodes
            children: ndarray[OnlineINode] = empty(shape=(self.branching_factor,), dtype=OnlineINode)
            for i, indices in enumerate(partition_indices):
                node_index, children[i] = self.recursive_build(data[indices], depth + 1, node_index)
            return node_index + 1, OnlineINode.create('boundedrandomprojectiononlineinode',
                                                      data_size=data.shape[0], children=children, depth=depth,
                                                      node_index=node_index, min_values=data.min(axis=0),
                                                      max_values=data.max(axis=0), projection_vector=projection_vector,
                                                      split_values=split_values)

    def unlearn(self, data: ndarray) -> 'BoundedRandomProjectionOnlineITree':
        # Subsample data in order to improve diversity among trees
        data: ndarray = data[self.random_generator.random(data.shape[0]) < self.subsample]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size -= data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit: float = OnlineITree.get_random_path_length(self.branching_factor, self.max_leaf_samples,
                                                                         self.data_size)
            # Recursively update the tree
            self.root: OnlineINode = self.recursive_unlearn(self.root, data)
        return self

    def recursive_unlearn(self, node: OnlineINode, data: ndarray) -> OnlineINode:
        # Update the number of data seen so far by the current node
        node.data_size -= data.shape[0]
        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, try to unsplit it
        else:
            # If there are not enough samples according to max leaf samples, unsplit the node
            if node.data_size < self.max_leaf_samples*OnlineITree.get_multiplier(self.type, node.depth):
                return self.recursive_unbuild(node)
            # If there are enough samples according to max leaf samples, recursively update all its children
            else:
                # Partition data
                partition_indices: list[ndarray[int]] = self.split_data(data, node.projection_vector, node.split_values)
                # Recursively update children
                for i, indices in enumerate(partition_indices):
                    node.children[i]: OnlineINode = self.recursive_unlearn(node.children[i], data[indices])
                # Update the vectors of minimum and maximum values seen so far by the current node
                node.min_values: ndarray = vstack([node.children[i].min_values for i, _ in enumerate(node.children)]).min(axis=0)
                node.max_values: ndarray = vstack([node.children[i].max_values for i, _ in enumerate(node.children)]).max(axis=0)
                return node

    def recursive_unbuild(self, node: OnlineINode) -> OnlineINode:
        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, unbuild it
        else:
            # Recursively unbuild children
            for i, _ in enumerate(node.children):
                node.children[i]: OnlineINode = self.recursive_unbuild(node.children[i])
            # Update the vectors of minimum and maximum values seen so far by the current node
            node.min_values: ndarray = vstack([node.children[i].min_values for i, _ in enumerate(node.children)]).min(axis=0)
            node.max_values: ndarray = vstack([node.children[i].max_values for i, _ in enumerate(node.children)]).max(axis=0)
            # Delete children nodes, projection vector and split values
            node.children: ndarray[OnlineINode] = None
            node.projection_vector: ndarray[float] = None
            node.split_values: ndarray[float] = None
            return node

    def predict(self, data: ndarray) -> ndarray[float]:
        # Compute depth of each sample
        if self.root:
            return self.recursive_depth_search(self.root, data, empty(shape=(data.shape[0],), dtype=float))
        else:
            return zeros(shape=(data.shape[0],), dtype=float)

    def recursive_depth_search(self, node: OnlineINode, data: ndarray, depths: ndarray[float]) -> ndarray[float]:
        # If the current node is a leaf, fill the depths vector with the current depth plus a normalization factor
        if node.children is None or data.shape[0] == 0:
            depths[:] = node.depth + OnlineITree.get_random_path_length(self.branching_factor, self.max_leaf_samples,
                                                                        node.data_size)
        else:
            # Partition data
            partition_indices: list[ndarray[int]] = self.split_data(data, node.projection_vector, node.split_values)
            # Fill the vector of depths
            for i, indices in enumerate(partition_indices):
                depths[indices]: ndarray[float] = self.recursive_depth_search(node.children[i], data[indices],
                                                                              depths[indices])
        return depths

    def split_data(self, data: ndarray, projection_vector: ndarray[float], split_values: ndarray[float]) -> list[ndarray[int]]:
        # Project data using projection vector
        projected_data: ndarray = data @ projection_vector
        # Sort projected data and keep sort indices
        sort_indices: ndarray = argsort(projected_data)
        # Split data according to their membership
        partition: list[ndarray[int]] = split(sort_indices, projected_data[sort_indices].searchsorted(split_values))
        return partition
