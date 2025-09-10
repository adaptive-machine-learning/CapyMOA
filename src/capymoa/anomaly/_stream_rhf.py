import numpy as np
from capymoa.base import AnomalyDetector


# Random Histogram Tree Node
class Node:
    def __init__(self, data, height, max_height, seed, node_id):
        self.data = data  # Data at this node
        self.height = height  # Current depth of the tree
        self.max_height = max_height  # Maximum allowed height of the tree
        self.seed = seed  # Random seed for attribute selection
        self.attribute = None  # Split attribute
        self.value = None  # Split value
        self.left = None  # Left child
        self.right = None  # Right child
        self.node_id = node_id  # Unique node identifier

    def is_leaf(self):
        return self.left is None and self.right is None


def collect_subtree_data(node, number_of_features):
    if node is None:
        # Return empty array for null nodes
        return np.empty((0, number_of_features))

    # Collect data from the current node and its children
    collected_data = (
        node.data if node.data is not None else np.empty((0, number_of_features))
    )
    if not node.is_leaf():
        left_data = collect_subtree_data(node.left, number_of_features)
        right_data = collect_subtree_data(node.right, number_of_features)
        collected_data = np.vstack((collected_data, left_data, right_data))
    return collected_data


def compute_kurtosis(data):
    if len(data) == 0:
        return np.zeros(data.shape[1])  # Return zero kurtosis for empty data
    data = np.asarray(data)
    kurtosis_values = np.zeros(data.shape[1])
    for feature_idx in range(data.shape[1]):
        feature_data = data[:, feature_idx]
        mean = np.mean(feature_data)
        variance = np.mean((feature_data - mean) ** 2)
        fourth_moment = np.mean((feature_data - mean) ** 4)
        kurtosis_value = fourth_moment / ((variance + 1e-10) ** 2)
        kurtosis_values[feature_idx] = np.log(kurtosis_value + 1)
    return kurtosis_values


def choose_split_attribute(kurt_values, random_seed):
    np.random.seed(int(random_seed))  # Ensure seed is an integer
    Ks = np.sum(kurt_values)
    r = np.random.uniform(0, Ks)
    cumulative = 0
    for idx, k_value in enumerate(kurt_values):
        cumulative += k_value
        if cumulative > r:
            return idx
    return len(kurt_values) - 1


def RHT_build(data, height, max_height, seed_array, node_id=1):
    node = Node(None, height, max_height, seed_array[node_id], node_id)
    if height == max_height or len(data) <= 1:
        node.data = data  # Only store data at leaf nodes
        return node

    kurt_values = compute_kurtosis(data)
    attribute = choose_split_attribute(kurt_values, node.seed)
    split_value = np.random.uniform(
        np.min(data[:, attribute]), np.max(data[:, attribute])
    )

    node.attribute = attribute
    node.value = split_value

    left_data = data[data[:, attribute] <= split_value]
    right_data = data[data[:, attribute] > split_value]

    node.left = RHT_build(
        left_data, height + 1, max_height, seed_array, node_id=2 * node_id
    )
    node.right = RHT_build(
        right_data, height + 1, max_height, seed_array, node_id=(2 * node_id) + 1
    )

    return node


# I think we have to pay more attention to the node_id's here
def insert(node, instance, max_height, seed_array):
    if not node.is_leaf():
        # Handle the case where node.data is None
        data_to_send = (
            np.vstack((node.data, instance))
            if node.data is not None
            else np.array([instance])
        )
        kurt_values = compute_kurtosis(data_to_send)
        new_attribute = choose_split_attribute(
            kurt_values, seed_array[node.node_id]
        )  # Use the correct seed

        if node.attribute != new_attribute:
            subtree_data = collect_subtree_data(node, data_to_send.shape[1])
            return RHT_build(
                np.vstack((subtree_data, instance)),
                node.height,
                max_height,
                seed_array,
                node_id=node.node_id,
            )

        if instance[node.attribute] <= node.value:
            node.left = insert(node.left, instance, max_height, seed_array)
        else:
            node.right = insert(node.right, instance, max_height, seed_array)
    else:
        # Handle the case where node.data is None
        data_to_send = (
            np.vstack((node.data, instance))
            if node.data is not None
            else np.array([instance])
        )
        if node.height == max_height:
            node.data = data_to_send
        else:
            # Since the max height has not been reached, we can continue to build the tree
            # return RHT_build(np.vstack((node.data, instance)), node.height, max_height, seed_array, node_id=1)
            return RHT_build(
                data_to_send, node.height, max_height, seed_array, node_id=node.node_id
            )
    return node


def score_instance(tree, instance, total_instances):
    node = tree
    while not node.is_leaf():
        if instance[node.attribute] <= node.value:
            node = node.left
        else:
            node = node.right

    leaf_size = len(node.data)
    # Handle division by zero
    if (total_instances == 0) or (leaf_size == 0):
        # float('inf')  # Still unsure if to assign the maximum or minimum anomaly score
        return 1
    if total_instances == leaf_size:
        return 0
    P_Q = leaf_size / total_instances
    # Compute the raw anomaly score
    raw_anomaly_score = np.log(1 / (P_Q + 1e-10))

    # Normalize the anomaly score to the range [0, 1]
    min_score = np.log(total_instances / (total_instances + (1e-10) * total_instances))
    max_score = np.log(total_instances / (1 + (1e-10) * total_instances))
    normalized_score = (raw_anomaly_score - min_score) / (
        max_score - min_score + 1e-10
    )  # Avoid division by zero

    return normalized_score


def print_tree_info(node):
    if node is None:
        return

    # Print information about the current node
    print(
        f"Node ID: {node.node_id}, Height: {node.height}, Data Shape: {None if node.data is None else node.data.shape}"
    )

    # Recursively print information for left and right children
    print_tree_info(node.left)
    print_tree_info(node.right)


class RandomHistogramForest:
    def __init__(self, num_trees, max_height, window_size, number_of_features):
        self.num_trees = num_trees
        self.max_height = max_height
        self.window_size = window_size
        self.forest = []
        self.seed_arrays = []
        self.reference_window = []
        self.current_window = []
        self.number_of_features = number_of_features

    def initialize_forest(self, seed=None):
        rng = np.random.default_rng(seed)

        self.forest = []
        # Maximum possible nodes in a full binary tree
        num_nodes = 2 ** (self.max_height + 1)
        # Each tree gets a seed array for all possible nodes
        self.seed_arrays = [
            rng.integers(0, 10000, size=num_nodes) for _ in range(self.num_trees)
        ]

        # this will just create the trees with empty data, just the root i'd say
        for i in range(self.num_trees):
            tree = RHT_build(
                np.empty((0, self.number_of_features)),
                0,
                self.max_height,
                self.seed_arrays[i],
                node_id=1,
            )
            self.forest.append(tree)

    def update_forest(self, instance):
        self.current_window.append(instance)

        if len(self.current_window) >= self.window_size:
            self.reference_window = self.current_window[-self.window_size :]
            self.current_window = []
            self.forest = []
            for i in range(self.num_trees):
                tree = RHT_build(
                    np.array(self.reference_window),
                    0,
                    self.max_height,
                    self.seed_arrays[i],
                    node_id=1,
                )
                self.forest.append(tree)

        for i, tree in enumerate(self.forest):
            self.forest[i] = insert(
                tree, instance, self.max_height, self.seed_arrays[i]
            )

    def score(self, instance):
        # Gather all unique instances from the first tree
        # total_instances = len(self.current_window)+len(self.reference_window)

        total_instances = self.window_size * 2
        # print('total instances ' + str(total_instances))

        # Compute the normalized anomaly score
        return np.mean(
            [score_instance(tree, instance, total_instances) for tree in self.forest]
        )

    def print_forest_info(self):
        for i, tree in enumerate(self.forest):
            print(f"Tree {i + 1}:")
            print_tree_info(tree)
            print("-" * 50)


class StreamRHF(AnomalyDetector):
    """StreamRHF anomaly detector

    StreamRHF: Streaming Random Histogram Forest for Anomaly Detection

    StreamRHF is an unsupervised anomaly detection algorithm tailored for
    real-time data streams. Building upon the principles of Random Histogram
    Forests (RHF), this algorithm extends its capabilities to handle dynamic
    data streams efficiently. StreamRHF combines the power of tree-based
    partitioning with kurtosis-driven feature selection to detect anomalies
    in a resource-constrained streaming environment.

    Reference:

    `STREAMRHF: Tree-Based Unsupervised Anomaly Detection for Data Streams.
    Stefan Nesic, Andrian Putina, Maroua Bahri, Alexis Huet, Jose Manuel Navarro, Dario Rossi, Mauro Sozio.
    <https://nonsns.github.io/paper/rossi22aiccsa.pdf>`_

    Example:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import StreamRHF
    >>> from capymoa.evaluation import AnomalyDetectionEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = StreamRHF(schema=schema, num_trees=5, max_height=3)
    >>> evaluator = AnomalyDetectionEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.73

    """

    def __init__(
        self, schema, max_height=5, num_trees=100, window_size=20, random_seed=0
    ):
        """
        Initialize the StreamRHF learner.
        :param schema: Schema of the data stream.
        :param max_height: Maximum height of the trees.
        :param num_trees: Number of trees in the forest.
        :param window_size: Size of the sliding window.
        :param random_seed: Random seed for reproducibility.
        """
        super().__init__(schema, random_seed)
        self.schema = schema
        self.max_height = max_height
        self.num_trees = num_trees
        self.window_size = window_size
        self.forest = RandomHistogramForest(
            num_trees, max_height, window_size, schema.get_num_attributes()
        )
        self.forest.initialize_forest(random_seed)

    def score_instance(self, instance) -> float:
        """
        Score a single instance.
        A score value close to 1 means that is an anomaly and close to 0 it means it is a normal instance
        :param instance: An instance from the stream.
        :return: Anomaly score for the instance.
        """
        # In the case that the score 1 means that is an anomaly and 0 if normal instance
        return float(self.forest.score(instance.x))

    def train(self, instance):
        """
        Train the learner with a single instance.
        :param instance: An instance from the stream.
        """
        # instance_array = instance.x.reshape(1, -1)
        self.forest.update_forest(instance.x)

    def predict(self, instance):
        """
        Predict anomaly score for a single instance.
        This method uses the anomaly score of the instance to classify it
        as normal (0) or anomalous (1) based on a threshold.
        :param instance: An instance from the stream.
        :return: 0 if the instance is classified as normal, 1 if classified as anomalous.
        """
        if self.score_instance(instance) > 0.5:
            return 0
        else:
            return 1

    def __str__(self):
        return "StreamRHF Anomaly Detector"
