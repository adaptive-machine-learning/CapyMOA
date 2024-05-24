from capymoa.base import (
    MOAAnomalyDetector,
)

from moa.classifiers.oneclass import HSTrees as _MOA_HSTrees


class HalfSpaceTrees(MOAAnomalyDetector):
    """ Half-Space Trees

    This class implements the Half-Space Trees (HS-Trees) algorithm, which is
    an ensemble anomaly detector capable of adapting to concept drift.

    HS-Trees is implemented in MOA (Massive Online Analysis) and provides several
    parameters for customization.

    Reference:
    Tan, S. C., Ting, K. M., & Liu, T. F. (2011, June). Fast anomaly detection for streaming data.
    In Twenty-second international joint conference on artificial intelligence.
    <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=73b6b7d9e7e225719ad86234927a3b60a4a873c0>

    Example usage:
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import HalfSpaceTrees
    >>> from capymoa.evaluation import AUCEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = HalfSpaceTrees(schema)
    >>> evaluator = AUCEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.54
    """
    def __init__(
        self, schema=None, CLI=None, random_seed=1, window_size=100, number_of_trees=25, max_depth=15
    ):
        """Construct a Half-Space Trees anomaly detector

        :param schema: The schema of the stream. If not provided, it will be inferred from the data.
        :param CLI: Command Line Interface (CLI) options for configuring the HS-Trees algorithm.
        :param random_seed: Random seed for reproducibility.
        :param window_size: The size of the window for each tree.
        :param number_of_trees: The number of trees in the ensemble.
        :param max_depth: The maximum depth of each tree.
        """
        if CLI is None:
            self.window_size = window_size
            self.number_of_trees = number_of_trees
            self.max_depth = max_depth
            CLI = f"-p {self.window_size} -t {self.number_of_trees} -h {self.max_depth}"

        super().__init__(
            schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=_MOA_HSTrees()
        )

    def __str__(self):
        # Overrides the default class name from MOA
        return "HalfSpaceTrees"
