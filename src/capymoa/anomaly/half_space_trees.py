from capymoa.base import (
    MOAClassifier,
)

from moa.classifiers.oneclass import HSTrees as _MOA_HSTrees


class HalfSpaceTrees(MOAClassifier):
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

    """
    def __init__(
        self, schema=None, CLI=None, random_seed=1, window_size=100, number_of_trees=25, max_depth=15
    ):
        # This method basically configures the CLI, object creation is delegated to MOAClassifier (the super class, through super().__init___()))
        # Initialize instance attributes with default values, if the CLI was not set.
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
