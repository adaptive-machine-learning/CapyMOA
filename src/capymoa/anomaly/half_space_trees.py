from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)

from moa.classifiers.oneclass import HSTrees as _MOA_HSTrees


class HalfSpaceTrees(MOAClassifier):
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
