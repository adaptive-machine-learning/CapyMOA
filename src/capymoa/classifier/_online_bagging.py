from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)

from moa.classifiers.meta import OzaBag as _MOA_OzaBag

class OnlineBagging(MOAClassifier):
    def __init__(
        self, schema=None, CLI=None, random_seed=1, base_learner=None, ensemble_size=100
    ):
        # This method basically configures the CLI, object creation is delegated to MOAClassifier (the super class, through super().__init___()))
        # Initialize instance attributes with default values, if the CLI was not set.
        if CLI is None:
            self.base_learner = (
                "trees.HoeffdingTree"
                if base_learner is None
                else _extract_moa_learner_CLI(base_learner)
            )
            self.ensemble_size = ensemble_size
            CLI = f"-l {self.base_learner} -s {self.ensemble_size}"

        super().__init__(
            schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=_MOA_OzaBag()
        )

    def __str__(self):
        # Overrides the default class name from MOA (OzaBag)
        return "OnlineBagging"
