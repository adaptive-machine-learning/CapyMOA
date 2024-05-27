from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)

from moa.classifiers.meta import LeveragingBag as _MOA_LeveragingBag

class LeveragingBagging(MOAClassifier):
    """Leveraging Bagging for evolving data streams using ADWIN. 
    
    Leveraging Bagging and Leveraging Bagging MC using Random Output Codes ( -o option).

    Reference:

    `Albert Bifet, Geoffrey Holmes, Bernhard Pfahringer.
    Leveraging Bagging for Evolving Data Streams Machine Learning and Knowledge
    Discovery in Databases, European Conference, ECML PKDD}, 2010.`

    See :py:class:`capymoa.base.MOAClassifier` for train, predict and predict_proba.

    """

    def __init__(
        self, schema=None, CLI=None, random_seed=1, base_learner=None, ensemble_size=100
    ):
        """Construct a Leveraging Bagging classifier.

        :param schema: The schema of the stream. If not provided, it will be inferred from the data.
        :param CLI: Command Line Interface (CLI) options for configuring the ARF algorithm.
            If not provided, default options will be used.
        :param random_seed: Seed for the random number generator.
        :param base_learner: The base learner to use. If not provided, a default Hoeffding Tree is used.
        :param ensemble_size: The number of trees in the ensemble.
        """
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
            schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=_MOA_LeveragingBag()
        )

    def __str__(self):
        # Overrides the default class name from MOA (LeveragingBag)
        return "Leveraging OnlineBagging"
