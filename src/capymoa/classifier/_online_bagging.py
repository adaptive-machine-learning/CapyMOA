from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)

from moa.classifiers.meta import OzaBag as _MOA_OzaBag

class OnlineBagging(MOAClassifier):
    """Incremental on-line bagging of Oza and Russell.

    Oza and Russell developed online versions of bagging and boosting for
    Data Streams. They show how the process of sampling bootstrap replicates
    from training data can be simulated in a data stream context. They observe
    that the probability that any individual example will be chosen for a
    replicate tends to a Poisson(1) distribution.
    This class implements the Adaptive Random Forest (ARF) algorithm, which is
    an ensemble classifier capable of adapting to concept drift.

    ARF is implemented in MOA (Massive Online Analysis) and provides several
    parameters for customization.

    Reference:

    `N. Oza and S. Russell. Online bagging and boosting.
    In Artiﬁcial Intelligence and Statistics 2001, pages 105–112.
    Morgan Kaufmann, 2001.`

    See :py:class:`capymoa.base.MOAClassifier` for train, predict and predict_proba.

    """
    
    def __init__(
        self, schema=None, CLI=None, random_seed=1, base_learner=None, ensemble_size=100
    ):
        """Construct an Online bagging classifier using online bootstrap sampling.

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
            schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=_MOA_OzaBag()
        )

    def __str__(self):
        # Overrides the default class name from MOA (OzaBag)
        return "OnlineBagging"
