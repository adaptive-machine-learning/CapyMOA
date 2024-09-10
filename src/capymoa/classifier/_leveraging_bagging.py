from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)

from moa.classifiers.meta import LeveragingBag as _MOA_LeveragingBag
from moa.classifiers.meta.minibatch import LeveragingBagMB as _MOA_LeveragingBagMB
import os

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
        self, 
        schema=None, 
        CLI=None, 
        random_seed=1, 
        base_learner=None, 
        ensemble_size=100,
        minibatch_size=None,
        number_of_jobs=None
    ):
        """Construct a Leveraging Bagging classifier.

        :param schema: The schema of the stream. If not provided, it will be inferred from the data.
        :param CLI: Command Line Interface (CLI) options for configuring the ARF algorithm.
            If not provided, default options will be used.
        :param random_seed: Seed for the random number generator.
        :param base_learner: The base learner to use. If not provided, a default Hoeffding Tree is used.
        :param ensemble_size: The number of trees in the ensemble.
        :param minibatch_size: The number of instances that a learner must accumulate before training.
        :param number_of_jobs: The number of parallel jobs to run during the execution of the algorithm.
            By default, the algorithm executes tasks sequentially (i.e., with `number_of_jobs=1`).
            Increasing the `number_of_jobs` can lead to faster execution on multi-core systems.
            However, setting it to a high value may consume more system resources and memory.
            This implementation focuses more on performance, therefore the predictive performance is modified.
            It's recommended to experiment with different values to find the optimal setting based on
            the available hardware resources and the nature of the workload.
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

            self.base_learner = (
                "trees.HoeffdingTree"
                if base_learner is None
                else _extract_moa_learner_CLI(base_learner)
            )
            self.ensemble_size = ensemble_size
            moa_learner = None
            if (number_of_jobs is None or number_of_jobs == 0 or number_of_jobs == 1) and (minibatch_size is None or minibatch_size <= 0 or minibatch_size == 1):
                #run the sequential version by default or when both parameters are None | 0 | 1
                self.number_of_jobs = 1
                self.minibatch_size = 1
                moa_learner = _MOA_LeveragingBag()
                CLI = f"-l {self.base_learner} -s {self.ensemble_size}"
            else:
                #run the minibatch parallel version when at least one of the number of jobs or the minibatch size parameters are greater than 1
                if number_of_jobs == 0 or number_of_jobs is None:
                    self.number_of_jobs = 1
                elif number_of_jobs < 0:
                    self.number_of_jobs = os.cpu_count()
                else:
                    self.number_of_jobs = int(min(number_of_jobs, os.cpu_count()))
                if minibatch_size <= 1:
                    # if the user sets the number of jobs and the minibatch_size less than 1 it is considered that the user wants a parallel execution of a single instance at a time
                    self.minibatch_size = 1
                elif minibatch_size is None:
                    # if the user sets only the number_of_jobs, we assume he wants the parallel minibatch version and initialize minibatch_size to the default 25
                    self.minibatch_size = 25
                else:
                    # if the user sets both parameters to values greater than 1, we initialize the minibatch_size to the user's choice
                    self.minibatch_size = int(minibatch_size)
                moa_learner = _MOA_LeveragingBagMB()
                CLI = f"-l {self.base_learner} -s {self.ensemble_size} -c {self.number_of_jobs} -b {self.minibatch_size} "


        super().__init__(
            schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=moa_learner
        )

    def __str__(self):
        # Overrides the default class name from MOA (LeveragingBag)
        return "Leveraging OnlineBagging"
