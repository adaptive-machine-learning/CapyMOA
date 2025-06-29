from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)
import os
from moa.classifiers.meta import OzaBag as _MOA_OzaBag
from moa.classifiers.meta.minibatch import OzaBagMB as _MOA_OzaBagMB


class OnlineBagging(MOAClassifier):
    """Incremental on-line bagging of Oza and Russell.

    Incremental on-line bagging of Oza and Russell [#0]_ is a ensemble classifier. Oza
    and Russell developed online versions of bagging and boosting for Data Streams. They
    show how the process of sampling bootstrap replicates from training data can be
    simulated in a data stream context. They observe that the probability that any
    individual example will be chosen for a replicate tends to a Poisson(1)
    distribution.

    >>> from capymoa.classifier import OnlineBagging
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = OnlineBagging(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    85.3

    .. [#0] `Oza, N. C., & Russell, S. J. (2001, January). Online bagging and boosting.
             In International workshop on artificial intelligence and statistics (pp.
             229-236). PMLR. <https://proceedings.mlr.press/r3/oza01a.html>`_
    """

    def __init__(
        self,
        schema=None,
        CLI=None,
        random_seed=1,
        base_learner=None,
        ensemble_size=100,
        minibatch_size=None,
        number_of_jobs=None,
    ):
        """Construct an Online bagging classifier using online bootstrap sampling.

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
        """
        if CLI is None:
            self.base_learner = (
                "trees.HoeffdingTree"
                if base_learner is None
                else _extract_moa_learner_CLI(base_learner)
            )
            self.ensemble_size = ensemble_size
            moa_learner = None
            if (
                number_of_jobs is None or number_of_jobs == 0 or number_of_jobs == 1
            ) and (
                minibatch_size is None or minibatch_size <= 0 or minibatch_size == 1
            ):
                # run the sequential version by default or when both parameters are None | 0 | 1
                self.number_of_jobs = 1
                self.minibatch_size = 1
                moa_learner = _MOA_OzaBag()
                CLI = f"-l {self.base_learner} -s {self.ensemble_size}"
            else:
                # run the minibatch parallel version when at least one of the number of jobs or the minibatch size parameters are greater than 1
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
                moa_learner = _MOA_OzaBagMB()
                CLI = f"-l {self.base_learner} -s {self.ensemble_size} -c {self.number_of_jobs} -b {self.minibatch_size} "
            # print(CLI)

        super().__init__(
            schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=moa_learner
        )

    def __str__(self):
        # Overrides the default class name from MOA (OzaBag)
        return "OnlineBagging"
