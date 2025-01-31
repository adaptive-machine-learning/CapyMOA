from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)

from moa.classifiers.meta import OzaBagAdwin as _MOA_OzaBagAdwin
from moa.classifiers.meta.minibatch import OzaBagAdwinMB as _MOA_OzaBagAdwinMB
import os


class OnlineAdwinBagging(MOAClassifier):
    """Bagging for evolving data streams using ADWIN.

    ADWIN is a change detector and estimator that solves in
    a well-speciﬁed way the problem of tracking the average of
    a stream of bits or real-valued numbers. ADWIN keeps a
    variable-length window of recently seen items, with the property
    that the window has the maximal length statistically consistent
    with the hypothesis “there has been no change in the average value
    inside the window”.<br />
    More precisely, an older fragment of the window is dropped if and only
    if there is enough evidence that its average value differs from that of
    the rest of the window. This has two consequences: one, that change
    reliably declared whenever the window shrinks; and two, that at any time
    the average over the existing window can be reliably taken as an estimation
    of the current average in the stream (barring a very small or very recent
    change that is still not statistically visible). A formal and quantitative
    statement of these two points (a theorem) appears in the reference paper.

    References:
    `Albert Bifet and Ricard Gavaldà. Learning from time-changing data with
    adaptive windowing. In SIAM International Conference on Data Mining, 2007.`
    `[BHPKG] Albert Bifet, Geoff Holmes, Bernhard Pfahringer, Richard Kirkby,
    and Ricard Gavaldà . New ensemble methods for evolving data streams.
    In 15th ACM SIGKDD International Conference on Knowledge Discovery and
    Data Mining, 2009.`

    ADWIN is parameter- and assumption-free in the sense that it automatically
    detects and adapts to the current rate of change. Its only parameter is a
    conﬁdence bound δ, indicating how conﬁdent we want to be in the algorithm’s
    output, inherent to all algorithms dealing with random processes. Also
    important, ADWIN does not maintain the window explicitly, but compresses it
    using a variant of the exponential histogram technique. This means that it
    keeps a window of length W using only O(log W) memory and O(log W) processing
    time per item.

    ADWIN Bagging is the online bagging method of Oza and Rusell with the
    addition of the ADWIN algorithm as a change detector and as an estimator for
    the weights of the boosting method. When a change is detected, the worst
    classiﬁer of the ensemble of classiﬁers is removed and a new classiﬁer is
    added to the ensemble.

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
        number_of_jobs=None,
    ):
        """Construct an Online bagging classifier using online bootstrap sampling with the addition of ADWIN drift detector.

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
                moa_learner = _MOA_OzaBagAdwin()
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
                moa_learner = _MOA_OzaBagAdwinMB()
                CLI = f"-l {self.base_learner} -s {self.ensemble_size} -c {self.number_of_jobs} -b {self.minibatch_size} "

        super().__init__(
            schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=moa_learner
        )

    def __str__(self):
        # Overrides the default class name from MOA (OzaBagAdwin)
        return "OnlineBagging with ADWIN"
