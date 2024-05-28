from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)

from moa.classifiers.meta import OzaBagAdwin as _MOA_OzaBagAdwin

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
        self, schema=None, CLI=None, random_seed=1, base_learner=None, ensemble_size=100
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

        super().__init__(
            schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=_MOA_OzaBagAdwin()
        )

    def __str__(self):
        # Overrides the default class name from MOA (OzaBagAdwin)
        return "OnlineBagging with ADWIN"
