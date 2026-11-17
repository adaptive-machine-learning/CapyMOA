from moa.classifiers.meta.imbalanced import CSMOTE as _MOA_CSMOTE

from capymoa.base import Classifier, MOAClassifier
from capymoa.stream import Schema


class CSMOTE(MOAClassifier):
    """Continuous Synthetic Minority Oversampling Technique.

    Continuous Synthetic Minority Oversampling Technique (C-SMOTE) [#0]_ is a
    meta-strategy. This strategy saves all the minority samples in a window managed by
    ADWIN. Meanwhile, a model is trained with the input data. When the minority sample
    ratio falls below a certain threshold, an online version of SMOTE is applied. A
    random minority sample is chosen from the window, and a new synthetic sample is
    generated until the minority sample ratio is greater than or equal to the threshold.
    The model is then trained with the newly generated samples.

    >>> from capymoa.classifier import CSMOTE
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = CSMOTE(stream.get_schema(), random_seed=0)
    >>> results = evaluate_classifier(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    83.1

    .. [#0] `Alessio Bernardo, Heitor Murilo Gomes, Jacob Montiel, Bernhard Pfahringer,
             Albert Bifet, Emanuele Della Valle. C-SMOTE: Continuous Synthetic Minority
             Oversampling for Evolving Data Streams. In BigData, IEEE, 2020.
             <https://ieeexplore.ieee.org/document/9377768>`_
    """

    def __init__(
        self,
        schema: Schema = None,
        random_seed: int | None = None,
        base_learner: str | Classifier = "trees.HoeffdingTree",
        neighbors: int = 10,
        threshold: float = 0.5,
        min_size_allowed: int = 100,
        disable_drift_detection: bool = False,
    ):
        """Construct C-SMOTE.

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param base_learner: The base learner to be trained. Default trees.HoeffdingTree.
        :param neighbors: Number of neighbors for SMOTE.
        :param threshold: Minority class samples threshold.
        :param min_size_allowed: Minimum number of samples in the minority class for applying SMOTE.
        :param disable_drift_detection: If set, disables ADWIN drift detector
        """

        options = {
            "baseLearner": base_learner,
            "neighbors": neighbors,
            "threshold": threshold,
            "minSizeAllowed": min_size_allowed,
            "disableDriftDetection": disable_drift_detection,
        }
        super().__init__(
            moa_learner=_MOA_CSMOTE,
            schema=schema,
            random_seed=random_seed,
            options=options,
        )
