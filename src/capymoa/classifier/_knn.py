from capymoa.base import MOAClassifier
from moa.classifiers.lazy import kNN as _moa_kNN


class KNN(MOAClassifier):
    """K-Nearest Neighbors.

    K-Nearest Neighbors (KNN) [#f1]_ is a lazy classifier. KNN in the streaming
    setting [#f2]_ stores a window of the most recent instances and uses them to
    classify new instances based on the majority class among the k-nearest
    neighbors.

    >>> from capymoa.classifier import KNN
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = KNN(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    80.7

    ..  seealso::

        :class:`~capymoa.classifier.SAMkNN`
            Self Adjusted Memory k-Nearest Neighbor (SAMkNN) classifier.

    .. [#f1] Fix, E., & Hodges, J. L. (1989). Discriminatory Analysis.
            Nonparametric Discrimination: Consistency Properties. International
            Statistical Review / Revue Internationale de Statistique, 57(3),
            238–247. https://doi.org/10.2307/1403797

    .. [#f2] Jesse Read, Albert Bifet, Bernhard Pfahringer, and Geoff Holmes.
            Batch-incremental versus instance-incremental learning in dynamic
            and evolving data. In Advances in Intelligent Data Analysis XI -
            11th International Symposium, IDA 2012, Helsinki, Finland, October
            25–27, 2012. Proceedings , pages 313–323, 2012.
    """

    def __init__(self, schema=None, CLI=None, random_seed=1, k=3, window_size=1000):
        # Important, should create the MOA object before invoking the super class __init__
        self.moa_learner = _moa_kNN()
        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )

        # Initialize instance attributes with default values, CLI was not set.
        if self.CLI is None:
            self.k = k
            self.window_size = window_size
            self.moa_learner.getOptions().setViaCLIString(
                f"-k {self.k} -w {self.window_size}"
            )
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()

    def __str__(self):
        # Overrides the default class name from MOA
        return "kNN"
