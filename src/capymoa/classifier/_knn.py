from moa.classifiers.lazy import kNN as _moa_kNN

from capymoa.base import MOAClassifier
from capymoa.stream import Schema


class KNN(MOAClassifier):
    """K-Nearest Neighbors.

    K-Nearest Neighbors (KNN) [#f1]_ is a lazy classifier. KNN in the streaming
    setting [#f2]_ stores a window of the most recent instances and uses them to
    classify new instances based on the majority class among the k-nearest
    neighbors.

    >>> from capymoa.classifier import KNN
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = KNN(stream.get_schema())
    >>> results = evaluate_classifier(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
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

    def __init__(
        self,
        schema: Schema,
        k: int = 3,
        window_size: int = 1000,
        random_seed: int | None = None,
    ) -> None:
        """Construct a KNN classifier.

        :param schema: Describes the stream's structure and types.
        :param k: The number of neighbors.
        :param window_size: The maximum number of instances to store.
        :param random_seed: Seed for all random choices. ``None`` is a fresh random seed each run.
        """
        super().__init__(
            moa_learner=_moa_kNN,
            schema=schema,
            random_seed=random_seed,
            options={"k": k, "limit": window_size},
        )

    def __str__(self):
        # Overrides the default class name from MOA
        return "kNN"
