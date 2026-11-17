from moa.classifiers.lazy import kNN as _moa_kNN

from capymoa.base import MOARegressor


class KNNRegressor(MOARegressor):
    """K Nearest Neighbor for data stream regression with sliding window

    The default number of neighbors (k) is set to 3 instead of 10 (as in MOA)

    There is no specific publication for online KNN, please refer to:

    `Bifet, Albert, Ricard Gavalda, Geoffrey Holmes, and Bernhard Pfahringer.
    Machine learning for data streams: with practical examples in MOA. MIT press, 2023.
    <https://moa.cms.waikato.ac.nz/book-html/>`_

    Example usage:

    >>> from capymoa.datasets import Fried
        >>> from capymoa.regressor import KNNRegressor
        >>> from capymoa.regressor import evaluate_regressor
    >>> stream = Fried()
    >>> schema = stream.get_schema()
    >>> learner = KNNRegressor(schema)
    >>> results = evaluate_regressor(stream, learner, max_instances=1000)
    >>> results["cumulative"]["rmse"]
    2.9811398077838542
    """

    def __init__(
        self,
        schema=None,
        k=3,
        median=False,
        window_size=1000,
        random_seed: int | None = None,
    ):
        """Construct a KNNRegressor.

        :param schema: Describes the stream's structure and types.
        :param k: The number of neighbors.
        :param median: Whether to use the median instead of the mean to combine the
            neighbors' targets.
        :param window_size: The maximum number of instances to store.
        :param random_seed: Seed for all random choices. ``None`` is a fresh random seed each run.
        """

        # Important, should create the MOA object before invoking the super class __init__
        self.moa_learner = _moa_kNN()
        super().__init__(
            schema=schema,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )

        self.k = k
        self.median = median
        self.window_size = window_size
        self.moa_learner.getOptions().setViaCLIString(
            f"-k {self.k} {'-m' if self.median else ''} -w \
         {self.window_size}"
        )
        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearning()

    def __str__(self):
        # Overrides the default class name from MOA
        return "kNNRegressor"
