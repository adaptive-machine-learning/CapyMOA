from moa.classifiers.lazy import kNN as _moa_kNN

from capymoa.base import MOARegressor
from capymoa.stream import Schema


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
        schema: Schema,
        k: int = 3,
        median: bool = False,
        window_size: int = 1000,
        random_seed: int | None = None,
    ) -> None:
        """Construct a KNNRegressor.

        :param schema: Describes the stream's structure and types.
        :param k: the number of the neighbours.
        :param median: choose to use mean or median as the aggregation for the final prediction.
        :param window_size: the size of the sliding window to store the instances.
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
