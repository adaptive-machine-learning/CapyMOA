from moa.classifiers.meta import OzaBag as _MOA_OzaBag

from capymoa.base import MOARegressor
from capymoa.stream import Schema


class OnlineBaggingRegressor(MOARegressor):
    """Incremental on-line bagging of Oza and Russell for regression.

    Online bagging [#0]_ simulates bootstrap sampling on a data stream by training
    each base learner on every instance with a weight drawn from a Poisson(1)
    distribution. For regression, the prediction is the mean of the base learners'
    predictions. The default base learner is :class:`~capymoa.regressor.FIMTDD`.

    See also :class:`capymoa.classifier.OnlineBagging`.

    >>> from capymoa.datasets import Fried
    >>> from capymoa.regressor import OnlineBaggingRegressor
    >>> from capymoa.regressor import evaluate_regressor
    >>>
    >>> stream = Fried()
    >>> learner = OnlineBaggingRegressor(stream.get_schema())
    >>> results = evaluate_regressor(stream, learner, max_instances=1000)
    >>> print(f"{results['cumulative']['rmse']:.2f}")
    7.10

    .. [#0] `Oza, N. C., & Russell, S. J. (2001, January). Online bagging and boosting.
             In International workshop on artificial intelligence and statistics (pp.
             229-236). PMLR. <https://proceedings.mlr.press/r3/oza01a.html>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 1,
        base_learner=None,
        ensemble_size: int = 100,
    ):
        """Construct an Online bagging regressor using online bootstrap sampling.

        :param schema: The schema of the stream.
        :param random_seed: Seed for the random number generator.
        :param base_learner: The base learner to use. It can be a CLI string, a CapyMOA
            regressor, or a MOA Java object. If not provided, a default FIMTDD is used.
        :param ensemble_size: The number of base learners in the ensemble.
        """
        self.base_learner = "trees.FIMTDD" if base_learner is None else base_learner
        self.ensemble_size = ensemble_size
        super().__init__(
            schema=schema,
            random_seed=random_seed,
            moa_learner=_MOA_OzaBag(),
            options={
                "baseLearner": self.base_learner,
                "ensembleSize": self.ensemble_size,
            },
        )

    def __str__(self):
        # Overrides the default class name from MOA (OzaBag)
        return "OnlineBaggingRegressor"
