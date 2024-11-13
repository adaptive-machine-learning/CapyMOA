from capymoa.base import SKRegressor
from sklearn.linear_model import (
    PassiveAggressiveRegressor as _SKPassiveAggressiveRegressor,
)
from capymoa.stream._stream import Schema


class PassiveAggressiveRegressor(SKRegressor):
    """Streaming Passive Aggressive regressor

    This wraps :sklearn:`linear_model.PassiveAggressiveRegressor` for
    ease of use in the streaming context. Some options are missing because
    they are not relevant in the streaming context.

    Reference:

    `Online Passive-Aggressive Algorithms K. Crammer, O. Dekel, J. Keshat, S.
    Shalev-Shwartz, Y. Singer - JMLR (2006)
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    Example Usage:

    >>> from capymoa.datasets import Fried
    >>> from capymoa.regressor import PassiveAggressiveRegressor
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = Fried()
    >>> schema = stream.get_schema()
    >>> learner = PassiveAggressiveRegressor(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].rmse()
    3.700...
    """

    sklearner: _SKPassiveAggressiveRegressor
    """The underlying scikit-learn object. See: :sklearn:`linear_model.PassiveAggressiveRegressor`"""

    def __init__(
        self,
        schema: Schema,
        max_step_size: float = 1.0,
        fit_intercept: bool = True,
        loss: str = "epsilon_insensitive",
        average: bool = False,
        random_seed=1,
    ):
        """Construct a passive aggressive regressor.

        :param schema: Stream schema
        :param max_step_size: Maximum step size (regularization).
        :param fit_intercept: Whether the intercept should be estimated or not.
            If False, the data is assumed to be already centered.
        :param loss: The loss function to be used:

          * ``"epsilon_insensitive"``: equivalent to PA-I in the reference paper.
          * ``"squared_epsilon_insensitive"``: equivalent to PA-II in the reference
            paper.

        :param average: When set to True, computes the averaged SGD weights and
            stores the result in the ``sklearner.coef_`` attribute. If set to an int greater
            than 1, averaging will begin once the total number of samples
            seen reaches average. So ``average=10`` will begin averaging after
            seeing 10 samples.
        :param random_seed: Seed for the random number generator.
        """

        super().__init__(
            _SKPassiveAggressiveRegressor(
                C=max_step_size,
                fit_intercept=fit_intercept,
                early_stopping=False,
                shuffle=False,
                verbose=0,
                loss=loss,
                warm_start=False,
                average=average,
                random_state=random_seed,
            ),
            schema,
            random_seed,
        )

    def __str__(self):
        return str("PassiveAggressiveRegressor")
