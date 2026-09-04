from typing import Literal
from capymoa.base import SKRegressor
from sklearn.linear_model import (
    SGDRegressor as _SKSGDRegressor,
)
from capymoa.stream._stream import Schema

# Maps the passive aggressive loss onto the equivalent scikit-learn learning
# rate schedule: PA-I for the epsilon insensitive loss and PA-II for its
# squared variant.
_LOSS_TO_LEARNING_RATE = {
    "epsilon_insensitive": "pa1",
    "squared_epsilon_insensitive": "pa2",
}


class PassiveAggressiveRegressor(SKRegressor):
    """Streaming Passive Aggressive regressor

    This wraps :sklearn:`linear_model.SGDRegressor` with a passive aggressive
    learning rate schedule for ease of use in the streaming context. Some
    options are missing because they are not relevant in the streaming context.

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

    sklearner: _SKSGDRegressor
    """The underlying scikit-learn object. See: :sklearn:`linear_model.SGDRegressor`"""

    def __init__(
        self,
        schema: Schema,
        max_step_size: float = 1.0,
        fit_intercept: bool = True,
        loss: Literal[
            "epsilon_insensitive", "squared_epsilon_insensitive"
        ] = "epsilon_insensitive",
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
        :raises ValueError: If ``loss`` is not one of the supported losses.
        """

        if loss not in _LOSS_TO_LEARNING_RATE:
            raise ValueError(
                f"Unknown loss {loss!r}, expected one of "
                f"{sorted(_LOSS_TO_LEARNING_RATE)}."
            )

        super().__init__(
            _SKSGDRegressor(
                loss="epsilon_insensitive",
                penalty=None,
                alpha=1.0,
                learning_rate=_LOSS_TO_LEARNING_RATE[loss],
                eta0=max_step_size,
                fit_intercept=fit_intercept,
                early_stopping=False,
                shuffle=False,
                verbose=0,
                warm_start=False,
                average=average,
                random_state=random_seed,
            ),
            schema,
            random_seed,
        )

    def __str__(self):
        return str("PassiveAggressiveRegressor")
