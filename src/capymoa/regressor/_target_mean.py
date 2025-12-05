from capymoa.base import MOARegressor
from capymoa.stream import Schema

from moa.classifiers.rules.functions import TargetMean as _FadingTargetMean


class TargetMean(MOARegressor):
    """Target Mean Regressor.

    Maintains the mean of the target values to make predictions.

    .. seealso::

        :func:`~capymoa.regressor.NoChange`
        :func:`~capymoa.regressor.FadingTargetMean`
    """

    def __init__(self, schema: Schema | None = None):
        """Construct Target Mean Regressor."""
        super().__init__(moa_learner=_FadingTargetMean(), schema=schema)
