from moa.classifiers.rules.functions import TargetMean as _FadingTargetMean

from capymoa.base import MOARegressor
from capymoa.stream import Schema


class TargetMean(MOARegressor):
    """Target Mean Regressor.

    Maintains the mean of the target values to make predictions.

    .. seealso::

        :func:`~capymoa.regressor.NoChange`
        :func:`~capymoa.regressor.FadingTargetMean`
    """

    def __init__(self, schema: Schema | None = None) -> None:
        """Construct a TargetMean regressor.

        :param schema: Describes the stream's structure and types.
        """
        super().__init__(moa_learner=_FadingTargetMean(), schema=schema)
