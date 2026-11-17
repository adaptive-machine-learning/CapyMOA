from moa.classifiers.rules.functions import FadingTargetMean as _FadingTargetMean

from capymoa.base import MOARegressor
from capymoa.stream import Schema


class FadingTargetMean(MOARegressor):
    """Fading Target Mean Regressor.

    Maintains a fading mean of the target values to make predictions. The fading factor
    determines the rate at which older observations are discounted. A fading factor close
    to 1.0 gives more weight to older observations, while a factor closer to 0.0 emphasizes
    more recent data.

    .. seealso::

        :func:`~capymoa.regressor.NoChange`
        :func:`~capymoa.regressor.TargetMean`
    """

    def __init__(self, schema: Schema | None = None, factor: float = 0.99):
        """Construct Fading Target Mean Regressor.

        :param schema: Describes the stream's structure and types.
        :param factor: The fading factor. Must be in (0, 1].
        """
        cli = [f"-f {factor}"]
        super().__init__(
            moa_learner=_FadingTargetMean(), schema=schema, CLI=" ".join(cli)
        )
