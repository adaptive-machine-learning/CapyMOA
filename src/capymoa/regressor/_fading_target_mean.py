from capymoa.base import MOARegressor
from capymoa.stream import Schema

from moa.classifiers.rules.functions import FadingTargetMean as _FadingTargetMean


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

        :param schema: Description of the stream's data types.
        :param fading_factor: The fading factor. Must be in (0, 1]. Defaults to 0.99.
        """
        cli = [f"-f {factor}"]
        super().__init__(
            moa_learner=_FadingTargetMean(), schema=schema, CLI=" ".join(cli)
        )
