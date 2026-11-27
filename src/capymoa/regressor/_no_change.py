from capymoa.base import Regressor
from capymoa.core import RegressionInstance, TargetValue
from capymoa.stream import Schema


class NoChange(Regressor):
    """No Change regressor.

    Predict the previous target value. Outputs ``NaN`` prior to seeing any data.

    .. seealso::

        :func:`~capymoa.regressor.TargetMean`
        :func:`~capymoa.regressor.FadingTargetMean`
    """

    def __init__(self, schema: Schema) -> None:
        """Construct a NoChange regressor.

        :param schema: Describes the stream's structure and types.
        """
        super().__init__(None, 0)
        self.previous_value: TargetValue = TargetValue(float("nan"))

    def train(self, instance: RegressionInstance):
        self.previous_value = instance.y_value

    def predict(self, instance: RegressionInstance) -> TargetValue:
        return self.previous_value
