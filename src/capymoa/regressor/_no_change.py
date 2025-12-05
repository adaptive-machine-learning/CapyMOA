from capymoa.base import Regressor
from capymoa.instance import RegressionInstance
from capymoa.type_alias import TargetValue


class NoChange(Regressor):
    """No Change regressor.

    Predict the previous target value. Outputs 0 prior to seeing any data.

    .. seealso::

        :func:`~capymoa.regressor.TargetMean`
        :func:`~capymoa.regressor.FadingTargetMean`
    """

    def __init__(self, schema=None):
        """Construct No Change Regressor."""
        super().__init__(None, 0)
        self.previous_value: TargetValue = TargetValue(0)

    def train(self, instance: RegressionInstance):
        self.previous_value = instance.y_value

    def predict(self, instance: RegressionInstance) -> TargetValue:
        return self.previous_value
