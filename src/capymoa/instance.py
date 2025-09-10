from typing import TYPE_CHECKING

import numpy as np
from com.yahoo.labs.samoa.instances import DenseInstance
from moa.core import InstanceExample
from typing import Optional, Union, Tuple
from jpype import JArray, JDouble

from capymoa.type_alias import FeatureVector, Label, LabelIndex, TargetValue

# Schema is only imported for type hinting. If we were to import Schema directly
# in this file, it would create a circular import.
if TYPE_CHECKING:
    from capymoa.stream import Schema


def _features_to_string(
    x: np.ndarray, prefix: str = "\n    x=", suffix: str = ","
) -> str:
    """Return an array as a pretty string that shortens and wraps them.

    Used for the ``__repr__`` methods of instances.
    """
    return (
        prefix
        + np.array2string(
            x,
            max_line_width=80,
            threshold=10,
            prefix=prefix,
            suffix=suffix,
            precision=3,
        )
        + suffix
    )


class Instance:
    """An instance is a single data point in a stream. It contains a feature vector
    and a schema that describes the datastream it belongs to.

    In supervised learning, your more likely to encounter :class:`LabeledInstance`
    or :class:`RegressionInstance` which are subclasses of :class:`Instance` with
    a class label or target value respectively.
    """

    def __init__(
        self, schema: "Schema", instance: Union[InstanceExample, FeatureVector]
    ) -> None:
        """Creates a new instance.

        Its recommended that you prefer using :meth:`from_array` or
        :meth:`from_java_instance` to create instances, as they provide a more
        user-friendly interface.

        :param schema: A schema that describes the datastream the instance belongs to.
        :param instance: A vector of features (float values) or a Java instance.
        :raises ValueError: If the given instance type is of an unsupported type.
        """
        self._schema: "Schema" = schema
        self._java_instance: Optional[InstanceExample] = None
        self._x: Optional[FeatureVector] = None

        if isinstance(instance, InstanceExample):
            self._java_instance = instance
        elif isinstance(instance, np.ndarray):
            self._x = instance
        else:
            raise ValueError(f"Given instance type unsupported: {type(instance)}")

    @classmethod
    def from_java_instance(
        cls, schema: "Schema", java_instance: InstanceExample
    ) -> "Instance":
        return cls(schema, java_instance)

    @classmethod
    def from_array(cls, schema: "Schema", instance: FeatureVector) -> "Instance":
        """A class constructor to create an instance from a schema and a vector of features.

        This is useful in the rare cases you need to create custom unlabeled instances
        from scratch. In most cases, your datastream will automatically create
        instances for you.

        >>> from capymoa.stream import Schema
        ...
        >>> from capymoa.instance import Instance
        >>> import numpy as np
        >>> schema = Schema.from_custom(
        ...     ["f1", "f2"],
        ...     dataset_name="CustomDataset",
        ...     values_for_class_label=["yes", "no"]
        ... )
        >>> x = np.array([0.1, 0.2])
        >>> instance = Instance.from_array(schema, x)
        >>> instance
        Instance(
            Schema(CustomDataset),
            x=[0.1 0.2],
        )

        :param schema: A schema that describes the datastream the instance belongs to.
        :param instance: A vector (:class:`numpy.ndarray`) of features (float values
        :return: A new :class:`Instance` object
        """
        return cls(schema, instance)

    @property
    def schema(self) -> "Schema":
        """Returns the schema of the instance and the stream it belongs to."""
        return self._schema

    @property
    def x(self) -> FeatureVector:
        """Returns a feature vector containing float values for the instance."""
        if self._x is not None:
            return self._x
        elif self._java_instance is not None:
            moa_instance = self.java_instance.getData()
            self._x = np.empty(moa_instance.numInputAttributes())
            for i in range(0, moa_instance.numInputAttributes()):
                self._x[i] = moa_instance.value(i)
            return self._x
        else:
            raise ValueError("Instance has no feature vector")

    def _set_y(self, instance: DenseInstance) -> DenseInstance:
        """Helper function to set the class label of an instance created in Python.
        It is overridden by :class:`LabeledInstance` and :class:`RegressionInstance`
        to change only the behavior of the target attribute.
        """
        instance.setMissing(instance.classIndex())
        return instance

    @property
    def java_instance(self) -> InstanceExample:
        """Returns a representation of the instance in Java for use in MOA. This
        method is for advanced users who want to directly interact with MOA's Java
        API.
        """
        if self._java_instance is not None:
            return self._java_instance
        elif self._x is not None:
            assert self._x.ndim == 1, "Feature vector must be 1D."
            # Allocate Array of doubles for the Java instance.
            # The number of attributes is the same as the number of features +
            # one for the target value regardless of if the instance type or
            # if it is labeled or not.
            jx = JArray(JDouble)(len(self.x) + 1)  # type: ignore
            # Set the values of the Java instance.
            jx[: len(self.x)] = self.x
            instance = DenseInstance(1.0, jx)
            instance.setDataset(self.schema.get_moa_header())
            self._java_instance = InstanceExample(self._set_y(instance))
            return self._java_instance

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + f"\n    Schema({self.schema.dataset_name}),"
            + _features_to_string(self.x)
            + "\n)"
        )


class LabeledInstance(Instance):
    """An :class:`Instance` with a class label.

    Most classification datastreams will automatically return instances for you
    with the class label and index. For example, the :class:`capymoa.datasets.ElectricityTiny`
    dataset:

    >>> from capymoa.datasets import ElectricityTiny
    ...
    >>> from capymoa.instance import LabeledInstance
    >>> stream = ElectricityTiny()
    >>> instance: LabeledInstance = stream.next_instance()
    >>> instance.y_label
    '1'

    The label and index are NOT the same. One is a human-readable string
    and the other is a integer representation of the class label.
    >>> instance.y_index
    1
    >>> instance.x
    array([0.      , 0.056443, 0.439155, 0.003467, 0.422915, 0.414912])
    """

    def __init__(
        self,
        schema: "Schema",
        instance: Union[InstanceExample, Tuple[FeatureVector, LabelIndex]],
    ) -> None:
        self._y_index: Optional[LabelIndex] = None
        if isinstance(instance, tuple):
            instance, self._y_index = instance
        super().__init__(schema, instance)

    @classmethod
    def from_array(
        cls, schema: "Schema", x: FeatureVector, y_index: LabelIndex
    ) -> "LabeledInstance":
        """Creates a new labeled instance from a schema, feature vector, and class index.

        This is useful in the rare cases you need to create custom labeled instances
        from scratch. In most cases, your datastream will automatically create
        instances for you.

        >>> from capymoa.stream import Schema
        ...
        >>> from capymoa.instance import LabeledInstance
        >>> import numpy as np
        >>> schema = Schema.from_custom(
        ...     ["f1", "f2"],
        ...     dataset_name="CustomDataset",
        ...     values_for_class_label=["yes", "no"]
        ... )
        >>> x = np.array([0.1, 0.2])
        >>> instance = LabeledInstance.from_array(schema, x, 0)
        >>> instance
        LabeledInstance(
            Schema(CustomDataset),
            x=[0.1 0.2],
            y_index=0,
            y_label='yes'
        )
        >>> instance.y_label
        'yes'
        >>> instance.java_instance.toString()
        '0.1,0.2,yes,'

        :param schema: _description_
        :param x: _description_
        :param y_index: _description_
        :return: _description_
        """
        return cls(schema, (x, int(y_index)))

    @property
    def y_label(self) -> Label:
        """Returns the class label of the instance as a string."""
        return self.schema.get_value_for_index(self.y_index)

    @property
    def y_index(self) -> LabelIndex:
        """Returns the index of the class. It is useful for classification
        tasks as it provides a numeric representation of the class label, ranging
        from zero to the number of classes.
        """
        if self._y_index is not None:
            return self._y_index
        elif self._java_instance is not None:
            self._y_index = int(self.java_instance.getData().classValue())
            return self._y_index
        else:
            raise ValueError(f"{self.__class__.__name__} must have a y_index.")

    def _set_y(self, instance: DenseInstance) -> DenseInstance:
        instance.setClassValue(self.y_index)
        return instance

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + f"\n    Schema({self.schema.dataset_name}),"
            + _features_to_string(self.x)
            + f"\n    y_index={self.y_index},"
            + f"\n    y_label='{self.y_label}'"
            + "\n)"
        )


class RegressionInstance(Instance):
    """An :class:`Instance` with a continuous target value.

    Most of the time, regression datastreams will automatically return instances
    for you with the target value. For example, the :class:`capymoa.datasets.Fried`
    dataset:

    >>> from capymoa.datasets import Fried
    ...
    >>> from capymoa.instance import RegressionInstance
    >>> stream = Fried()
    >>> instance: RegressionInstance = stream.next_instance()
    >>> instance.y_value
    17.949
    >>> instance.x
    array([0.487, 0.072, 0.004, 0.833, 0.765, 0.6  , 0.132, 0.886, 0.073,
           0.342])

    """

    def __init__(
        self,
        schema: "Schema",
        instance: Union[InstanceExample, Tuple[FeatureVector, TargetValue]],
    ) -> None:
        self._y_value: Optional[TargetValue] = None
        if isinstance(instance, tuple):
            instance, self._y_value = instance
        super().__init__(schema, instance)

    @classmethod
    def from_array(
        cls, schema: "Schema", x: FeatureVector, y_value: TargetValue
    ) -> "RegressionInstance":
        """Creates a new regression instance from a schema, feature vector, and target value.

        This is useful in the rare cases you need to create custom regression instances
        from scratch. In most cases, your datastream will automatically create
        these for you.

        >>> from capymoa.stream import Schema
        ...
        >>> from capymoa.instance import LabeledInstance
        >>> import numpy as np
        >>> schema = Schema.from_custom(
        ...     ["f1", "f2"],
        ...     dataset_name="CustomDataset",
        ...     target_type='numeric'
        ... )
        >>> x = np.array([0.1, 0.2])
        >>> instance = RegressionInstance.from_array(schema, x, 0.5)
        >>> instance
        RegressionInstance(
            Schema(CustomDataset),
            x=[0.1 0.2],
            y_value=0.5
        )
        >>> instance.y_value
        0.5
        >>> instance.java_instance.toString()
        '0.1,0.2,0.5,'

        :param schema: A schema describing the datastream the instance belongs to.
        :param x: A vector of features :class:`numpy.ndarray` containing float values.
        :param y_value: A float value representing the target value or dependent variable.
        :return: A new :class:`RegressionInstance` object.
        """
        return cls(schema, (x, y_value))

    @property
    def y_value(self) -> TargetValue:
        """Returns the target value of the instance."""
        if self._y_value is not None:
            return self._y_value
        elif self._java_instance is not None:
            self._y_value = self.java_instance.getData().classValue()
            return self._y_value
        else:
            raise ValueError(f"{self.__class__.__name__} must have a y_value.")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + f"\n    Schema({self.schema.dataset_name}),"
            + _features_to_string(self.x)
            + f"\n    y_value={self.y_value}"
            + "\n)"
        )

    def _set_y(self, instance: DenseInstance) -> DenseInstance:
        instance.setClassValue(self._y_value)
        return instance
