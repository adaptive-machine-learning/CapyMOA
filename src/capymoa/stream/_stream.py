import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generic, Iterator, Literal, Optional, Sequence, TypeVar, Union

from capymoa.exception import StreamTypeError
import numpy as np
from com.yahoo.labs.samoa.instances import (
    Attribute,
    DenseInstance,
    Instances,
    InstancesHeader,
)
from moa.core import FastVector
from moa.streams import ArffFileStream, InstanceStream

from capymoa.instance import (
    Instance,
    LabeledInstance,
    RegressionInstance,
)


# Private functions
def _target_is_categorical(targets, target_type):
    if target_type is None:
        if isinstance(targets[0], (str, bool)):
            return True
        if isinstance(targets[0], (np.integer, int)):
            num_unique = len(np.unique(targets))
            if num_unique >= 20:
                warnings.warn(
                    f"target variable includes {num_unique} (≥ 20) unique values, inferred as numeric, "
                    f"set target_type = 'categorical' if you intend categorical targets"
                )
                return False
            else:
                warnings.warn(
                    f"target variable includes {num_unique} (< 20) unique values, inferred as categorical, "
                    f"set target_type = 'numeric' if you intend numeric targets"
                )
                return True
    elif target_type != "numeric" and target_type != "categorical":
        raise ValueError("target_type must be either numeric or categorical")
    else:
        return target_type == "categorical"


class Schema:
    """Schema describes the structure of a stream.

    It contains the attribute names, datatype, and the possible values for nominal attributes.
    The schema is crucial for a learner to know how to interpret instances correctly.

    When working with datasets built into CapyMOA (see :mod:`capymoa.datasets`)
    and ARFF files, the schema is automatically created. However, in some cases
    you might want to create a schema manually. This can be done using the
    :meth:`from_custom` method.
    """

    def __init__(self, moa_header: InstancesHeader):
        """Construct a schema by wrapping a ``InstancesHeader``.

        To create a schema without an ``InstancesHeader`` use
        :meth:`from_custom` method.

        :param moa_header: A Java MOA header object.
        """
        assert moa_header.numOutputAttributes() == 1, (
            "Only one output attribute is supported."
        )

        self._moa_header = moa_header
        # Internally, we store the number of attributes + the class/target.
        # This is because MOA methods expect the numAttributes to also account for the class/target.
        self._regression = not self._moa_header.outputAttribute(1).isNominal()
        self._shape = (self.get_num_numeric_attributes(),)
        self._label_values: Optional[Sequence[str]] = None
        self._label_index_map: Optional[Dict[str, int]] = None

        if not self._regression:
            values = self._moa_header.outputAttribute(1).getAttributeValues()
            self._label_values = list(map(str, values))
            self._label_index_map = {
                label: i for i, label in enumerate(self._label_values)
            }

        # TODO: iterate over the attributes and create a dictionary representation for the nominal attributes.
        # There should be a way to configure that manually like setting the self.labels instead of using a MOA header.

    def _assert_classification(self):
        if not self.is_classification():
            raise StreamTypeError("Schema is not for a classification task.")

    def get_label_values(self) -> Sequence[str]:
        """Return the possible values for the class label."""
        self._assert_classification()
        return self._label_values

    def get_label_indexes(self) -> Sequence[int]:
        """Return the possible indexes for the class label."""
        self._assert_classification()
        return list(range(self.get_num_classes()))

    def get_value_for_index(self, y_index: Optional[int]) -> Optional[str]:
        """Return the value for the class label index y_index."""
        self._assert_classification()
        if y_index is None or y_index < 0:
            return None
        return self._label_values[y_index]

    def get_index_for_label(self, y: str):
        """Return the index for the class label y."""
        self._assert_classification()
        return self._label_index_map[y]

    def get_moa_header(self) -> InstancesHeader:
        """Get the JAVA MOA header. Useful for advanced users.

        This is needed for advanced operations that are not supported by the
        Python wrappers (yet).
        """
        return self._moa_header

    def get_num_attributes(self) -> int:
        """Return the number of attributes excluding the target attribute."""
        return self._moa_header.numAttributes() - self._moa_header.numOutputAttributes()

    def get_num_nominal_attributes(self) -> int:
        """Return the number of nominal attributes."""
        return sum(
            1
            for i in range(self._moa_header.numAttributes())
            if self._moa_header.attribute(i).isNominal()
            and self._moa_header.classIndex() != i
        )

    def get_num_numeric_attributes(self) -> int:
        """Return the number of numeric attributes."""
        return sum(
            1
            for i in range(self._moa_header.numAttributes())
            if self._moa_header.attribute(i).isNumeric()
            and self._moa_header.classIndex() != i
        )

    def get_nominal_attributes(self) -> Dict[str, Sequence[str]]:
        """Return a dict of nominal attributes."""
        nominal_attributes = {}
        for i in range(self._moa_header.numAttributes()):
            attr = self._moa_header.attribute(i)
            if attr.isNominal() and self._moa_header.classIndex() != i:
                values = attr.getAttributeValues()
                nominal_attributes[attr.name()] = list(map(str, values))
        return nominal_attributes

    def get_numeric_attributes(self) -> Sequence[str]:
        """Return a list of numeric attribute names."""
        numeric_attributes = []
        for i in range(self._moa_header.numAttributes()):
            attr = self._moa_header.attribute(i)
            if attr.isNumeric() and self._moa_header.classIndex() != i:
                numeric_attributes.append(attr.name())
        return numeric_attributes

    def get_num_classes(self) -> int:
        """Return the number of possible classes. If regression, returns 1."""
        if self._regression:
            return 1
        return len(self._label_values)

    def is_regression(self) -> bool:
        """Return True if the problem is a regression problem."""
        return self._regression

    def is_classification(self) -> bool:
        """Return True if the problem is a classification problem."""
        return not self._regression

    def is_y_index_in_range(self, y_index: int) -> bool:
        """Return True if the y_index is in the range of the class label indexes."""
        self._assert_classification()
        return 0 <= y_index < self.get_num_classes()

    @property
    def dataset_name(self) -> str:
        """Returns the name of the dataset."""
        return self._moa_header.getRelationName()

    @property
    def shape(self) -> Sequence[int]:
        """The shape of the input ``x`` instances.

        Usually :py:attr:`capymoa.instance.Instance.x` is a vector but some learners
        need to know the shape of the input. For example, a CNN needs to know the height
        and width of an image.
        """
        return self._shape

    @shape.setter
    def shape(self, value: Sequence[int]):
        """Set the shape of the input ``x`` instances."""
        n_attr = self.get_num_numeric_attributes()
        # ensure the product of the shape matches the number of attributes
        if np.prod(value) != n_attr:
            raise ValueError(
                f"Shape {value} is incompatible with number of attributes {n_attr}"
            )
        self._shape = value

    @staticmethod
    def from_custom(
        features: Sequence[str],
        target: str,
        categories: Optional[Dict[str, Sequence[str]]] = None,
        name: str = "unnamed",
    ):
        """Create a CapyMOA Schema that defines each attribute in the stream.

        The following example shows how to use this method to create a classification
        schema:

        >>> from capymoa.stream import Schema
        >>> schema = Schema.from_custom(
        ...     features=["f1", "f2", "class"],
        ...     target="class",
        ...     categories={"class": ["yes", "no"], "f1": ["low", "medium", "high"]},
        ...     name="classification-example"
        ... )
        >>> print(schema)
        @relation classification-example
        <BLANKLINE>
        @attribute f1 {low,medium,high}
        @attribute f2 numeric
        @attribute class {yes,no}
        <BLANKLINE>
        @data
        >>> print(schema.is_classification())
        True

        The following example shows how to use this method to create a regression
        schema:

        >>> schema = Schema.from_custom(
        ...     features=["f1", "f2", "target"],
        ...     target="target",
        ...     categories={"f1": ["A", "B", "C"]},
        ...     name="regression-example"
        ... )
        >>> print(schema)
        @relation regression-example
        <BLANKLINE>
        @attribute f1 {A,B,C}
        @attribute f2 numeric
        @attribute target numeric
        <BLANKLINE>
        @data
        >>> print(schema.is_regression())
        True

        :param features: A list of feature names.
        :param target: The name of the target attribute. Must be in features as well.
        :param categories: A dictionary mapping feature names to their possible values.
            When the target attribute is included in this dictionary the task is
            considered classification.
        :param name: The name of the dataset.
        :return: A CapyMOA Schema object.
        """
        return Schema(
            _new_instances_header(
                relation=name,
                target=target,
                attributes=features,
                nominals=categories or {},
            )
        )

    def __repr__(self) -> str:
        """Return a string representation of the schema as an ARFF header."""
        return str(self)

    def __str__(self):
        """Return a string representation of the schema as an ARFF header."""
        return str(self._moa_header.toString()).strip()


_AnyInstance = TypeVar("_AnyInstance", bound=Instance)
"""A generic type that is bound to an instance type.

Such as :class:`LabeledInstance` or :class:`RegressionInstance`.
"""


class Stream(ABC, Generic[_AnyInstance], Iterator[_AnyInstance]):
    """A datastream that can be learnt instance by instance."""

    def __iter__(self) -> Iterator[_AnyInstance]:
        """Get an iterator over the stream.

        This will NOT restart the stream if it has already been iterated over.
        Please use the :meth:`restart` method to restart the stream.

        :yield: An iterator over the stream.
        """
        return self

    def __next__(self) -> _AnyInstance:
        """Get the next instance in the stream.

        :return: The next instance in the stream.
        """
        return self.next_instance()

    def __str__(self):
        """Return the name of the datastream from the schema."""
        if moa_stream := self.get_moa_stream():
            return str(moa_stream.getHeader().getRelationName()).replace(" ", "")
        return object.__str__(self)

    def cli_help(self) -> str:
        """Return a help message"""
        if moa_stream := self.get_moa_stream():
            return str(moa_stream.getOptions().getHelpString())
        return "No CLI help available."

    @abstractmethod
    def has_more_instances(self) -> bool:
        """Return ``True`` if the stream have more instances to read."""

    @abstractmethod
    def next_instance(self) -> _AnyInstance:
        """Return the next instance in the stream.

        :raises ValueError: If the machine learning task is neither a regression
            nor a classification task.
        :return: A labeled instances or a regression depending on the schema.
        """

    @abstractmethod
    def get_schema(self) -> Schema:
        """Return the schema of the stream."""

    def get_moa_stream(self) -> Optional[InstanceStream]:
        """Get the MOA stream object if it exists."""
        return None

    @abstractmethod
    def restart(self):
        """Restart the stream to read instances from the beginning."""


class MOAStream(Stream[_AnyInstance]):
    """A datastream that can be learnt instance by instance."""

    # TODO: A problem in stream is that it has lots of conditional logic to
    # support a variety of ways to create a Stream object. This makes the code
    # harder to understand and maintain. We should consider refactoring this
    # with a abstract base class and subclasses for each type of stream.
    # TODO (update): We have created an abstract base class for Stream but
    # we the rest of the refactor is still work in progress.

    def __init__(
        self,
        moa_stream: Optional[InstanceStream] = None,
        schema: Optional[Schema] = None,
        CLI: Optional[str] = None,
    ):
        """Construct a Stream from a MOA stream object.

        Usually, you will want to construct a Stream using :class:`ARFFStream`,
        :class:`NumpyStream`, or :class:`CSVStream`.


        :param moa_stream: The MOA stream object to read instances from. Is None
            if the stream is created from a numpy array.
        :param schema: The schema of the stream. If None, the schema is inferred
            from the moa_stream.
        :param CLI: Additional command line arguments to pass to the MOA stream.
        :raises ValueError: If no schema is provided and no moa_stream is provided.
        :raises ValueError: If command line arguments are provided without a moa_stream.
        """
        self.schema = schema
        self.moa_stream = moa_stream
        self._CLI = CLI

        # Set the CLI arguments if they are provided.
        if self._CLI is not None:
            if self.moa_stream is None:
                raise ValueError(
                    "Command line arguments cannot be used without a moa_stream"
                )
            self.moa_stream.getOptions().setViaCLIString(CLI)

        # Infer the schema from the moa_stream if it is not provided.
        if self.schema is None:
            if self.moa_stream is None:
                raise ValueError("Must provide a schema if no moa_stream is provided.")
            self.moa_stream.prepareForUse()  # This is necessary to get the header from the stream.
            self.schema = Schema(moa_header=self.moa_stream.getHeader())
        elif self.moa_stream is not None:
            self.moa_stream.prepareForUse()

    def __str__(self):
        """Return the name of the datastream from the schema."""
        return str(self.schema._moa_header.getRelationName()).replace(" ", "")

    def cli_help(self) -> str:
        """Return cli help string for the stream."""
        return str(
            self.moa_stream.getOptions().getHelpString()
            if self.moa_stream is not None
            else ""
        )

    def has_more_instances(self) -> bool:
        """Return `True` if the stream have more instances to read."""
        return self.moa_stream.hasMoreInstances()

    def next_instance(self) -> _AnyInstance:
        """Return the next instance in the stream.

        :raises ValueError: If the machine learning task is neither a regression
            nor a classification task.
        :return: A labeled instances or a regression depending on the schema.
        """
        if not self.has_more_instances():
            raise StopIteration()
        java_instance = self.moa_stream.nextInstance()
        if self.schema.is_regression():
            return RegressionInstance.from_java_instance(self.schema, java_instance)
        elif self.schema.is_classification():
            return LabeledInstance.from_java_instance(self.schema, java_instance)
        else:
            raise ValueError(
                "Unknown machine learning task must be a regression "
                "or classification task"
            )

    def get_schema(self) -> Schema:
        """Return the schema of the stream."""
        return self.schema

    def get_moa_stream(self) -> Optional[InstanceStream]:
        """Get the MOA stream object if it exists."""
        return self.moa_stream

    def restart(self):
        """Restart the stream to read instances from the beginning."""
        self.moa_stream.restart()


class ARFFStream(MOAStream[_AnyInstance]):
    """A datastream originating from an ARFF file."""

    def __init__(
        self, path: Union[str, Path], CLI: Optional[str] = None, class_index: int = -1
    ):
        """Construct an ARFFStream object from a file path.

        :param path: A filepath
        :param CLI: Additional command line arguments to pass to the MOA stream.
        """
        # convert 0-based index to 1-based index for MOA
        if class_index >= 0:
            class_index += 1
        elif class_index != -1:
            raise ValueError("class_index must be -1 (last attribute) or >= 0")

        moa_stream = ArffFileStream(str(path), class_index)
        super().__init__(moa_stream=moa_stream, CLI=CLI)


class NumpyStream(Stream[_AnyInstance]):
    """A datastream originating from a numpy array.

    >>> from capymoa.stream import NumpyStream
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = np.array([0, 1])
    >>> stream: NumpyStream[LabeledInstance] = NumpyStream(X, y, dataset_name="MyDataset")
    >>> for instance in stream:
    ...     print(instance)
    LabeledInstance(
        Schema(MyDataset),
        x=[1. 2. 3.],
        y_index=0,
        y_label='0'
    )
    LabeledInstance(
        Schema(MyDataset),
        x=[4. 5. 6.],
        y_index=1,
        y_label='1'
    )
    """

    # This class is more complex than ARFFStream because it needs to read and convert the CSV to an ARFF in memory.
    # target_type to specify the target as 'categorical' or 'numeric', None for detecting automatically.

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "No_Name",
        feature_names: Sequence[str] | None = None,
        target_name: str | None = None,
        target_type: Literal["numeric", "categorical"] = "categorical",
    ):
        """Construct a NumpyStream object from a numpy array.

        :param X: Numpy array of shape (n_samples, n_features) with the feature values
        :param y: Numpy array of shape (n_samples,) with the target values
        :param dataset_name: The name to give to the datastream, defaults to "No_Name"
        :param feature_names: The names given to the features, defaults to None
        :param target_name: The name given to target values, defaults to None
        :param target_type: 'categorical' or 'numeric' target, defaults to None
        """

        features = []
        if feature_names is None:
            features = [f"{i}" for i in range(X.shape[1])]
        else:
            features = list(feature_names)

        if target_name is None:
            target_name = "target"
        features.append(target_name)

        categories: Dict[str, Sequence[str]] = {}
        if target_type == "categorical":
            n_classes = np.sum(~np.isnan(np.unique(y, equal_nan=True)))
            categories[target_name] = [str(i) for i in range(n_classes)]

        self.schema = Schema.from_custom(
            features=features,
            target=target_name,
            categories=categories,
            name=dataset_name,
        )
        self._index = 0
        self._len = X.shape[0]
        self._x_data = X
        self._y_data = y

    def has_more_instances(self):
        return self._len > self._index

    def next_instance(self) -> _AnyInstance:
        # Raise StopIteration if there are no more instances
        if not self.has_more_instances():
            raise StopIteration()

        # Get the next instance
        x = self._x_data[self._index]
        y = self._y_data[self._index]
        self._index += 1

        if self.schema.is_classification():
            return LabeledInstance.from_array(self.schema, x, y)  # type: ignore
        elif self.schema.is_regression():
            return RegressionInstance.from_array(self.schema, x, y)  # type: ignore
        else:
            raise ValueError(
                "Unknown machine learning task must be a regression or "
                "classification task"
            )

    def get_schema(self):
        return self.schema

    def restart(self):
        self._index = 0

    def __len__(self) -> int:
        return self._len


def _numpy_to_arff(
    X,
    y,
    dataset_name: str = "No_Name",
    feature_names: str = None,
    target_name: str = None,
    target_type: str = None,
):
    """Converts a numpy X and y into a ARFF format. The code first check if the user has specified the type of the
    target values, if not, the code infers whether it is a categorical or numeric target by _target_is_categorical
    method, i.e., if the unique values in the targets are more than 20, interpret as numeric, and vice versa.
    """
    number_of_instances = X.shape[0]
    class_labels = (
        None
        if not _target_is_categorical(y, target_type) or target_type == "numeric"
        else [str(value) for value in np.unique(y)]
    )
    feature_names = (
        [f"attrib_{i}" for i in range(X.shape[1])]
        if feature_names is None
        else feature_names
    )
    moa_stream, moa_header = _init_moa_stream_and_create_moa_header(
        number_of_instances=number_of_instances,
        feature_names=feature_names,
        values_for_class_label=class_labels,
        dataset_name=dataset_name,
        target_attribute_name=target_name,
        target_type=target_type,
    )
    _add_instances_to_moa_stream(moa_stream, moa_header, X, y)
    return moa_stream, moa_header, class_labels


def _create_nominal_attribute(name: str, values: Sequence[str]):
    value_list = FastVector()
    for value in values:
        value_list.addElement(str(value))
    return Attribute(name, value_list)


def _new_instances_header(
    relation: str,
    target: str,
    attributes: Sequence[str],
    nominals: Dict[str, Sequence[str]],
) -> InstancesHeader:
    attributes_ = FastVector()
    for attribute in attributes:
        if attribute in nominals:
            attr = _create_nominal_attribute(attribute, nominals[attribute])
        else:
            attr = Attribute(attribute)
        attributes_.addElement(attr)

    moa_stream = Instances(relation, attributes_, 0)
    moa_stream.setClassIndex(attributes.index(target))
    return InstancesHeader(moa_stream)


def _init_moa_stream_and_create_moa_header(
    number_of_instances: int = 100,
    feature_names: list = None,
    values_for_nominal_features={},
    values_for_class_label: list = None,
    dataset_name="No_Name",
    target_attribute_name=None,
    target_type: str = None,
):
    """Initialize a moa stream with number_of_instances capacity and create a
    MOA header containing all the necessary attribute information.

     Note: The instances are not added to the moa_stream.

    Sample code to get relevant information from two Numpy arrays: X[rows][features] and y[rows]

        feature_names = [f"attrib_{i}" for i in range(X.shape[1])]
        number_of_instances = X.shape[0]
        values_for_class_label = [str(value) for value in np.unique(y)]

    :param number_of_instances: number of instances in the stream
    :param feature_names: a list containing names of features. if none sets a default name
    :param values_for_nominal_features: possible values of each nominal feature.
    e.g {i: [1,2,3], k: [Aa, BB]}. Key is integer. Values are turned into strings
    :param values_for_class_label: possible values for class label. Values are turned into strings
    :param dataset_name: name of the dataset. Defaults to "No_Name"
    :param target_attribute_name: name for the target/class attribute
    :param target_type: specifies the type of target as 'categorical' or 'numeric', None to detect automatically

    :return moa_stream: initialized moa stream with capacity number_of_instances.
    :return moa_header: initialized moa header which contain all necessary attribute information for all features and
        the class label
    """
    attributes = FastVector()
    # Attribute("name") will create a numeric attribute;
    # Attribute("name", array_of_values) will create a nominal attribute
    if feature_names is None:
        raise ValueError("feature_names are None")

    if target_type == "numeric" or values_for_class_label is None:
        if target_attribute_name is None:
            target_attribute = Attribute("target")
        else:
            target_attribute = Attribute(target_attribute_name)
    elif target_type == "categorical" or target_type is None:
        target_attribute = _create_nominal_attribute(
            ("class" if target_attribute_name is None else target_attribute_name),
            values_for_class_label,
        )
    else:
        raise ValueError("target_type must be either `numeric` or `categorical`")

    # we don't want to add the class attribute as a feature
    if target_attribute.name() in feature_names:
        feature_names.remove(target_attribute.name())

    for name in feature_names:
        if name in values_for_nominal_features:
            attribute = _create_nominal_attribute(
                name,
                values_for_nominal_features.get(name),
            )
        else:
            attribute = Attribute(name)
        attributes.addElement(attribute)

    attributes.addElement(target_attribute)

    moa_stream = Instances(dataset_name, attributes, number_of_instances)
    # set last index for class index
    moa_stream.setClassIndex(len(attributes) - 1)
    # create stream header
    moa_header = InstancesHeader(moa_stream)
    return moa_stream, moa_header


def _add_instances_to_moa_stream(moa_stream, moa_header, X, y):
    for instance_index in range(X.shape[0]):
        instance = DenseInstance(moa_header.numAttributes())

        for attribute_index in range(X.shape[1]):
            instance.setValue(
                attribute_index, X[instance_index, attribute_index]
            )  # set value for each attribute

        instance.setDataset(moa_header)
        instance.setWeight(1.0)  # a default weight of 1.0
        instance.setClassValue(y[instance_index])  # set class value

        moa_stream.add(instance)
