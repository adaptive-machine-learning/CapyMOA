import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generic, Iterator, Optional, Sequence, TypeVar, Union

import numpy as np
from com.yahoo.labs.samoa.instances import (
    Attribute,
    DenseInstance,
    Instances,
    InstancesHeader,
)
from moa.core import FastVector, InstanceExample
from moa.streams import ArffFileStream, InstanceStream
from numpy.lib import recfunctions as rfn

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
            raise RuntimeError("Should only be called for classification problems.")

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
        if y_index is None:
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
        num_features = self.get_num_attributes()
        num_nominal = 0
        for i in range(num_features):
            if self._moa_header.attribute(i).isNominal():
                num_nominal += 1
        return num_nominal

    def get_num_numeric_attributes(self) -> int:
        """Return the number of numeric attributes."""
        num_features = self.get_num_attributes()
        num_numeric = 0
        for i in range(num_features):
            if self._moa_header.attribute(i).isNumeric():
                num_numeric += 1
        return num_numeric

    def get_nominal_attributes(self) -> dict | None:
        """Return a dict of nominal attributes."""
        num_features = self.get_num_attributes()
        if self.get_num_nominal_attributes() <= 0:
            return None
        else:
            nominal_attributes = {}
            for i in range(num_features):
                if self._moa_header.attribute(i).isNominal():
                    nominal_attributes[self._moa_header.attribute(i).name()] = list(
                        self._moa_header.attribute(i).getAttributeValues()
                    )
            return nominal_attributes

    def get_numeric_attributes(self) -> list | None:
        """Return a list of numeric attribute names."""
        num_features = self.get_num_attributes()
        if self.get_num_numeric_attributes() <= 0:
            return None
        else:
            numeric_attributes = []
            for i in range(num_features):
                if self._moa_header.attribute(i).isNumeric():
                    numeric_attributes.append(self._moa_header.attribute(i).name())
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

    @staticmethod
    def from_custom(
        feature_names: Sequence[str],
        values_for_nominal_features: Dict[str, Sequence[str]] = {},
        values_for_class_label: Sequence[str] = None,
        dataset_name="No_Name",
        target_attribute_name=None,
        target_type=None,
    ):
        """Create a CapyMOA Schema that defines each attribute in the stream.

        The following example shows how to use this method to create a classification schema:

        >>> from capymoa.stream import Schema
        ...
        >>> Schema.from_custom(
        ...     feature_names=["attrib_1", "attrib_2"],
        ...     dataset_name="MyClassification",
        ...     target_attribute_name="class",
        ...     values_for_class_label=["yes", "no"])
        @relation MyClassification
        <BLANKLINE>
        @attribute attrib_1 numeric
        @attribute attrib_2 numeric
        @attribute class {yes,no}
        <BLANKLINE>
        @data

        The following example shows how to use this method to create a regression schema:

        >>> Schema.from_custom(
        ...     feature_names=["attrib_1", "attrib_2"],
        ...     values_for_nominal_features={"attrib_1": ["a", "b"]},
        ...     dataset_name="MyRegression",
        ...     target_attribute_name="target",
        ...     target_type='numeric')
        @relation MyRegression
        <BLANKLINE>
        @attribute attrib_1 {a,b}
        @attribute attrib_2 numeric
        @attribute target numeric
        <BLANKLINE>
        @data

        Sample code to get relevant information from two Numpy arrays: X[rows][features] and y[rows]

        :param feature_names: A list containing names of features. if none sets
            a default name.
        :param values_for_nominal_features: Possible values of each nominal feature.
        :param values_for_class_label: Possible values for class label. Values
            are turned into strings.
        :param dataset_name: Name of the dataset. Default is "No_Name".
        :param target_attribute_name: Name of the target/class attribute.
            Default is None.
        :param target_type: Set the target type as 'categorical' or 'numeric', None to detect automatically.
        :return CayMOA Schema: Initialized CapyMOA Schema which contain all
            necessary attribute information for all features and the class label
        """
        _, moa_header = _init_moa_stream_and_create_moa_header(
            feature_names=feature_names,
            values_for_nominal_features=values_for_nominal_features,
            values_for_class_label=values_for_class_label,
            dataset_name=dataset_name,
            target_attribute_name=target_attribute_name,
            target_type=target_type,
        )
        return Schema(moa_header=moa_header)

    def __repr__(self) -> str:
        """Return a string representation of the schema as an ARFF header."""
        return str(self)

    def __str__(self):
        """Return a string representation of the schema as an ARFF header."""
        return str(self._moa_header.toString()).strip()

    def __eq__(self, other: "Schema") -> bool:
        """Return True if the schema is equal to another schema.

        This is used by :meth:`ConcatStream` to check if the schemas are compatible
        before concatenating the streams.
        """
        if self.is_classification() and other.is_classification():
            return (
                self.get_num_classes() == other.get_num_classes()
                and self.get_num_attributes() == other.get_num_attributes()
            )
        elif self.is_regression() and other.is_regression():
            return self.get_num_attributes() == other.get_num_attributes()
        else:
            return False


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
        if not self.has_more_instances():
            raise StopIteration()
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

        Usually, you will want to construct a Stream using the :func:`capymoa.stream.stream_from_file`
        function.

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
        dataset_name="No_Name",
        feature_names=None,
        target_name=None,
        target_type: str = None,  # numeric or categorical
    ):
        """Construct a NumpyStream object from a numpy array.

        :param X: Numpy array of shape (n_samples, n_features) with the feature values
        :param y: Numpy array of shape (n_samples,) with the target values
        :param dataset_name: The name to give to the datastream, defaults to "No_Name"
        :param feature_names: The names given to the features, defaults to None
        :param target_name: The name given to target values, defaults to None
        :param target_type: 'categorical' or 'numeric' target, defaults to None
        """
        self.current_instance_index = 0

        self.arff_instances_data, self.arff_instances_header, class_labels = (
            _numpy_to_arff(
                X,
                y,
                dataset_name,
                feature_names=feature_names,
                target_name=target_name,
                target_type=target_type,
            )
        )

        self.schema = Schema(moa_header=self.arff_instances_header)

    def has_more_instances(self):
        return self.arff_instances_data.numInstances() > self.current_instance_index

    def next_instance(self) -> _AnyInstance:
        # Return None if all instances have been read already.
        if not self.has_more_instances():
            return None

        instance = self.arff_instances_data.instance(self.current_instance_index)
        self.current_instance_index += 1

        # TODO: We should natively support Numpy as a type of instance, rather
        # than converting it to a Java instance. We can probably combine the logic
        # for pytorch and numpy into a single method.
        if self.schema.is_classification():
            return LabeledInstance.from_java_instance(
                self.schema, InstanceExample(instance)
            )
        elif self.schema.is_regression():
            return RegressionInstance.from_java_instance(
                self.schema, InstanceExample(instance)
            )
        else:
            raise ValueError(
                "Unknown machine learning task must be a regression or "
                "classification task"
            )

    def get_schema(self):
        return self.schema

    def restart(self):
        self.current_instance_index = 0

    def __len__(self) -> int:
        return self.arff_instances_data.numInstances()


def stream_from_file(
    path_to_csv_or_arff: Union[str, Path],
    dataset_name: str = "NoName",
    class_index: int = -1,
    target_type: str = None,  # "numeric" or "categorical"
) -> Stream:
    """Create a datastream from a csv or arff file.

    >>> from capymoa.stream import stream_from_file
    >>> stream = stream_from_file("data/electricity_tiny.csv", dataset_name="Electricity")
    >>> stream.next_instance()
    LabeledInstance(
        Schema(Electricity),
        x=[0.    0.056 0.439 0.003 0.423 0.415],
        y_index=1,
        y_label='1'
    )
    >>> stream.next_instance().x
    array([0.021277, 0.051699, 0.415055, 0.003467, 0.422915, 0.414912])

    :param path_to_csv_or_arff: A file path to a CSV or ARFF file.
    :param dataset_name: A descriptive name given to the dataset, defaults to "NoName"
    :param class_index: The index of the column containing the class label. By default, the algorithm assumes that the
        class label is located in the column specified by this index. However, if the class label is located in a
        different column, you can specify its index using this parameter.
    :param target_type: When working with a CSV file, this parameter
        allows the user to specify the target values in the data to be interpreted as categorical or numeric.
        Defaults to None to detect automatically.
    """
    filename = Path(path_to_csv_or_arff)
    if not filename.exists():
        raise FileNotFoundError(f"No such file or directory: '{filename}'")
    if filename.is_dir():
        raise IsADirectoryError(f"Is a directory: '{filename}'")

    if filename.suffix == ".arff":
        return ARFFStream(path=filename.as_posix(), class_index=class_index)
    elif filename.suffix == ".csv":
        return CSVStream(
            filename.as_posix(),
            dataset_name=dataset_name,
            class_index=class_index,
            target_type=target_type,
        )
    else:
        raise ValueError(
            f"Unsupported file type: expected '.arff' or '.csv', but got '{filename.suffix}'"
        )


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


def _create_nominal_attribute(attribute_name=None, possible_values: list = None):
    value_list = FastVector()
    for value in possible_values:
        value_list.addElement(str(value))
    return Attribute(attribute_name, value_list)


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
            attribute_name=(
                "class" if target_attribute_name is None else target_attribute_name
            ),
            possible_values=values_for_class_label,
        )
    else:
        raise ValueError("target_type must be either `numeric` or `categorical`")

    # we don't want to add the class attribute as a feature
    if target_attribute.name() in feature_names:
        feature_names.remove(target_attribute.name())

    for name in feature_names:
        if name in values_for_nominal_features:
            attribute = _create_nominal_attribute(
                attribute_name=name,
                possible_values=values_for_nominal_features.get(name),
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


class CSVStream(Stream[_AnyInstance]):
    def __init__(
        self,
        csv_file_path,
        dtypes: list = None,  # [('column1', np.float64), ('column2', np.int32), ('column3', np.float64), ('column3', str)] reads nomonal attributes as str
        values_for_nominal_features={},  # {i: [1,2,3], k: [Aa, BB]}. Key is integer. Values are turned into strings
        class_index: int = -1,
        values_for_class_label: list = None,
        target_attribute_name=None,
        target_type: str = None,
        skip_header: bool = False,
        delimiter=",",
        dataset_name: Optional[str] = None,
    ):
        self.csv_file_path = csv_file_path
        self.values_for_nominal_features = values_for_nominal_features
        self.class_index = class_index
        self.values_for_class_label = values_for_class_label
        self.target_attribute_name = target_attribute_name
        self.target_type = target_type
        self.skip_header = skip_header
        self.delimiter = delimiter

        if dataset_name is None:
            dataset_name = f"CSVStream({csv_file_path})"

        self.dtypes = []  # [('column1', np.float64), ('column2', np.int32), ('column3', np.float64), ('column3', str)] reads nomonal attributes as str
        if (
            dtypes is None or len(dtypes) == 0
        ):  # data definition for each column not provided
            if (
                len(self.values_for_nominal_features) == 0
            ):  # data definition for nominal features are given
                # need to infer number of columns, then generate full data definition using nominal information
                # LOADS FIRST TWO ROWS INTO THE MEMORY
                data = np.genfromtxt(
                    self.csv_file_path,
                    delimiter=self.delimiter,
                    dtype=None,
                    names=True,
                    skip_header=0,
                    max_rows=2,
                )
                if (
                    not self.target_type == "numeric"
                    and self.values_for_class_label is None
                ):
                    # LOADS THE FULL FILE INTO THE MEMORY
                    data = np.genfromtxt(
                        self.csv_file_path,
                        delimiter=self.delimiter,
                        dtype=None,
                        names=True,
                        skip_header=1 if skip_header else 0,
                    )
                    y = data[data.dtype.names[self.class_index]]
                    self.values_for_class_label = [str(value) for value in np.unique(y)]
                for i, data_info in enumerate(data.dtype.descr):
                    column_name, data_type = data_info
                    if (
                        self.values_for_nominal_features.get(i) is not None
                    ):  # i is in nominal feature keys
                        self.dtypes.append((column_name, "str"))
                    else:
                        self.dtypes.append((column_name, data_type))
            else:  # need to infer data definitions
                # LOADS THE FULL FILE INTO THE MEMORY
                data = np.genfromtxt(
                    self.csv_file_path,
                    delimiter=self.delimiter,
                    dtype=None,
                    names=True,
                    skip_header=1 if skip_header else 0,
                )
                self.dtypes = data.dtype
                if (
                    not self.target_type == "numeric"
                    and self.values_for_class_label is None
                ):
                    y = data[data.dtype.names[self.class_index]]
                    self.values_for_class_label = [str(value) for value in np.unique(y)]
        else:  # data definition for each column are provided
            self.dtypes = dtypes

        self.total_number_of_lines = 0
        if self.skip_header:
            self.n_lines_to_skip = 1
        else:
            row1_data = np.genfromtxt(
                self.csv_file_path,
                delimiter=self.delimiter,
                dtype=None,
                names=True,
                skip_header=0,
                max_rows=1,
            )
            row2_data = np.genfromtxt(
                self.csv_file_path,
                delimiter=self.delimiter,
                dtype=None,
                names=True,
                skip_header=1,
                max_rows=1,
            )
            if row1_data.dtype.names != row2_data.dtype.names:
                self.n_lines_to_skip = 1
            else:
                self.n_lines_to_skip = 0

        self.__moa_stream_with_only_header, self.moa_header = (
            _init_moa_stream_and_create_moa_header(
                number_of_instances=1,  # we only need this to initialize the MOA header
                feature_names=[data_info[0] for data_info in self.dtypes],
                values_for_nominal_features=self.values_for_nominal_features,
                values_for_class_label=self.values_for_class_label,
                dataset_name=dataset_name,
                target_attribute_name=self.target_attribute_name,
                target_type=self.target_type,
            )
        )

        self.schema = Schema(moa_header=self.moa_header)
        self.count_number_of_lines()

    def count_number_of_lines(self):
        with open(self.csv_file_path, "r") as file:
            for line in file:
                # Process each line here
                self.total_number_of_lines += 1

    def has_more_instances(self):
        return self.total_number_of_lines > self.n_lines_to_skip

    def next_instance(self) -> _AnyInstance:
        if not self.has_more_instances():
            return None
        # skip header
        data = np.genfromtxt(
            self.csv_file_path,
            delimiter=self.delimiter,
            dtype=self.dtypes,
            names=None,
            skip_header=self.n_lines_to_skip,
            max_rows=1,
        )
        self.n_lines_to_skip += 1

        y = rfn.structured_to_unstructured(data[[data.dtype.names[self.class_index]]])[
            0
        ]
        # X = data[[item for item in data.dtype.names if item != data.dtype.names[self.class_index]]].view('f4')
        X = rfn.structured_to_unstructured(
            data[
                [
                    item
                    for item in data.dtype.names
                    if item != data.dtype.names[self.class_index]
                ]
            ]
        )

        if self.schema.is_classification():
            return LabeledInstance.from_array(self.schema, X, int(y))
        elif self.schema.is_regression():
            return RegressionInstance.from_array(self.schema, X, float(y))
        else:
            raise ValueError(
                "Unknown machine learning task must be a regression or "
                "classification task"
            )

    def get_schema(self):
        return self.schema

    def restart(self):
        self.total_number_of_lines = 0
        self.n_lines_to_skip = 1 if self.skip_header else 0


class ConcatStream(Stream[_AnyInstance]):
    """Concatenate multiple streams into a single stream.

    When the end of a stream is reached, the next stream in the list is used.

    >>> from capymoa.stream import ConcatStream, NumpyStream
    >>> import numpy as np
    >>> X1 = np.array([[1, 2, 3]])
    >>> X2 = np.array([[4, 5, 6]])
    >>> y1 = np.array([0])
    >>> y2 = np.array([0])
    >>> stream1 = NumpyStream(X1, y1)
    >>> stream2 = NumpyStream(X2, y2)
    >>> concat_stream = ConcatStream([stream1, stream2])
    >>> for instance in concat_stream:
    ...     print(instance)
    LabeledInstance(
        Schema(No_Name),
        x=[1. 2. 3.],
        y_index=0,
        y_label='0'
    )
    LabeledInstance(
        Schema(No_Name),
        x=[4. 5. 6.],
        y_index=0,
        y_label='0'
    )

    """

    def __init__(self, streams: Sequence[Stream]):
        """Construct a ConcatStream object from a list of streams.
        :param streams: A list of streams to chain together.
        """
        super().__init__()
        # Check that all streams have the same schema.
        schema = streams[0].get_schema()
        for stream in streams[1:]:
            if stream.get_schema() != schema:
                raise ValueError("All streams must have the same schema.")

        self.streams = streams
        self.stream_index = 0

        self._length: Optional[int] = None
        if all(hasattr(stream, "__len__") for stream in streams):
            self._length = sum(len(stream) for stream in streams)

    def has_more_instances(self) -> bool:
        """Return ``True`` if the stream have more instances to read."""
        return any(s.has_more_instances() for s in self.streams[self.stream_index :])

    def next_instance(self) -> _AnyInstance:
        """Return the next instance in the stream.

        :raises ValueError: If the machine learning task is neither a regression
            nor a classification task.
        :return: A labeled instances or a regression depending on the schema.
        """
        stream = self.streams[self.stream_index]
        if not stream.has_more_instances():
            self.stream_index += 1

        if not self.has_more_instances():
            raise StopIteration()

        return self.streams[self.stream_index].next_instance()

    def get_schema(self) -> Schema:
        """Return the schema of the stream."""
        return self.streams[self.stream_index].get_schema()

    def get_moa_stream(self) -> Optional[InstanceStream]:
        """Get the MOA stream object if it exists."""
        return self.streams[self.stream_index].get_moa_stream()

    def restart(self):
        """Restart the stream to read instances from the beginning."""
        for stream in self.streams:
            stream.restart()
        self.stream_index = 0

    def __len__(self) -> None:
        """Return the length of the stream."""
        if self._length is None:
            raise RuntimeError(
                "Only supports ``len()`` if contained streams have a length."
            )
        return self._length
