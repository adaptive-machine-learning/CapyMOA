import typing
from typing import Dict, Optional, Sequence

import numpy as np
from numpy.lib import recfunctions as rfn

from com.yahoo.labs.samoa.instances import (
    Attribute,
    DenseInstance,
    Instances,
    InstancesHeader,
)
from moa.core import FastVector, InstanceExample
from moa.streams import ArffFileStream, InstanceStream

# MOA/Java imports

from capymoa.instance import (
    Instance,
    LabeledInstance,
    RegressionInstance,
)


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
        assert (
            moa_header.numOutputAttributes() == 1
        ), "Only one output attribute is supported."

        self._moa_header = moa_header
        # Internally, we store the number of attributes + the class/target. This is because MOA methods expect the numAttributes
        # to also account for the class/target.
        self._regression = not self._moa_header.outputAttribute(1).isNominal()
        self._label_values: Optional[Sequence[str]] = None
        self._label_index_map: Optional[Dict[str, int]] = None

        if not self._regression:
            values = self._moa_header.outputAttribute(1).getAttributeValues()
            self._label_values = list(map(str, values))
            self._label_index_map = {
                label: i for i, label in enumerate(self._label_values)
            }

        # TODO: might want to iterate over the other attributes and create a dictionary representation for the nominal attributes.
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
        enforce_regression=False,
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
        ...     enforce_regression=True)
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
        :param enforce_regression: If True, the schema is interpreted as a
            regression problem. Default is False.
        :return CayMOA Schema: Initialized CapyMOA Schema which contain all
            necessary attribute information for all features and the class label
        """
        _, moa_header = _init_moa_stream_and_create_moa_header(
            feature_names=feature_names,
            values_for_nominal_features=values_for_nominal_features,
            values_for_class_label=values_for_class_label,
            dataset_name=dataset_name,
            target_attribute_name=target_attribute_name,
            enforce_regression=enforce_regression,
        )
        return Schema(moa_header=moa_header)

    def __repr__(self) -> str:
        """Return a string representation of the schema as an ARFF header."""
        return str(self)

    def __str__(self):
        """Return a string representation of the schema as an ARFF header."""
        return str(self._moa_header.toString()).strip()


class Stream:
    """A datastream that can be learnt instance by instance."""

    # TODO: A problem in stream is that is has lots of conditional logic to
    # support a variety of ways to create a Stream object. This makes the code
    # harder to understand and maintain. We should consider refactoring this
    # with a abstract base class and subclasses for each type of stream.

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

    def __str__(self):
        """Return the name of the datastream from the schema."""
        return str(self.schema._moa_header.getRelationName()).replace(" ", "")

    def CLI_help(self) -> str:
        """Return cli help string for the stream."""
        return str(
            self.moa_stream.getOptions().getHelpString()
            if self.moa_stream is not None
            else ""
        )

    def has_more_instances(self) -> bool:
        """Return `True` if the stream have more instances to read."""
        return self.moa_stream.hasMoreInstances()

    def next_instance(self) -> typing.Union[LabeledInstance, RegressionInstance]:
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


class ARFFStream(Stream):
    """A datastream originating from an ARFF file."""

    def __init__(self, path: str, CLI: Optional[str] = None):
        """Construct an ARFFStream object from a file path.

        :param path: A filepath
        :param CLI: Additional command line arguments to pass to the MOA stream.
        """
        moa_stream = ArffFileStream(path, -1)
        super().__init__(moa_stream=moa_stream, CLI=CLI)


class NumpyStream(Stream):
    """A datastream originating from a numpy array."""

    # This class is more complex than ARFFStream because it needs to read and convert the CSV to an ARFF in memory.
    # enforce_regression overrides the default behavior of inferring whether the data represents a regression or classification task.
    # TODO: class_index is currently ignored while reading the file in numpy_to_ARFF

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name="No_Name",
        feature_names=None,
        target_name=None,
        enforce_regression=False,
    ):
        """Construct a NumpyStream object from a numpy array.

        :param X: Numpy array of shape (n_samples, n_features) with the feature values
        :param y: Numpy array of shape (n_samples,) with the target values
        :param dataset_name: The name to give to the datastream, defaults to "No_Name"
        :param feature_names: The names given to the features, defaults to None
        :param target_name: The name given to target values, defaults to None
        :param enforce_regression: Should it be used as regression, defaults to False
        """
        self.current_instance_index = 0

        self.arff_instances_data, self.arff_instances_header, class_labels = (
            _numpy_to_ARFF(
                X,
                y,
                dataset_name,
                feature_names=feature_names,
                target_name=target_name,
                enforce_regression=enforce_regression,
            )
        )

        self.schema = Schema(moa_header=self.arff_instances_header)

        super().__init__(schema=self.schema, CLI=None, moa_stream=None)

    def has_more_instances(self):
        return self.arff_instances_data.numInstances() > self.current_instance_index

    def next_instance(self) -> Instance:
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

    def get_moa_stream(self):
        raise ValueError("Not a moa_stream, a numpy read file")

    def restart(self):
        self.current_instance_index = 0


## TODO (20/10/2023): Add logic to interpret nominal values (strings) in the class label.
## TODO: add extra fluffiness like allowing to not have a header for the csv (then we need to create names for each column).
## TODO: if no name is given for the dataset_name, we can use the file name from the csv.
## TODO: implement class_index logic when reading from a CSV.
# TODO: path_to_csv_or_arff should be a positional argument because it is required.
def stream_from_file(
    path_to_csv_or_arff: str = None,
    dataset_name: str = "NoName",
    enforce_regression: bool = False,
) -> Stream:
    """Create a datastream from a csv or arff file.

    >>> from capymoa.stream import stream_from_file
    >>> stream = stream_from_file("data/electricity_tiny.csv", dataset_name="Electricity")
    >>> stream.next_instance()
    LabeledInstance(
        Schema(Electricity),
        x=ndarray(..., 6),
        y_index=1,
        y_label='1'
    )
    >>> stream.next_instance().x
    array([0.021277, 0.051699, 0.415055, 0.003467, 0.422915, 0.414912])

    :param path_to_csv_or_arff: A file path to a CSV or ARFF file.
    :param dataset_name: A descriptive name given to the dataset, defaults to "NoName"
    :param enforce_regression: When working with a CSV file, this parameter
        allows the user to force the data to be interpreted as a regression
        problem. Defaults to False.
    """
    assert path_to_csv_or_arff is not None, "A file path must be provided."
    if path_to_csv_or_arff.endswith(".arff"):
        # Delegate to the ARFFFileStream object within ARFFStream to actually read the file.
        return ARFFStream(path=path_to_csv_or_arff)
    elif path_to_csv_or_arff.endswith(".csv"):
        # TODO: Upgrade to CSVStream once its faster and notebook tests don't fail
        x_features = np.genfromtxt(path_to_csv_or_arff, delimiter=",", skip_header=1)
        targets = x_features[:, -1]
        targets = targets.astype(int)
        x_features = x_features[:, :-1]
        return NumpyStream(
            x_features,
            targets,
            dataset_name=dataset_name,
            enforce_regression=enforce_regression,
        )


def _numpy_to_ARFF(
    X,
    y,
    dataset_name="No_Name",
    feature_names=None,
    target_name=None,
    enforce_regression=False,
):
    """Converts a numpy X and y into a ARFF format. The code infers whether it is a classification or regression problem
    based on the y type. If y[0] is a double, then assumes regression (thus output will be numeric) otherwise assume
    it as a classifiation problem. If the user desires to "force" regression, then set enforce_regression=True
    """
    number_of_instances = X.shape[0]
    enforce_regression = (
        True if enforce_regression else np.issubdtype(type(y[0]), np.double)
    )
    class_labels = (
        None if enforce_regression else [str(value) for value in np.unique(y)]
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
        enforce_regression=enforce_regression,
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
    enforce_regression=False,
):
    """Initialize a moa stream with number_of_instances capacity and create a mao header which contains all the necessary
     attribute information.

     Note: Each instance is not added to the moa_stream.

    :param number_of_instances: number of instances in the stream
    :param feature_names: a list containing names of features. if none sets a default name
    :param values_for_nominal_features: possible values of each nominal feature.
    e.g {i: [1,2,3], k: [Aa, BB]}. Key is integer. Values are turned into strings
    :param values_for_class_label: possible values for class label. Values are turned into strings
    :param dataset_name: name of the dataset. Defaults to "No_Name"
    :param target_attribute_name: name for the target/class attribute
    :param enforce_regression: If True assumes the problem as a regression problem

    :return moa_stream: initialized moa stream with capacity number_of_instances.
    :return moa_header: initialized moa header which contain all necessary attribute information for all features and the class label

    Sample code to get relevant information from two Numpy arrays: X[rows][features] and y[rows]

    feature_names = [f"attrib_{i}" for i in range(X.shape[1])]

    number_of_instances = X.shape[0]

    values_for_class_label = [str(value) for value in np.unique(y)]

    enforce_regression = np.issubdtype(type(y[0]), np.double)

    """
    attributes = FastVector()
    # Attribute("name") will create a numeric attribute; Attribute("name", array_of_values) will create a nominal attribute
    if feature_names is None:
        raise ValueError("feature_names are None")

    for name in feature_names:
        if name in values_for_nominal_features:
            attribute = _create_nominal_attribute(
                attribute_name=name,
                possible_values=values_for_nominal_features.get(name),
            )
        else:
            attribute = Attribute(name)
        attributes.addElement(attribute)

    if enforce_regression:
        if target_attribute_name is None:
            attributes.addElement(Attribute("target"))
        else:
            attributes.addElement(Attribute(target_attribute_name))
    else:
        if values_for_class_label is None:
            raise ValueError(
                "values_for_class_label are None and enforce_regression is False. Looks like a regression problem?"
            )
        else:
            class_attribute = _create_nominal_attribute(
                attribute_name=(
                    "class" if target_attribute_name is None else target_attribute_name
                ),
                possible_values=values_for_class_label,
            )
            attributes.addElement(class_attribute)

    moa_stream = Instances(dataset_name, attributes, number_of_instances)
    # set last index for class index
    moa_stream.setClassIndex(attributes.size() - 1)
    # create stream header
    moa_header = InstancesHeader(moa_stream)
    # moa_header.setClassIndex(moa_header.classIndex())
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


class CSVStream(Stream):
    def __init__(
        self,
        csv_file_path,
        dtypes: list = None,  # [('column1', np.float64), ('column2', np.int32), ('column3', np.float64), ('column3', str)] reads nomonal attributes as str
        values_for_nominal_features={},  # {i: [1,2,3], k: [Aa, BB]}. Key is integer. Values are turned into strings
        class_index: int = -1,
        values_for_class_label: list = None,
        target_attribute_name=None,
        enforce_regression=False,
        skip_header: bool = False,
        delimiter=",",
    ):
        self.csv_file_path = csv_file_path
        self.values_for_nominal_features = values_for_nominal_features
        self.class_index = class_index
        self.values_for_class_label = values_for_class_label
        self.target_attribute_name = target_attribute_name
        self.enforce_regression = enforce_regression
        self.skip_header = skip_header
        self.delimiter = delimiter

        self.dtypes = (
            []
        )  # [('column1', np.float64), ('column2', np.int32), ('column3', np.float64), ('column3', str)] reads nomonal attributes as str
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
                if not self.enforce_regression and self.values_for_class_label is None:
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
                if not self.enforce_regression and self.values_for_class_label is None:
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
                dataset_name="CSVDataset",
                target_attribute_name=self.target_attribute_name,
                enforce_regression=self.enforce_regression,
            )
        )

        self.schema = Schema(moa_header=self.moa_header)
        super().__init__(schema=self.schema, CLI=None, moa_stream=None)
        self.count_number_of_lines()

    def count_number_of_lines(self):
        with open(self.csv_file_path, "r") as file:
            for line in file:
                # Process each line here
                self.total_number_of_lines += 1

    def has_more_instances(self):
        return self.total_number_of_lines > self.n_lines_to_skip

    def next_instance(self):
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
        # np.genfromtxt() returns a structured https://numpy.org/doc/stable/user/basics.rec.html#structured-arrays
        self.n_lines_to_skip += 1

        # data = np.expand_dims(data, axis=0)
        # y = data[[data.dtype.names[self.class_index]]].view('i4')
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
            return LabeledInstance.from_array(self.schema, X, y)
        elif self.schema.is_regression():
            return RegressionInstance.from_array(self.schema, X, y)
        else:
            raise ValueError(
                "Unknown machine learning task must be a regression or "
                "classification task"
            )

    def get_schema(self):
        return self.schema

    def get_moa_stream(self):
        raise ValueError("Not a moa_stream, a numpy read file")

    def restart(self):
        self.total_number_of_lines = 0
        self.n_lines_to_skip = 1 if self.skip_header else 0
