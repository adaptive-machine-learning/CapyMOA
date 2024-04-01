import re
import typing
from typing import Dict, Optional, Sequence

import numpy as np
from com.yahoo.labs.samoa.instances import (
    Attribute,
    DenseInstance,
    Instances,
    InstancesHeader,
)
from moa.core import FastVector, InstanceExample
from moa.streams import ArffFileStream
from moa.streams import ConceptDriftStream as MOA_ConceptDriftStream

# MOA/Java imports
from moa.streams.generators import RandomTreeGenerator as MOA_RandomTreeGenerator
from moa.streams.generators import SEAGenerator as MOA_SEAGenerator

from capymoa.stream.instance import (
    Instance,
    LabeledInstance,
    RegressionInstance,
)


class Schema:
    """
    The schema of a stream, defines what instances look like. It contains the
    attribute names and the possible values nominal attributes. The schema is
    crucial for a learner to know how to interpret instances correctly.

    When working with datasets built into CapyMOA (see :mod:`capymoa.datasets`)
    and ARFF files, the schema is automatically created. However, in some cases
    you might want to create a schema manually. This can be done using the
    :meth:`from_custom` method.
    """

    def __init__(self, moa_header: InstancesHeader):
        """Construct a schema by wrapping a Java MOA `InstancesHeader`. To
        create a schema without one use the :meth:`from_custom` method.

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
        assert (
            self.is_classification()
        ), "Should only be called for classification problems."

    def get_label_values(self) -> Sequence[str]:
        """Return the possible values for the class label. Raises an error if
        the problem is a regression problem."""
        self._assert_classification()
        return self._label_values

    def get_label_indexes(self) -> Sequence[int]:
        """Return the possible indexes for the class label. Raises an error if
        the problem is a regression problem."""
        self._assert_classification()
        return list(range(self.get_num_classes()))

    def get_value_for_index(self, y_index: Optional[int]) -> Optional[str]:
        """Return the value for the class label index y_index. Raises an error if
        the problem is a regression problem.
        """
        self._assert_classification()
        if y_index is None:
            return None
        return self._label_values[y_index]

    def get_index_for_label(self, y: str):
        """Return the index for the class label y. Raises an error if
        the problem is a regression problem or if y is not a valid label.
        """
        self._assert_classification()
        return self._label_index_map[y]

    def get_moa_header(self) -> InstancesHeader:
        """An advanced method to get the JAVA MOA header. This is needed for
        advanced operations that are not supported by the Python wrappers (yet).
        """
        return self._moa_header

    def get_num_attributes(self):
        """Returns the number of attributes excluding the target attribute."""
        return self._moa_header.numAttributes() - self._moa_header.numOutputAttributes()

    def get_num_classes(self) -> int:
        """Return the number of possible classes. If regression, returns 1."""
        if self._regression:
            return 1
        return len(self._label_values)

    def is_regression(self):
        """Returns True if the problem is a regression problem."""
        return self._regression

    def is_classification(self):
        """Returns True if the problem is a classification problem."""
        return not self._regression

    def is_y_index_in_range(self, y_index: int):
        """Returns True if the y_index is in the range of the class label indexes."""
        return 0 <= y_index < self.get_num_classes()

    @property
    def dataset_name(self):
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
        """
        Create a CapyMOA Schema which contains all the necessary attribute information.

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
        """Returns a string representation of the schema as an ARFF header."""
        return str(self)

    def __str__(self):
        return str(self._moa_header.toString()).strip()


class Stream:
    def __init__(self, schema=None, CLI=None, moa_stream=None):
        self.schema = schema
        self.CLI = CLI

        self.moa_stream = moa_stream

        if self.moa_stream is None:
            pass
            # self.moa_stream = MOA_RandomTreeGenerator()

        if self.CLI is not None:
            if self.moa_stream is not None:
                self.moa_stream.getOptions().setViaCLIString(CLI)
            else:
                raise RuntimeError("Must provide a moa_stream to set via CLI.")

        if self.moa_stream is not None:
            # Must call this method exactly here, because prepareForUse invoke the method to initialize the
            # header file of the stream (synthetic ones)
            self.moa_stream.prepareForUse()
        else:
            # NumpyStream or PytorchStream: does not have a CLI string on moa_stream
            pass

        if self.schema is None:
            if self.moa_stream is not None:
                self.schema = Schema(moa_header=self.moa_stream.getHeader())
            else:
                raise RuntimeError(
                    "Must provide a moa_stream to initialize the Schema."
                )

    def __str__(self):
        return str(self.schema._moa_header.getRelationName()).replace(" ", "")

    def CLI_help(self):
        return str(
            self.moa_stream.getOptions().getHelpString()
            if self.moa_stream is not None
            else ""
        )

    def has_more_instances(self):
        return self.moa_stream.hasMoreInstances()

    def next_instance(self) -> typing.Union[LabeledInstance, RegressionInstance]:
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

    def get_schema(self):
        return self.schema

    def get_moa_stream(self):
        return self.moa_stream

    def restart(self):
        self.moa_stream.restart()


class ARFFStream(Stream):
    """
    Just delegates the file reading to the ArffFileStream from MOA.
    TODO: The CLI can be used to pass the path and class_index, just for consistency with other methods...
    """

    def __init__(self, schema=None, CLI=None, path="", class_index=-1):
        moa_stream = ArffFileStream(path, class_index)
        super().__init__(schema=schema, CLI=CLI, moa_stream=moa_stream)


class NumpyStream(Stream):
    """
    This class is more complex than ARFFStream because it needs to read and convert the CSV to an ARFF in memory.
    enforce_regression overrides the default behavior of inferring whether the data represents a regression or classification task.
    TODO: class_index is currently ignored while reading the file in numpy_to_ARFF
    """

    def __init__(
        self,
        X,
        y,
        class_index=-1,
        dataset_name="No_Name",
        feature_names=None,
        target_name=None,
        enforce_regression=False,
    ):
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


# TODO: put this function on a 'utils' module
def _get_moa_creation_CLI(moa_object):
    moa_class_id = str(moa_object.getClass().getName())
    moa_class_id_parts = moa_class_id.split(".")
    moa_stream_str = f"{moa_class_id_parts[-2]}.{moa_class_id_parts[-1]}"

    moa_cli_creation = str(moa_object.getCLICreationString(moa_object.__class__))
    CLI = moa_cli_creation.split(" ", 1)

    if len(CLI) > 1 and len(CLI[1]) > 1:
        moa_stream_str = f"({moa_stream_str} {CLI[1]})"

    return moa_stream_str


class RandomTreeGenerator(Stream):
    def __init__(
        self,
        schema=None,
        CLI=None,
        instance_random_seed=1,
        tree_random_seed=1,
        num_classes=2,
        num_nominals=5,
        num_numerics=5,
        num_vals_per_nominal=5,
        max_tree_depth=5,
        first_leaf_level=3,
        leaf_fraction=0.15,
    ):
        self.moa_stream = MOA_RandomTreeGenerator()

        self.CLI = CLI
        if self.CLI is None:
            self.instance_random_seed = instance_random_seed
            self.tree_random_seed = tree_random_seed
            self.num_classes = num_classes
            self.num_nominals = num_nominals
            self.num_numerics = num_numerics
            self.num_vals_per_nominal = num_vals_per_nominal
            self.max_tree_depth = max_tree_depth
            self.first_leaf_level = first_leaf_level
            self.leaf_fraction = leaf_fraction

            self.CLI = f"-i {instance_random_seed} -r {self.tree_random_seed} \
               -c {self.num_classes} -o {self.num_nominals} -u {self.num_numerics} -v {self.num_vals_per_nominal} \
               -d {max_tree_depth} -l {first_leaf_level} -f {leaf_fraction}"

        super().__init__(schema=schema, CLI=self.CLI, moa_stream=self.moa_stream)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (
                f"tree_random_seed={self.tree_random_seed}"
                if self.tree_random_seed != 1
                else None
            ),
            f"num_classes={self.num_classes}" if self.num_classes != 2 else None,
            f"num_nominals={self.num_nominals}" if self.num_nominals != 5 else None,
            f"num_numerics={self.num_numerics}" if self.num_numerics != 5 else None,
            (
                f"num_vals_per_nominal={self.num_vals_per_nominal}"
                if self.num_vals_per_nominal != 5
                else None
            ),
            (
                f"max_tree_depth={self.max_tree_depth}"
                if self.max_tree_depth != 5
                else None
            ),
            (
                f"first_leaf_level={self.first_leaf_level}"
                if self.first_leaf_level != 3
                else None
            ),
            (
                f"leaf_fraction={self.leaf_fraction}"
                if self.leaf_fraction != 0.15
                else None
            ),
        ]

        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"RTG({', '.join(non_default_attributes)})"


class SEA(Stream):
    def __init__(
        self,
        schema=None,
        CLI=None,
        instance_random_seed=1,
        function=1,
        balance_classes=False,
        noise_percentage=10,
    ):
        self.moa_stream = MOA_SEAGenerator()

        self.CLI = CLI
        if self.CLI is None:
            self.instance_random_seed = instance_random_seed
            self.function = function
            self.balance_classes = balance_classes
            self.noise_percentage = noise_percentage

            self.CLI = f"-i {instance_random_seed} -f {self.function} \
               {'-b' if self.balance_classes else ''} -p {self.noise_percentage}"

        super().__init__(schema=schema, CLI=self.CLI, moa_stream=self.moa_stream)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            f"function={self.function}",
            f"balance_classes={self.balance_classes}" if self.balance_classes else None,
            (
                f"noise_percentage={self.noise_percentage}"
                if self.noise_percentage != 10
                else None
            ),
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"SEA({', '.join(non_default_attributes)})"


############################################################################################################
############################################### DRIFT STREAM ###############################################
############################################################################################################


class DriftStream(Stream):
    def __init__(self, schema=None, CLI=None, moa_stream=None, stream=None):
        # If moa_stream is specified, just instantiate it directly. We can check whether it is a ConceptDriftStream object or not.
        # if composite_stream is set, then the ConceptDriftStream object is build according to the list of Concepts and Drifts specified in composite_stream
        # ```moa_stream``` and ```CLI``` allow the user to specify the stream using a ConceptDriftStream from MOA alongside its CLI. However, in the future we might remove that functionality to make the code simpler.

        self.stream = stream
        self.drifts = []
        moa_concept_drift_stream = MOA_ConceptDriftStream()

        if CLI is None:
            stream1 = None
            stream2 = None
            drift = None

            CLI = ""
            for component in self.stream:
                if isinstance(component, Stream):
                    if stream1 is None:
                        stream1 = component
                    else:
                        stream2 = component
                        if drift is None:
                            raise ValueError(
                                "A Drift object must be specified between two Stream objects."
                            )

                        CLI += f" -d {_get_moa_creation_CLI(stream2.moa_stream)} -w {drift.width} -p {drift.position} -r {drift.random_seed} -a {drift.alpha}"
                        CLI = CLI.replace(
                            "streams.", ""
                        )  # got to remove package name from streams.ConceptDriftStream

                        stream1 = Stream(moa_stream=moa_concept_drift_stream, CLI=CLI)
                        stream2 = None

                elif isinstance(component, Drift):
                    # print(component)
                    drift = component
                    self.drifts.append(drift)
                    CLI = f" -s {_get_moa_creation_CLI(stream1.moa_stream)} "

            # print(CLI)
            # CLI = command_line
            moa_stream = moa_concept_drift_stream
        else:
            # [EXPERIMENTAL]
            # If the user is attempting to create a DriftStream using a MOA CLI, we need to derive the Drift meta-data through the CLI.
            # The number of ConceptDriftStream occurrences corresponds to the number of Drifts.
            # +1 because we expect at least one drift from an implit ConceptDriftStream (i.e. not shown in the CLI because it is the moa_stream object)
            num_drifts = CLI.count("ConceptDriftStream") + 1

            # This is a best effort in obtaining the meta-data from a MOA CLI.
            # Notice that if the width (-w) or position (-p) are not explicitly shown in the CLI it is difficult to infer them.
            pattern_position = r"-p (\d+)"
            pattern_width = r"-w (\d+)"
            matches_position = re.findall(pattern_position, CLI)
            matches_width = re.findall(pattern_width, CLI)

            for i in range(0, num_drifts):
                if len(matches_width) == len(matches_position):
                    self.drifts.append(
                        Drift(
                            position=int(matches_position[i]),
                            width=int(matches_width[i]),
                        )
                    )
                else:
                    # Assuming the width of the drifts (or at least one) are not show, implies that the default value (1000) was used.
                    self.drifts.append(
                        Drift(position=int(matches_position[i]), width=1000)
                    )

        super().__init__(schema=schema, CLI=CLI, moa_stream=moa_stream)

    def get_num_drifts(self):
        return len(self.drifts)

    def get_drifts(self):
        return self.drifts

    def __str__(self):
        if self.stream is not None:
            return ",".join(str(component) for component in self.stream)
        # If the stream was defined using the backward compatility (MOA object + CLI) then there are no Stream objects in stream.
        # Best we can do is return the CLI directly.
        return f"ConceptDriftStream {self.CLI}"


# TODO: remove width from the base Drift class and keep it only on the GradualDrift


class Drift:
    """
    Represents a concept drift in a DriftStream.

    Parameters:
        - position (int): The location of the drift in terms of the number of instances processed prior to it occurring.
        - width (int, optional): The size of the window of change. A width of 0 or 1 corresponds to an abrupt drift.
            Default is 0.
        - alpha (float, optional): The grade of change (See 2.7.1 Concept Drift Framework in [1]). Default is 0.0.
        - random_seed (int, optional): Seed for random number generation (See 2.7.1 Concept Drift Framework [1]). Default is 1.

    References:
    [1] Bifet, Albert, et al. "Data stream mining: a practical approach." COSI (2011).
    """

    def __init__(self, position, width=0, alpha=0.0, random_seed=1):
        self.width = width
        self.position = position
        self.alpha = alpha
        self.random_seed = random_seed

    def __str__(self):
        drift_kind = "GradualDrift"
        if self.width == 0 or self.width == 1:
            drift_kind = "AbruptDrift"
        attributes = [
            f"position={self.position}",
            f"width={self.width}" if self.width not in [0, 1] else None,
            f"alpha={self.alpha}" if self.alpha != 0.0 else None,
            f"random_seed={self.random_seed}" if self.random_seed != 1 else None,
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"{drift_kind}({', '.join(non_default_attributes)})"


class GradualDrift(Drift):
    def __init__(
        self, position=None, width=None, start=None, end=None, alpha=0.0, random_seed=1
    ):
        # since python doesn't allow overloading functions we need to check if the user hasn't defined position + width and start+end.
        if (
            position is not None
            and width is not None
            and start is not None
            and end is not None
        ):
            raise ValueError(
                "Either use start and end OR position and width to determine the location of the gradual drift."
            )

        if start is None and end is None:
            self.width = width
            self.position = position
            self.start = int(position - width / 2)
            self.end = int(position + width / 2)
        elif position is None and width is None:
            self.start = start
            self.end = end
            self.width = end - start
            print(width)
            self.position = int((start + end) / 2)

        self.alpha = alpha
        self.random_seed = random_seed

        super().__init__(
            position=self.position, random_seed=self.random_seed, width=self.width
        )

    def __str__(self):
        attributes = [
            f"position={self.position}",
            f"start={self.start}",
            f"end={self.end}",
            f"width={self.width}",
            f"alpha={self.alpha}" if self.alpha != 0.0 else None,
            f"random_seed={self.random_seed}" if self.random_seed != 1 else None,
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"GradualDrift({', '.join(non_default_attributes)})"


class AbruptDrift(Drift):
    def __init__(self, position, random_seed=1):
        self.position = position
        self.random_seed = random_seed

        super().__init__(position=position, random_seed=random_seed)

    def __str__(self):
        attributes = [
            f"position={self.position}",
            f"random_seed={self.random_seed}" if self.random_seed != 1 else None,
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"AbruptDrift({', '.join(non_default_attributes)})"


## TODO (20/10/2023): Add logic to interpret nominal values (strings) in the class label.
## TODO: add extra fluffiness like allowing to not have a header for the csv (then we need to create names for each column).
## TODO: if no name is given for the dataset_name, we can use the file name from the csv.
## TODO: implement class_index logic when reading from a CSV.
def stream_from_file(
    schema=None,
    path_to_csv_or_arff="",
    class_index=-1,
    dataset_name="NoName",
    enforce_regression=False,
):
    if path_to_csv_or_arff.endswith(".arff"):
        # Delegate to the ARFFFileStream object within ARFFStream to actually read the file.
        return ARFFStream(path=path_to_csv_or_arff, class_index=class_index)
    elif path_to_csv_or_arff.endswith(".csv"):
        # Do the file reading here.
        _data = np.genfromtxt(
            path_to_csv_or_arff, delimiter=",", skip_header=1
        )  # Assuming a header row

        # Extract the feature data (all columns except the last one) and target data (last column)
        # TODO: class_index logic should appear in here.
        X = _data[:, :-1]
        y = _data[:, -1]

        # Extract the header from the CSV file (first row)
        with open(path_to_csv_or_arff, "r") as file:
            header = file.readline().strip().split(",")

        return NumpyStream(
            X=X,
            y=y.astype(int),
            dataset_name=dataset_name,
            feature_names=header[:-1],
            target_name=header[-1],
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
    """
    Converts a numpy X and y into a ARFF format. The code infers whether it is a classification or regression problem
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


def create_nominal_attribute(attribute_name=None, possible_values: list = None):
    value_list = FastVector()
    for value in possible_values:
        value_list.addElement(str(value))
    return Attribute(attribute_name, value_list)


"""

"""


def _init_moa_stream_and_create_moa_header(
    number_of_instances: int = 100,
    feature_names: list = None,
    values_for_nominal_features={},
    values_for_class_label: list = None,
    dataset_name="No_Name",
    target_attribute_name=None,
    enforce_regression=False,
):
    """
    Initialize a moa stream with number_of_instances capacity and create a mao header which contains all the necessary
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
            attribute = create_nominal_attribute(
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
            class_attribute = create_nominal_attribute(
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
