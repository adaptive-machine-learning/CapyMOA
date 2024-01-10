# Create the JVM and add the MOA jar to the classpath
from prepare_jpype import start_jpype

start_jpype()

# Python imports
import numpy as np

# MOA/Java imports
from moa.streams.generators import RandomTreeGenerator as MOA_RandomTreeGenerator
from moa.streams import ArffFileStream
from moa.core import FastVector, InstanceExample, Example
from com.yahoo.labs.samoa.instances import (
    Instances,
    InstancesHeader,
    Attribute,
    DenseInstance,
)


# TODO: STATIC METHOD TO CREATE A SCHEMA USING A MOA_HEADER. (e.g. withMOAHeader...)
class Schema:
    """
    This class is a wrapper for the MOA header, but it can be set up in Python directly by specifying the labels attribute.
    If moa_header is specified, then it overrides everything else.
    In the future, we might want to have a way to specify the values for nominal attributes as well, so far just the labels.
    The number of attributes is instrumental for Evaluators that need it, such as adjusted coefficient of determination.
    """

    def __init__(self, moa_header=None, labels=None, num_attributes=1):
        self.moa_header = moa_header
        self.label_values = labels
        self.label_indexes = None
        # Internally, we store the number of attributes + the class/target. This is because MOA methods expect the numAttributes
        # to also account for the class/target.
        self.num_attributes_including_output = num_attributes + 1

        self.regression = False
        if self.moa_header is not None:
            # TODO: might want to iterate over the other attributes and create a dictionary representation for the nominal attributes.
            # There should be a way to configure that manually like setting the self.labels instead of using a MOA header.
            if self.moa_header.outputAttribute(1).isNominal():
                # Important: a Java.String is different from a Python str, so it is important to str(*) before storing the values.
                self.label_values = [
                    str(g)
                    for g in self.moa_header.outputAttribute(1).getAttributeValues()
                ]
            else:
                # This is a regression task, there are no label values.
                self.regression = True
            # The numAttributes in MOA also account for the class label.
            self.num_attributes_including_output = self.moa_header.numAttributes()
        # else logic: the label_values must be set, so that the first time the get_label_indexes is invoked, they are correctly created.

    def get_label_values(self):
        if self.label_values is None:
            return None
        else:
            return self.label_values

    def get_label_indexes(self):
        if self.label_values is None:
            return None
        else:
            if self.label_indexes is None:
                self.label_indexes = list(range(len(self.label_values)))
            return self.label_indexes

    def get_moa_header(self):
        return self.moa_header

    def get_num_attributes(self):
        # ignoring the class/target value.
        return self.num_attributes_including_output - 1

    def get_num_classes(self):
        return len(self.get_label_indexes())

    def get_valid_index_for_label(self, y):
        if self.label_indexes is None:
            raise ValueError(
                "Schema was not properly initialised, please define a proper Schema."
            )

        # print(f"get_valid_index_for_label( y = {y} )")

        # Check of y is a string and if the labelValues contains strings.
        # print(f"isinstance {type(y)}, {type(self.label_values[0])}")
        if isinstance(y, type(self.label_values[0])):
            if y in self.label_values:
                return self.label_values.index(y)

        # If it is not a valid value, then maybe it is an index
        if y in self.label_indexes:
            return y

        # This is neither a valid label value nor a valid index.
        return None

    def is_regression(self):
        return self.regression

    def is_classification(self):
        return not self.regression


class Instance:
    """
    Wraps a MOA InstanceExample to make it easier to manipulate these objects through python.
    TODO: Add Schema and capabilities to create an instance from a non-MOA source.
    """

    # def __init__(self, MOAInstanceExample=None, schema=None, x=None, y=None):
    # 	if MOAInstanceExample is None:
    # 	self.MOAInstanceExample = MOAInstanceExample

    def __init__(self, MOAInstanceExample=None, schema=None, x=None, y=None):
        if MOAInstanceExample is not None:
            self.MOAInstanceExample = MOAInstanceExample

    def get_MOA_InstanceExample(self):
        return self.MOAInstanceExample

    def y(self):
        # return np.array(self.MOAInstanceExample.getData().classValue(), ndmin=0)
        return float(self.MOAInstanceExample.getData().classValue())

    # Assume data is numeric.
    def x(self):
        moa_instance = self.get_MOA_InstanceExample().getData()
        x_array = np.empty(moa_instance.numInputAttributes())
        for i in range(0, moa_instance.numInputAttributes()):
            x_array[i] = moa_instance.value(i)

        return x_array


class Stream:
    def __init__(self, schema=None, CLI=None, moa_stream=None):
        self.schema = schema
        self.CLI = CLI

        self.moa_stream = moa_stream

        if self.moa_stream is None:
            self.moa_stream = MOA_RandomTreeGenerator()

        if self.CLI is not None:
            self.moa_stream.getOptions().setViaCLIString(CLI)

        # Must call this method exactly here, because prepareForUse invoke the method to initialize the
        # header file of the stream (synthetic ones)
        self.moa_stream.prepareForUse()

        if self.schema is None:
            self.schema = Schema(moa_header=self.moa_stream.getHeader())

        self.moa_stream.prepareForUse()

    def __str__(self):
        return str(self.moa_stream.getHeader().getRelationName()).replace(" ", "")

    def CLI_help(self):
        return str(self.moa_stream.getOptions().getHelpString())

    def has_more_instances(self):
        return self.moa_stream.hasMoreInstances()

    def next_instance(self):
        return Instance(self.moa_stream.nextInstance())

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

        self.arff_instances_data, self.arff_instances_header = numpy_to_ARFF(
            X,
            y,
            dataset_name,
            feature_names=feature_names,
            target_name=target_name,
            enforce_regression=enforce_regression,
        )

        self.schema = Schema(moa_header=self.arff_instances_header)

        super().__init__(schema=self.schema, CLI=None, moa_stream=None)

    def has_more_instances(self):
        return self.arff_instances_data.numInstances() > self.current_instance_index

    def next_instance(self):
        # Return None if all instances have been read already.
        instance = None
        if self.has_more_instances():
            instance = self.arff_instances_data.instance(self.current_instance_index)
            self.current_instance_index += 1
        return Instance(InstanceExample(instance))

    def get_schema(self):
        return self.schema

    def get_moa_stream(self):
        raise ValueError("Not a moa_stream, a numpy read file")

    def restart(self):
        self.current_instance_index = 0


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

        # stop converting to int in here

        return NumpyStream(
            X=X,
            y=y.astype(int),
            dataset_name=dataset_name,
            feature_names=header[:-1],
            target_name=header[-1],
            enforce_regression=enforce_regression,
        )


def numpy_to_ARFF(
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
    attributes = FastVector()
    # Attribute("name") will create a numeric attribute; Attribute("name", array_of_values) will create a nominal attribute
    for attribute_index in range(X.shape[1]):
        if feature_names is None:
            attributes.addElement(Attribute(f"attrib_{attribute_index}"))
        else:
            attributes.addElement(Attribute(feature_names[attribute_index]))

    # Infer whether we have a classification (int values) or regression task.
    # Check only if the first value is a double.
    # enforce_regression overrides the inference.
    if np.issubdtype(type(y[0]), np.double) or enforce_regression:
        if target_name is None:
            attributes.addElement(Attribute("target"))
        else:
            attributes.addElement(Attribute(target_name))
    else:
        if np.issubdtype(type(y[0]), np.integer):
            classLabels = FastVector()
            unique_class_labels = np.unique(y)  # Extract unique integer values from 'y'
            for value in unique_class_labels:
                classLabels.addElement(str(value))
            if target_name is None:
                attributes.addElement(Attribute("class", classLabels))
            else:
                attributes.addElement(Attribute(target_name, classLabels))
        else:
            raise ValueError(
                "y is neither a float or an int, can't infer whether it is regression or classification"
            )

    # if it is a string, then do the unique thing and map then (create the schema manually?)

    capacity = X.shape[0]
    arff_dataset = Instances(dataset_name, attributes, capacity)

    streamHeader = InstancesHeader(arff_dataset)
    streamHeader.setClassIndex(streamHeader.numAttributes() - 1)

    for instance_index in range(X.shape[0]):
        instance = DenseInstance(streamHeader.numAttributes())

        for attribute_index in range(X.shape[1]):
            instance.setValue(attribute_index, X[instance_index, attribute_index])

        instance.setDataset(streamHeader)
        instance.setWeight(1.0)  # a default weight of 1.0
        instance.setClassValue(y[instance_index])

        arff_dataset.add(instance)

    return arff_dataset, streamHeader


# Example loading an ARFF file in python without using MOA
# from scipy.io import arff

# from io import StringIO

# content = """
# @relation foo
# @attribute width  numeric
# @attribute height numeric
# @attribute color  {red,green,blue,yellow,black}
# @data
# 5.0,3.25,blue
# 4.5,3.75,green
# 3.0,4.00,red
# """
# f = StringIO(content)

# data, meta = arff.loadarff(f)

# print(data)

# print(meta)
