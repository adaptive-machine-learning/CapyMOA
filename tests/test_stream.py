"""This module is for testing the speeds of different stream implementations."""

from functools import partial
from typing import Optional

from capymoa.exception import StreamTypeError
import numpy as np
import pytest
import torch
from com.yahoo.labs.samoa.instances import (
    InstancesHeader,
)
from moa.streams import InstanceStream
from torch.utils.data import TensorDataset

from capymoa.instance import Instance, LabeledInstance, RegressionInstance
from capymoa.stream import (
    ARFFStream,
    CSVStream,
    NumpyStream,
    Stream,
    TorchClassifyStream,
    stream_from_file,
)
from pathlib import Path

allclose = partial(np.allclose, atol=0.001, equal_nan=True)


def check_instance(instance: Instance, x: np.ndarray, target: float):
    # Verify that the java instance is created correctly
    assert instance.java_instance is not None
    instance_data = instance.java_instance.getData()
    class_index = instance_data.classIndex()  # index of class attribute
    jxy = np.array(instance_data.toDoubleArray())
    jx = np.delete(jxy, class_index)
    jy = jxy[class_index]
    assert allclose(jx, x)

    # Verify that the python instance is created correctly
    if instance.schema.is_classification():
        assert isinstance(instance, LabeledInstance)
        assert allclose(instance.x, x)
        assert allclose(instance.y_index, target)
        assert isinstance(instance.x, np.ndarray)
        assert isinstance(instance.y_index, int)
        if np.isnan(jy) or jy == -1:
            assert target == -1
            assert instance.y_label is None
            assert instance_data.classIsMissing()
        else:
            assert target != -1
            assert isinstance(instance.y_label, str)
            assert allclose(jy, target)
            assert instance_data.classValue() == target
    elif instance.schema.is_regression():
        assert isinstance(instance, RegressionInstance)
        assert isinstance(instance.x, np.ndarray)
        assert isinstance(instance.y_value, float)
        assert allclose(instance.x, x)
        assert allclose(instance.y_value, target)
        assert allclose(jy, target)
    else:
        assert False


def check_attributes(numeric_attributes, nominal_attributes, num_attributes, schema):
    assert isinstance(schema.get_moa_header(), InstancesHeader)
    assert len(schema.get_nominal_attributes()) == len(nominal_attributes)
    assert len(schema.get_numeric_attributes()) == len(numeric_attributes)
    assert schema.get_nominal_attributes() == nominal_attributes
    assert schema.get_num_attributes() == num_attributes
    assert schema.get_num_nominal_attributes() == len(nominal_attributes)
    assert schema.get_num_numeric_attributes() == len(numeric_attributes)
    assert schema.get_numeric_attributes() == numeric_attributes


FEATURES = ["num1", "num2", "cat1", "cat2"]
NUMERIC_ATTRS = ["num1", "num2"]
NOMINAL_ATTRS = {"cat1": ["A", "B", "C"], "cat2": ["X", "Y"]}
RESOURCES = Path("tests/resources/stream")


DATA = np.array(
    [
        [-1.10, -1.00, 0.00, 0.00],
        [0.10, 1.00, 1.00, 1.00],
        [1.10, 0.00, 2.00, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ]
)
XN1 = np.delete(DATA, 0, axis=1)
YN1 = DATA[:, 0]
XN2 = np.delete(DATA, 1, axis=1)
YN2 = DATA[:, 1]
XC1 = np.delete(DATA, 2, axis=1)
YC1 = DATA[:, 2]
XC2 = np.delete(DATA, 3, axis=1)
YC2 = DATA[:, 3]
DATASET_C1 = TensorDataset(torch.tensor(XC1), torch.tensor(YC1))
DATASET_C2 = TensorDataset(torch.tensor(XC2), torch.tensor(YC2))
ARFF = RESOURCES / "stream_test.arff"
CSV = RESOURCES / "stream_test.csv"


@pytest.mark.parametrize(
    ["stream", "target", "length"],
    [
        (ARFFStream(ARFF, class_index=2), "cat1", None),
        (ARFFStream(ARFF, class_index=-1), "cat2", None),
        (stream_from_file(ARFF, class_index=2), "cat1", None),
        (stream_from_file(ARFF, class_index=-1), "cat2", None),
        (CSVStream(CSV, "cat1", NOMINAL_ATTRS), "cat1", None),
        (CSVStream(CSV, "cat2", NOMINAL_ATTRS), "cat2", None),
        (stream_from_file(CSV, class_index=2), "cat1", 5),
        (stream_from_file(CSV, class_index=3), "cat2", 5),
        (NumpyStream(XC1, YC1, target_type="categorical"), "cat1", 5),
        (NumpyStream(XC2, YC2, target_type="categorical"), "cat2", 5),
        (TorchClassifyStream(DATASET_C1, 3), "cat1", 5),  # type: ignore
        (TorchClassifyStream(DATASET_C2, 2), "cat2", 5),  # type: ignore
    ],
)
def test_stream_classification(
    stream: Stream[LabeledInstance], target: str, length: Optional[int]
):
    """Test the classification stream interface for a variety of stream types."""

    # Expected schema/attributes
    numeric_attributes = NUMERIC_ATTRS.copy()
    nominal_attributes = NOMINAL_ATTRS.copy()

    label_values = nominal_attributes.pop(target)
    label_indexes = list(range(len(label_values)))
    num_attributes = len(numeric_attributes) + len(nominal_attributes)

    # NumpyStream and PyTorch streams do not have nominal labels by default.
    if isinstance(stream, (NumpyStream, TorchClassifyStream)):
        numeric_attributes = list(map(str, range(num_attributes)))
        nominal_attributes = {}
        label_values = [str(i) for i in label_indexes]

    # Expected data
    target_index = FEATURES.index(target)
    X = np.delete(DATA, target_index, axis=1)
    Y = np.nan_to_num(DATA[:, target_index], nan=-1).astype(int)

    schema = stream.get_schema()

    # Label values/indexes
    assert schema.get_label_values() == label_values
    assert schema.get_label_indexes() == label_indexes
    assert schema.get_num_classes() == len(label_values)
    for i, label in enumerate(label_values):
        assert schema.get_value_for_index(i) == label
        assert schema.get_index_for_label(label) == i

    # Check attributes
    check_attributes(numeric_attributes, nominal_attributes, num_attributes, schema)

    # Check regression/classification methods
    assert schema.is_regression() is False
    assert schema.is_classification() is True
    assert schema.is_y_index_in_range(schema.get_num_classes() - 1) is True
    assert schema.is_y_index_in_range(schema.get_num_classes()) is False
    assert schema.is_y_index_in_range(-1) is False
    assert schema.dataset_name is not None

    # Check the stream interface.
    assert length is None or len(stream) == length  # type: ignore
    moa_stream = stream.get_moa_stream()
    assert moa_stream is None or isinstance(moa_stream, InstanceStream)
    assert stream.cli_help()

    # Python style iterator
    stream.restart()
    for i, instance in enumerate(stream):
        check_instance(instance, X[i], Y[i])

    # Java style iterator
    stream.restart()
    i = 0
    while stream.has_more_instances():
        instance = stream.next_instance()
        check_instance(instance, X[i], Y[i])
        i += 1


@pytest.mark.parametrize(
    ["stream", "target"],
    [
        (ARFFStream(ARFF, class_index=0), "num1"),
        (ARFFStream(ARFF, class_index=1), "num2"),
        (stream_from_file(ARFF, class_index=0), "num1"),
        (stream_from_file(ARFF, class_index=1), "num2"),
        (CSVStream(CSV, "num1", NOMINAL_ATTRS), "num1"),
        (CSVStream(CSV, "num2", NOMINAL_ATTRS), "num2"),
        (stream_from_file(CSV, class_index=0), "num1"),
        (stream_from_file(CSV, class_index=1), "num2"),
        (NumpyStream(XN1, YN1, target_type="numeric"), "num1"),
        (NumpyStream(XN2, YN2, target_type="numeric"), "num2"),
    ],
)
def test_regression_stream(stream: Stream[RegressionInstance], target: str):
    numeric_attributes = NUMERIC_ATTRS.copy()
    numeric_attributes.remove(target)
    nominal_attributes = NOMINAL_ATTRS.copy()
    num_attributes = len(numeric_attributes) + len(nominal_attributes)

    # Stream treats nominal attributes as numeric
    if isinstance(stream, NumpyStream):
        numeric_attributes = list(map(str, range(num_attributes)))
        nominal_attributes = {}

    target_index = FEATURES.index(target)
    X = np.delete(DATA, target_index, axis=1)
    Y = DATA[:, target_index]

    schema = stream.get_schema()

    # Check label methods raise StreamTypeError
    with pytest.raises(StreamTypeError):
        schema.get_label_values()
    with pytest.raises(StreamTypeError):
        schema.get_label_indexes()
    assert schema.get_num_classes() == 1
    assert schema.is_regression() is True
    assert schema.is_classification() is False
    assert schema.dataset_name is not None
    check_attributes(numeric_attributes, nominal_attributes, num_attributes, schema)

    # Python style iterator
    stream.restart()
    for i, instance in enumerate(stream):
        check_instance(instance, X[i], Y[i])

    # Java style iterator
    stream.restart()
    i = 0
    while stream.has_more_instances():
        instance = stream.next_instance()
        check_instance(instance, X[i], Y[i])
        i += 1
