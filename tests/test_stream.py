"""This module is for testing the speeds of different stream implementations."""

from functools import partial
from typing import Optional, Sized

from capymoa.stream._stream import ConcatStream
import numpy as np
import pytest
import torch
from com.yahoo.labs.samoa.instances import (
    InstancesHeader,
)
from moa.streams import InstanceStream
from torch.utils.data import TensorDataset

from capymoa.datasets import ElectricityTiny, FriedTiny
from capymoa.datasets._utils import get_download_dir
from capymoa.instance import Instance, LabeledInstance, RegressionInstance
from capymoa.stream import (
    ARFFStream,
    CSVStream,
    NumpyStream,
    Stream,
    TorchClassifyStream,
    stream_from_file,
)

CLASSIFICATION_X = np.array(
    [
        [0.0, 0.056, 0.439, 0.003, 0.423, 0.415],
        [0.021, 0.052, 0.415, 0.003, 0.423, 0.415],
        [0.043, 0.051, 0.385, 0.003, 0.423, 0.415],
        [0.064, 0.045, 0.315, 0.003, 0.423, 0.415],
        [0.085, 0.042, 0.251, 0.003, 0.423, 0.415],
        [0.106, 0.041, 0.208, 0.003, 0.423, 0.415],
        [0.128, 0.041, 0.172, 0.003, 0.423, 0.415],
        [0.149, 0.041, 0.153, 0.003, 0.423, 0.415],
        [0.17, 0.041, 0.135, 0.003, 0.423, 0.415],
        [0.191, 0.041, 0.141, 0.003, 0.423, 0.415],
    ]
)

CLASSIFICATION_Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
CLASSIFICATION_TENSOR_DATASET = TensorDataset(
    torch.tensor(CLASSIFICATION_X), torch.tensor(CLASSIFICATION_Y)
)
CLASSIFICATION_CSV = get_download_dir() / "electricity_tiny.csv"
CLASSIFICATION_ARFF = get_download_dir() / "electricity_tiny.arff"

REGRESSION_X = np.array(
    [
        [0.487, 0.072, 0.004, 0.833, 0.765, 0.6, 0.132, 0.886, 0.073, 0.342],
        [0.223, 0.401, 0.659, 0.528, 0.843, 0.713, 0.58, 0.473, 0.572, 0.528],
        [0.903, 0.913, 0.94, 0.979, 0.561, 0.744, 0.627, 0.818, 0.309, 0.51],
        [0.791, 0.857, 0.359, 0.844, 0.155, 0.948, 0.114, 0.292, 0.412, 0.991],
        [0.326, 0.593, 0.085, 0.927, 0.926, 0.633, 0.431, 0.326, 0.031, 0.73],
        [0.562, 0.89, 0.006, 0.691, 0.72, 0.208, 0.279, 0.283, 0.116, 0.882],
        [0.481, 0.613, 0.499, 0.572, 0.914, 0.783, 0.204, 0.428, 0.828, 0.487],
        [0.625, 0.197, 0.725, 0.628, 0.541, 0.481, 0.46, 0.021, 0.765, 0.392],
        [0.21, 0.519, 0.029, 0.61, 0.724, 0.515, 0.371, 0.731, 0.575, 0.73],
        [0.084, 0.496, 0.486, 0.813, 0.406, 0.491, 0.418, 0.344, 0.978, 0.409],
    ]
)
REGRESSION_Y = np.array(
    [17.949, 13.815, 20.766, 18.301, 22.989, 25.986, 17.15, 14.006, 18.566, 12.107]
)
REGRESSION_CSV = get_download_dir() / "fried_tiny.csv"
REGRESSION_ARFF = get_download_dir() / "fried_tiny.arff"


allclose = partial(np.allclose, atol=0.001)


def check_java_instance(instance: Instance, x: np.ndarray, target: float):
    assert instance.java_instance is not None
    assert instance.java_instance.getData().classValue() == target
    assert instance.java_instance.getData().classIndex() == len(x)
    jxy = np.array(instance.java_instance.getData().toDoubleArray())
    jx = jxy[: len(x)]  # Get the first elements for the features.
    jy = jxy[len(x)]  # The last element is the label.
    assert allclose(jx, x)
    assert jy == target


@pytest.mark.parametrize(
    ["stream", "length"],
    [
        (stream_from_file(CLASSIFICATION_CSV), None),
        (CSVStream(CLASSIFICATION_CSV), None),
        (stream_from_file(CLASSIFICATION_ARFF), None),
        (ARFFStream(CLASSIFICATION_ARFF), None),
        (ElectricityTiny(get_download_dir()), 2_000),
        (TorchClassifyStream(CLASSIFICATION_TENSOR_DATASET, 2), 10),
        (
            NumpyStream(CLASSIFICATION_X, CLASSIFICATION_Y, target_type="categorical"),
            10,
        ),
        (
            ConcatStream(
                [
                    NumpyStream(
                        CLASSIFICATION_X, CLASSIFICATION_Y, target_type="categorical"
                    ),
                    NumpyStream(
                        CLASSIFICATION_X, CLASSIFICATION_Y, target_type="categorical"
                    ),
                ]
            ),
            20,
        ),
        # Add new stream types here.
    ],
)
def test_stream_classification(stream: Stream[LabeledInstance], length: Optional[int]):
    """Test the classification stream interface for a variety of stream types."""
    # Check the stream schema.
    schema = stream.get_schema()
    assert schema.get_label_values() == ["0", "1"]
    assert schema.get_label_indexes() == [0, 1]
    assert schema.get_value_for_index(0) == "0"
    assert schema.get_index_for_label("0") == 0
    assert isinstance(schema.get_moa_header(), InstancesHeader)
    assert schema.get_num_attributes() == 6
    assert schema.get_num_nominal_attributes() == 0
    assert schema.get_num_numeric_attributes() == 6
    assert schema.get_nominal_attributes() is None
    assert len(schema.get_numeric_attributes()) == 6
    assert schema.get_num_classes() == 2
    assert schema.is_regression() is False
    assert schema.is_classification() is True
    assert schema.is_y_index_in_range(2) is False
    assert schema.dataset_name is not None

    # Check attribute names.
    if not isinstance(stream, (TorchClassifyStream, NumpyStream, ConcatStream)):
        assert schema.get_numeric_attributes() == [
            "period",
            "nswprice",
            "nswdemand",
            "vicprice",
            "vicdemand",
            "transfer",
        ]

    # Check the stream interface.
    assert length is None or (isinstance(stream, Sized) and len(stream) == length)
    moa_stream = stream.get_moa_stream()
    assert moa_stream is None or isinstance(moa_stream, InstanceStream)
    assert stream.cli_help()

    # Check java style next instance.
    instance = stream.next_instance()
    assert isinstance(instance, LabeledInstance)
    assert allclose(instance.x, CLASSIFICATION_X[0])
    assert allclose(instance.y_index, CLASSIFICATION_Y[0])
    assert isinstance(instance.x, np.ndarray)
    assert isinstance(instance.y_index, int)
    assert isinstance(instance.y_label, str)
    check_java_instance(instance, CLASSIFICATION_X[0], CLASSIFICATION_Y[0])

    # Check python style next instance.
    instance = next(stream)
    assert isinstance(instance, LabeledInstance)
    assert allclose(instance.x, CLASSIFICATION_X[1])
    assert allclose(instance.y_index, CLASSIFICATION_Y[1])
    assert isinstance(instance.x, np.ndarray)
    assert isinstance(instance.y_index, int)
    assert isinstance(instance.y_label, str)
    check_java_instance(instance, CLASSIFICATION_X[1], CLASSIFICATION_Y[1])

    # Check stream iteration.
    stream.restart()
    for i, instance in enumerate(stream):
        if i >= len(CLASSIFICATION_X):
            break
        assert allclose(instance.x, CLASSIFICATION_X[i])
        assert allclose(instance.y_index, CLASSIFICATION_Y[i])
        check_java_instance(instance, CLASSIFICATION_X[i], CLASSIFICATION_Y[i])

    # Check exhausting the stream.
    for _ in stream:
        pass
    assert stream.has_more_instances() is False


@pytest.mark.parametrize(
    ["stream", "length"],
    [
        (FriedTiny(), 1_000),
        (stream_from_file(REGRESSION_CSV, target_type="numeric"), None),
        (CSVStream(REGRESSION_CSV, target_type="numeric"), None),
        (stream_from_file(REGRESSION_ARFF, target_type="numeric"), None),
        (ARFFStream(REGRESSION_ARFF), None),
        (
            NumpyStream(REGRESSION_X, REGRESSION_Y, target_type="numeric"),
            10,
        ),
        (
            ConcatStream(
                [
                    NumpyStream(
                        REGRESSION_X[:5], REGRESSION_Y[:5], target_type="numeric"
                    ),
                    NumpyStream(
                        REGRESSION_X[5:], REGRESSION_Y[5:], target_type="numeric"
                    ),
                ]
            ),
            10,
        ),
        # Add new stream types here.
    ],
)
def test_stream_regression(stream: Stream[RegressionInstance], length: Optional[int]):
    """Test the regression stream interface for a variety of stream types."""

    # Check the stream schema.
    schema = stream.get_schema()
    assert isinstance(schema.get_moa_header(), InstancesHeader)
    assert schema.get_num_attributes() == 10
    assert schema.get_num_nominal_attributes() == 0
    assert schema.get_num_numeric_attributes() == 10
    assert schema.get_nominal_attributes() is None
    assert len(schema.get_numeric_attributes()) == 10
    assert schema.get_num_classes() == 1
    assert schema.is_regression() is True
    assert schema.is_classification() is False
    assert schema.dataset_name is not None
    # Some schema methods are regression specific.
    ex = RuntimeError
    with pytest.raises(ex):
        schema.get_label_values()
    with pytest.raises(ex):
        schema.get_label_indexes()
    with pytest.raises(ex):
        schema.get_value_for_index(0)
    with pytest.raises(ex):
        schema.get_index_for_label("0")
    with pytest.raises(ex):
        schema.is_y_index_in_range(0)

    # Check attribute names.
    if not isinstance(stream, (TorchClassifyStream, NumpyStream, ConcatStream)):
        assert schema.get_numeric_attributes() == [f"attr_{i}" for i in range(10)]

    # Check the stream interface.
    assert length is None or (isinstance(stream, Sized) and len(stream) == length)
    moa_stream = stream.get_moa_stream()
    assert moa_stream is None or isinstance(moa_stream, InstanceStream)
    assert stream.cli_help()
    assert stream.has_more_instances()
    assert iter(stream) == stream

    # Check java style next instance.
    instance = stream.next_instance()
    assert isinstance(instance, RegressionInstance)
    assert allclose(instance.x, REGRESSION_X[0])
    assert allclose(instance.y_value, REGRESSION_Y[0])
    assert isinstance(instance.y_value, float)
    assert isinstance(instance.x, np.ndarray)
    check_java_instance(instance, REGRESSION_X[0], REGRESSION_Y[0])

    # Check python style next instance.
    instance = next(stream)
    assert isinstance(instance, RegressionInstance)
    assert allclose(instance.x, REGRESSION_X[1])
    assert allclose(instance.y_value, REGRESSION_Y[1])
    assert isinstance(instance.y_value, float)
    assert isinstance(instance.x, np.ndarray)
    check_java_instance(instance, REGRESSION_X[1], REGRESSION_Y[1])

    # Check stream iteration.
    stream.restart()
    for i, instance in enumerate(stream):
        if i >= len(CLASSIFICATION_X):
            break
        assert allclose(instance.x, REGRESSION_X[i])
        assert allclose(instance.y_value, REGRESSION_Y[i])
        check_java_instance(instance, REGRESSION_X[i], REGRESSION_Y[i])

    # Check exhausting the stream.
    for _ in stream:
        pass
    assert stream.has_more_instances() is False
