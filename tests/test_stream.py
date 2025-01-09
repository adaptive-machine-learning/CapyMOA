"""This module is for testing the speeds of different stream implementations."""

from capymoa.evaluation.evaluation import prequential_ssl_evaluation
from capymoa.stream import stream_from_file
from typing import List, Optional
import numpy as np

from capymoa.base import ClassifierSSL
from capymoa.stream import Stream, CSVStream, NumpyStream
from capymoa.instance import Instance, LabeledInstance
from capymoa.type_alias import LabelIndex, LabelProbabilities
from capymoa.evaluation import prequential_evaluation
import pytest


def _get_streams() -> List[Stream]:
    return [
        stream_from_file("data/electricity_tiny.csv"),
        stream_from_file("data/electricity_tiny.arff"),
        CSVStream("data/electricity_tiny.csv"),
    ]


def test_stream_consistency():
    streams = _get_streams()

    for schema in [stream.get_schema() for stream in streams]:
        assert schema.get_num_attributes() == 6
        assert schema.get_num_classes() == 2

    i = 0
    for instances in zip(*streams, strict=True):
        instances = list(instances)
        i += 1

        prototype = instances.pop()
        for instance in instances:
            assert np.allclose(
                prototype.x, instance.x
            ), f"Streams are not consistent at instance {i}"
            assert (
                prototype.y_index == instance.y_index
            ), f"Streams are not consistent at instance {i}"


@pytest.mark.parametrize("stream", _get_streams())
def test_iterator_api(stream: Stream[LabeledInstance]):
    for instance in stream:
        assert isinstance(instance, Instance), f"got {type(instance)}"
        assert isinstance(instance, LabeledInstance)
        assert isinstance(instance.x, np.ndarray)
        assert isinstance(instance.y_index, int)
        assert instance.y_index >= 0
        assert instance.y_index < stream.get_schema().get_num_classes()
        assert instance.x.shape == (stream.get_schema().get_num_attributes(),)


def test_stream():
    """Basic check for the evaluation loops."""

    class DummyClassifier(ClassifierSSL):
        def __init__(self, schema):
            super().__init__(schema)
            self.i = 0

        def __str__(self):
            return "DummyClassifier"

        def train(self, instance: LabeledInstance):
            assert instance.x == self.i
            self.i += 1

        def train_on_unlabeled(self, instance: Instance):
            assert instance.x == self.i
            self.i += 1

        def predict(self, instance: Instance) -> Optional[LabelIndex]:
            return 0

        def predict_proba(self, instance: Instance) -> LabelProbabilities:
            return [1.0, 0.0]

    stream = NumpyStream(
        np.arange(100).reshape((100, 1)), np.zeros(100), target_type="categorical"
    )

    prequential_evaluation(
        stream, DummyClassifier(stream.get_schema()), max_instances=10
    )
    assert next(stream).x == 10

    prequential_ssl_evaluation(
        stream,
        DummyClassifier(stream.get_schema()),
        max_instances=10,
        restart_stream=True,
    )
    assert next(stream).x == 10
