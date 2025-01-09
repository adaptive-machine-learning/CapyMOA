"""This module is for testing the speeds of different stream implementations."""

from capymoa.stream import stream_from_file
from typing import List
import numpy as np

from capymoa.stream import Stream
from capymoa.instance import Instance, LabeledInstance
from capymoa.stream._stream import CSVStream
import pytest


def _get_streams() -> List[Stream]:
    return [
        stream_from_file("data/electricity_tiny.csv"),
        stream_from_file("data/electricity_tiny.arff"),
        CSVStream("data/electricity_tiny.csv"),
    ]


def test_stream_consistency():
    streams = _get_streams()

    def _has_more_instance():
        return [stream.has_more_instances() for stream in streams]

    def _next_instance():
        return [stream.next_instance() for stream in streams]

    for schema in [stream.get_schema() for stream in streams]:
        assert schema.get_num_attributes() == 6
        assert schema.get_num_classes() == 2

    i = 0
    while any(_has_more_instance()):
        assert all(
            _has_more_instance()
        ), "Not all streams have the same number of instances"
        i += 1

        instances = _next_instance()
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
        assert isinstance(instance, Instance)
        assert isinstance(instance, LabeledInstance)
        assert isinstance(instance.x, np.ndarray)
        assert isinstance(instance.y_index, int)
        assert instance.y_index >= 0
        assert instance.y_index < stream.get_schema().get_num_classes()
        assert instance.x.shape == (stream.get_schema().get_num_attributes(),)
