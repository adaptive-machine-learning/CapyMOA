"""This module is for testing the speeds of different stream implementations.
"""
import time
from capymoa.stream import stream_from_file
from cProfile import Profile
from typing import List
import numpy as np

from capymoa.stream import Stream
from capymoa.stream.instance import Instance
from capymoa.stream.stream import CSVStream
import csv

def _get_streams() -> List[Stream]:
    return [
        stream_from_file("data/electricity_tiny.csv"),
        stream_from_file("data/electricity_tiny.arff"),
        CSVStream("data/electricity_tiny.csv")
    ]

def test_stream_consistency():
    streams = _get_streams()

    def _has_more_instance():
        return [stream.has_more_instances() for stream in streams]
    
    def _next_instance():
        return [stream.next_instance() for stream in streams]
    
    i = 0
    while any(_has_more_instance()):
        assert all(_has_more_instance()), "Not all streams have the same number of instances"
        i += 1

        instances = _next_instance()
        prototype = instances.pop()
        for instance in instances:
            assert np.allclose(prototype.x, instance.x), f"Streams are not consistent at instance {i}"
            assert prototype.y_index == instance.y_index, f"Streams are not consistent at instance {i}"
