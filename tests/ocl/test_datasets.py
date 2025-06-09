from typing import Type
from capymoa.ocl import datasets
from capymoa.stream import Stream
from capymoa.stream._stream import Schema
import numpy as np
import pytest
import inspect

ALL_OCL_SCENARIO = [
    cls
    for _, cls in inspect.getmembers(datasets)
    if inspect.isclass(cls)
    and issubclass(cls, datasets._BuiltInCIScenario)
    and cls != datasets._BuiltInCIScenario
]


@pytest.mark.parametrize("scenario_type", ALL_OCL_SCENARIO)
def test_ocl_split_datamodule_constructors(
    scenario_type: Type[datasets._BuiltInCIScenario],
):
    # Skip all except MNIST since downloading datasets can be slow on CI
    if scenario_type != datasets.TinySplitMNIST:
        pytest.skip("Skipping non-MNIST scenarios")

    scenario: datasets._BuiltInCIScenario = scenario_type()
    assert isinstance(scenario.train_tasks, list)
    assert isinstance(scenario.test_tasks, list)
    assert isinstance(scenario.schema, Schema)
    assert isinstance(scenario.stream, Stream)
    assert isinstance(scenario.task_schedule, list)
    assert len(scenario.task_schedule) == scenario.default_task_count
    assert len(scenario.train_tasks) == scenario.default_task_count
    assert len(scenario.test_tasks) == scenario.default_task_count

    train_instance = scenario.stream.next_instance()
    test_instance = scenario.stream.next_instance()

    assert isinstance(train_instance.x, np.ndarray)
    assert isinstance(test_instance.y_index, int)
    assert isinstance(test_instance.x, np.ndarray)
    assert isinstance(test_instance.y_index, int)
