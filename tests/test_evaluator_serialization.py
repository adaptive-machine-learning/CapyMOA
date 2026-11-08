import tempfile

import pytest

from capymoa.classifier.evaluate import (
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
)
from capymoa.core.io import load_model, save_model
from capymoa.stream.generator import SEA


@pytest.mark.parametrize(
    "evaluator_cls,kwargs",
    [
        (ClassificationEvaluator, {}),
        (ClassificationWindowedEvaluator, {"window_size": 50}),
    ],
)
def test_evaluator_save_load_roundtrip(evaluator_cls, kwargs):
    """Evaluators must round-trip through save_model/load_model."""
    stream = SEA()
    evaluator = evaluator_cls(schema=stream.get_schema(), **kwargs)
    for _ in range(100):
        evaluator.update(stream.next_instance().y_index, 0)

    with tempfile.TemporaryFile() as fd:
        save_model(evaluator, fd)
        fd.seek(0)
        restored = load_model(fd)

    assert isinstance(restored, evaluator_cls)
    assert restored.instances_seen == evaluator.instances_seen
    assert restored.accuracy() == pytest.approx(evaluator.accuracy())


@pytest.mark.parametrize(
    "cls",
    [ClassificationEvaluator, ClassificationWindowedEvaluator],
)
def test_evaluator_getattr_does_not_recurse_before_init(cls):
    """__getattr__ must raise AttributeError, not recurse, on a bare object."""
    bare = cls.__new__(cls)
    assert not hasattr(bare, "__setstate__")
