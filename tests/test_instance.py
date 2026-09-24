"""Verify CapyMOA reads MOA instances correctly regardless of Dense vs Sparse storage.

CapyMOA never constructs a ``SparseInstance`` itself, but a MOA stream can hand one
back (e.g. an ARFF file using sparse row syntax, see ``test_stream.py``). These tests
construct matching ``DenseInstance``/``SparseInstance`` objects directly and confirm
``Instance``/``LabeledInstance``/``RegressionInstance`` extract the same values from
both, since CapyMOA passes the underlying MOA instance through unchanged.
"""

import numpy as np
from com.yahoo.labs.samoa.instances import DenseInstance, SparseInstance
from jpype import JArray, JDouble, JInt
from moa.core import InstanceExample

from capymoa.core import LabeledInstance, RegressionInstance
from capymoa.stream import Schema

# Logical row shared by both representations: f1=1.1, f2=0.0, f3=2.0, target=<last>.
# f2 is deliberately 0.0 so the sparse instance can omit it and still be genuinely
# sparse (fewer stored values than attributes), not just a relabelled dense instance.
F_VALUES = [1.1, 0.0, 2.0]


def _dense_instance(header, class_value: float) -> DenseInstance:
    values = JArray(JDouble)([*F_VALUES, class_value])
    instance = DenseInstance(1.0, values)
    instance.setDataset(header)
    return instance


def _sparse_instance(header, class_value: float) -> SparseInstance:
    # Omit index 1 (f2 == 0.0): only 3 of 4 attributes are explicitly stored.
    values = JArray(JDouble)([F_VALUES[0], F_VALUES[2], class_value])
    indices = JArray(JInt)([0, 2, 3])
    instance = SparseInstance(1.0, values, indices, 4)
    instance.setDataset(header)
    assert instance.numValues() < instance.numAttributes(), (
        "test setup bug: instance is not actually sparse"
    )
    return instance


def test_dense_and_sparse_instance_agree_classification():
    schema = Schema.from_custom(
        features=["f1", "f2", "f3", "class"],
        target="class",
        categories={"class": ["yes", "no"]},
        name="instance-representation-test",
    )
    header = schema.get_moa_header()
    class_value = 1.0  # "no"

    dense = _dense_instance(header, class_value)
    sparse = _sparse_instance(header, class_value)

    for raw_instance in (dense, sparse):
        instance = LabeledInstance.from_java_instance(
            schema, InstanceExample(raw_instance)
        )
        assert np.allclose(instance.x, F_VALUES)
        assert instance.y_index == 1
        assert instance.y_label == "no"
        assert instance.java_instance.getData().weight() == 1.0


def test_dense_and_sparse_instance_agree_regression():
    schema = Schema.from_custom(
        features=["f1", "f2", "f3", "target"],
        target="target",
        name="instance-representation-test-regression",
    )
    header = schema.get_moa_header()
    class_value = 42.5

    dense = _dense_instance(header, class_value)
    sparse = _sparse_instance(header, class_value)

    for raw_instance in (dense, sparse):
        instance = RegressionInstance.from_java_instance(
            schema, InstanceExample(raw_instance)
        )
        assert np.allclose(instance.x, F_VALUES)
        assert instance.y_value == class_value
