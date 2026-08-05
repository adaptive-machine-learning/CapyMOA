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
    TorchStream,
    stream_from_file,
)
from capymoa.stream.drift import (
    AbruptDrift,
    Concept,
    Drift,
    DriftStream,
    GradualDrift,
    RecurrentConceptDriftStream,
)
from capymoa.stream.generator import SEA, LEDGeneratorDrift, RandomTreeGenerator
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
DATASET_N1 = TensorDataset(torch.tensor(XN1), torch.tensor(YN1))
DATASET_N2 = TensorDataset(torch.tensor(XN2), torch.tensor(YN2))
ARFF = RESOURCES / "stream_test.arff"
CSV = RESOURCES / "stream_test.csv"


def recurrent_drift_concepts():
    return [
        RandomTreeGenerator(tree_random_seed=1, instance_random_seed=1),
        RandomTreeGenerator(tree_random_seed=2, instance_random_seed=1),
    ]


def test_led_generator_drift_str_with_drift_attributes():
    stream = LEDGeneratorDrift(number_of_attributes_with_drift=1)

    assert str(stream) == "LEDGeneratorDrift(number_of_attributes_with_drift=1)"


def test_recurrent_concept_drift_stream_accepts_gradual_position_width():
    stream = RecurrentConceptDriftStream(
        concept_list=recurrent_drift_concepts(),
        max_recurrences_per_concept=1,
        transition_type_template=GradualDrift(position=100, width=10),
    )

    assert stream.get_num_drifts() == 1
    assert str(stream.get_drifts()[0]) == (
        "GradualDrift(position=100, start=95, end=105, width=10)"
    )
    # The MOA CLI is now produced on demand rather than being the implementation.
    assert "-w 10 -p 100" in stream.to_moa_stream()._CLI


@pytest.mark.parametrize(
    ["kwargs", "expected"],
    [
        (
            {"position": 100, "width": 10},
            "GradualDrift(position=100, start=95, end=105, width=10)",
        ),
        (
            {"start": 95, "end": 105},
            "GradualDrift(position=100, start=95, end=105, width=10)",
        ),
    ],
)
def test_gradual_drift_valid_forms(kwargs, expected):
    """Both ways of locating a gradual drift describe the same drift."""
    assert str(GradualDrift(**kwargs)) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"position": 100},
        {"width": 10},
        {"start": 95},
        {"end": 105},
        {"position": 100, "start": 95},
        {"width": 10, "end": 105},
        {"position": 100, "width": 10, "start": 95, "end": 105},
    ],
)
def test_gradual_drift_rejects_incomplete_forms(kwargs):
    """Incomplete or mixed styles raise a clear error, not an internal one.

    Previously these fell through the branching and raised ``TypeError`` from
    arithmetic on ``None``, or left the object without a ``position`` at all
    and failed later.
    """
    with pytest.raises(ValueError, match="exactly one of"):
        GradualDrift(**kwargs)


def _range_and_position_equivalents():
    """The same stream written both ways."""
    range_form = DriftStream(
        stream=[
            Concept(SEA(function=1), num_instances=1000),
            AbruptDrift(),
            Concept(SEA(function=2), num_instances=500),
            GradualDrift(num_instances=500),
            Concept(SEA(function=3), num_instances=500),
        ]
    )
    position_form = DriftStream(
        stream=[
            SEA(function=1),
            AbruptDrift(position=1000),
            SEA(function=2),
            GradualDrift(position=1750, width=500),
            SEA(function=3),
        ]
    )
    return range_form, position_form


def test_range_form_resolves_to_the_same_drifts():
    """Lengths are translated into the positions they imply."""
    range_form, position_form = _range_and_position_equivalents()

    assert range_form.range_form is True
    assert position_form.range_form is False
    assert [str(d) for d in range_form.get_drifts()] == [
        str(d) for d in position_form.get_drifts()
    ]
    assert str(range_form.get_drifts()[0]) == "AbruptDrift(position=1000)"
    assert str(range_form.get_drifts()[1]) == (
        "GradualDrift(position=1750, start=1500, end=2000, width=500)"
    )


def test_range_form_produces_the_same_instances():
    """Translation only -- the two forms describe one stream."""
    range_form, position_form = _range_and_position_equivalents()

    a = np.array([range_form.next_instance().x for _ in range(2500)])
    b = np.array([position_form.next_instance().x for _ in range(2500)])

    assert np.allclose(a, b)


def test_range_form_accepts_python_native_concepts():
    """Lengths work with streams MOA cannot represent."""
    rng = np.random.default_rng(0)
    before = NumpyStream(rng.random((300, 3)), rng.integers(0, 2, 300), "before")
    after = NumpyStream(rng.random((300, 3)) + 10, rng.integers(0, 2, 300), "after")

    stream = DriftStream(
        stream=[
            Concept(before, num_instances=100),
            AbruptDrift(),
            Concept(after, num_instances=100),
        ]
    )

    x = np.array([stream.next_instance().x for _ in range(150)])
    assert str(stream.get_drifts()[0]) == "AbruptDrift(position=100)"
    assert x[:98].mean() < 5
    assert x[100:].mean() > 5


@pytest.mark.parametrize(
    ["definition", "match"],
    [
        # Mixing the two forms.
        (
            lambda: [Concept(SEA(function=1), 100), AbruptDrift(), SEA(function=3)],
            "either all be wrapped",
        ),
        # A position alongside lengths contradicts them.
        (
            lambda: [
                Concept(SEA(function=1), 100),
                AbruptDrift(position=50),
                Concept(SEA(function=3), 100),
            ],
            "cannot carry a position",
        ),
        (
            lambda: [
                Concept(SEA(function=1), 100),
                GradualDrift(position=50, width=10),
                Concept(SEA(function=3), 100),
            ],
            "cannot carry a position",
        ),
    ],
)
def test_range_form_rejects_contradictory_definitions(definition, match):
    with pytest.raises(ValueError, match=match):
        DriftStream(stream=definition())


def test_concept_counts_match_actually_consuming_the_stream():
    """The prediction must be exact, not an estimate.

    Routing depends only on each transition's own seeded generator and
    counter, never on the instances, so replaying it reproduces the same
    branch pattern without generating any data.
    """
    from capymoa.stream.drift import _ConceptNode

    def build():
        return DriftStream(
            stream=[
                Concept(SEA(function=1), num_instances=1000),
                AbruptDrift(),
                Concept(SEA(function=2), num_instances=500),
                GradualDrift(num_instances=500),
                Concept(SEA(function=3), num_instances=500),
            ]
        )

    predicted = build().get_concept_counts(2500)

    # Count for real, by tagging each leaf of the composition tree.
    stream = build()
    leaves = []

    def tag(node):
        if isinstance(node, _ConceptNode):
            node.drawn = 0
            original = node.next_instance

            def counting(_node=node, _original=original):
                _node.drawn += 1
                return _original()

            node.next_instance = counting
            leaves.append(node)
        else:
            tag(node.before)
            tag(node.after)

    tag(stream._root)
    for _ in range(2500):
        stream.next_instance()

    assert predicted == [leaf.drawn for leaf in leaves]
    assert sum(predicted) == 2500


def test_concept_counts_show_gradual_overlap():
    """A gradual drift makes the counts diverge from the declared lengths."""
    stream = DriftStream(
        stream=[
            Concept(SEA(function=1), num_instances=1000),
            AbruptDrift(),
            Concept(SEA(function=2), num_instances=500),
            GradualDrift(num_instances=500),
            Concept(SEA(function=3), num_instances=500),
        ]
    )
    counts = stream.get_concept_counts(2500)

    # The abrupt drift is a clean switch, so the first concept is exact.
    assert counts[0] == 999
    # The two concepts either side of the gradual drift share its window, so
    # both are drawn from more often than their declared length.
    assert counts[1] > 500
    assert counts[2] > 500


def test_concept_counts_work_for_the_position_form():
    stream = DriftStream(
        stream=[
            SEA(function=1),
            AbruptDrift(position=1000),
            SEA(function=3),
        ]
    )
    assert stream.get_concept_counts(2000) == [999, 1001]
    assert "position form" in stream.describe(2000)


def test_concept_counts_reject_bad_input():
    stream = DriftStream(
        stream=[SEA(function=1), AbruptDrift(position=100), SEA(function=3)]
    )
    with pytest.raises(ValueError, match="zero or positive"):
        stream.get_concept_counts(-1)


def test_horizon_defaults_to_the_declared_length_for_the_range_form():
    stream = DriftStream(
        stream=[
            Concept(SEA(function=1), num_instances=1000),
            GradualDrift(num_instances=200),
            Concept(SEA(function=3), num_instances=1000),
        ]
    )
    # 1000 + 200 (the drift spends its own) + 1000
    assert sum(stream.get_concept_counts()) == 2200
    assert "from the declared lengths" in stream.describe()


def test_horizon_is_estimated_for_the_position_form():
    """The final concept is open-ended, so its length is inferred."""
    stream = DriftStream(
        stream=[
            SEA(function=1),
            AbruptDrift(position=1000),
            SEA(function=2),
            AbruptDrift(position=2000),
            SEA(function=3),
        ]
    )
    # drifts 1000 apart, so the last concept is assumed to run about as long
    assert sum(stream.get_concept_counts()) == 3000
    report = stream.describe()
    assert "estimated" in report
    assert "open-ended" in report


def test_horizon_must_be_given_when_there_are_no_drifts():
    stream = DriftStream(stream=[SEA(function=1)])
    with pytest.raises(ValueError, match="no drifts"):
        stream.get_concept_counts()
    assert sum(stream.get_concept_counts(500)) == 500


def test_describe_reports_declared_lengths_for_the_range_form():
    stream = DriftStream(
        stream=[
            Concept(SEA(function=1), num_instances=1000),
            AbruptDrift(),
            Concept(SEA(function=3), num_instances=500),
        ]
    )
    report = stream.describe(1500)

    assert "range form" in report
    assert "declared" in report
    assert "AbruptDrift(position=1000)" in report


def test_concept_requires_a_positive_length():
    for bad in (0, -1, None):
        with pytest.raises(ValueError, match="positive ``num_instances``"):
            Concept(SEA(function=1), bad)


def test_abrupt_drift_without_position_is_rejected_by_position_form():
    """An unplaced drift is only meaningful in the range form.

    ``AbruptDrift()`` is how the range form spells "switch here"; the concept
    lengths decide where that is. A definition built from positions has no such
    information, so it rejects the drift rather than defaulting it to zero.
    """
    assert AbruptDrift().position is None

    with pytest.raises(ValueError, match="has no position"):
        DriftStream(
            stream=[
                RandomTreeGenerator(tree_random_seed=1),
                AbruptDrift(),
                RandomTreeGenerator(tree_random_seed=2),
            ]
        )


def test_drift_stream_accepts_python_native_concepts():
    """Python-native streams work as concepts now that composition is in Python.

    Delegating to MOA's ConceptDriftStream meant concepts had to be MOA-backed,
    and a Python-native one was silently dropped.
    """
    rng = np.random.default_rng(0)
    before = NumpyStream(rng.random((200, 3)), rng.integers(0, 2, 200), "before")
    after = NumpyStream(rng.random((200, 3)) + 10, rng.integers(0, 2, 200), "after")

    stream = DriftStream(stream=[before, AbruptDrift(position=100), after])
    x = np.array([stream.next_instance().x for _ in range(150)])

    assert stream.get_num_drifts() == 1
    # The concepts are separated by 10 in feature space, so the switch is
    # unambiguous either side of the drift.
    assert x[:98].mean() < 5
    assert x[100:].mean() > 5


def test_to_moa_stream_refuses_python_native_concepts():
    """The MOA conversion is opt-in and says why it cannot apply."""
    rng = np.random.default_rng(0)
    before = NumpyStream(rng.random((20, 3)), rng.integers(0, 2, 20), "before")
    after = NumpyStream(rng.random((20, 3)), rng.integers(0, 2, 20), "after")

    stream = DriftStream(stream=[before, AbruptDrift(position=10), after])
    with pytest.raises(ValueError, match="every concept to be MOA-backed"):
        stream.to_moa_stream()


def _numpy_concepts(rng, n=40):
    x = rng.random((n, 3))
    return x, rng.integers(0, 2, n)


@pytest.mark.parametrize(
    ["name", "differs_in"],
    [
        ("attributes", "attributes"),
        ("classes", "classes"),
        ("task", "task"),
    ],
)
def test_drift_stream_rejects_incompatible_concepts(name, differs_in):
    """Concepts must describe the same learning problem, not just the same count.

    Every concept's instances go to the same learner and are scored with the
    schema of the first, so a mismatch produces misinterpreted instances rather
    than an error. Comparing only ``get_num_attributes()`` let a classification
    concept drift into a regression one.
    """
    rng = np.random.default_rng(0)
    x, y = _numpy_concepts(rng)
    before = NumpyStream(x, y, "before")

    if name == "attributes":
        after = NumpyStream(rng.random((40, 4)), y, "after")
    elif name == "classes":
        after = NumpyStream(x, rng.integers(0, 3, 40), "after")
    else:
        after = NumpyStream(x, rng.random(40), "after", target_type="numeric")

    with pytest.raises(ValueError, match=differs_in):
        DriftStream(stream=[before, AbruptDrift(position=10), after])


def test_drift_stream_has_more_instances_ignores_spent_concepts():
    """A finite concept the drift has moved past must not end the stream.

    ``has_more_instances`` used to require every concept to have data, so a
    loop guarded by it stopped as soon as the *old* concept ran out, while
    ``next_instance`` could still produce instances from the new one.
    """
    rng = np.random.default_rng(0)
    before = NumpyStream(rng.random((99, 3)), rng.integers(0, 2, 99), "before")
    after = NumpyStream(rng.random((10, 3)), rng.integers(0, 2, 10), "after")

    stream = DriftStream(stream=[before, AbruptDrift(position=100), after])

    consumed = 0
    while stream.has_more_instances():
        stream.next_instance()
        consumed += 1

    assert consumed == 109


def test_drift_stream_rejects_non_stream_component():
    with pytest.raises(ValueError, match="cannot use str"):
        DriftStream(
            stream=[RandomTreeGenerator(tree_random_seed=1), AbruptDrift(10), "nope"]
        )


@pytest.mark.parametrize(
    "call",
    [
        # The keyword form callers actually used.
        lambda: Drift(position=1000, width=500, alpha=0.1),
        lambda: GradualDrift(position=1000, width=500, alpha=0.1),
        # The positional forms, where the leftover argument would otherwise
        # slide into random_seed and only fail later inside MOA with
        # `NumberFormatException: For input string: "0.1"`.
        lambda: Drift(1000, 500, 0.1),
        lambda: GradualDrift(1000, 500, None, None, 0.1),
    ],
)
def test_drift_rejects_leftover_alpha_argument(call):
    """A leftover ``alpha`` must fail at the call site, keyword or positional.

    ``random_seed`` is keyword-only so that the extra positional argument
    cannot quietly become the seed.
    """
    with pytest.raises(TypeError):
        call()


def test_drift_random_seed_is_keyword_only():
    """The supported forms still work."""
    assert Drift(1000, 500).random_seed == 1
    assert Drift(position=1000, width=500, random_seed=7).random_seed == 7
    assert GradualDrift(1000, 500).random_seed == 1
    assert GradualDrift(start=750, end=1250, random_seed=7).random_seed == 7
    assert AbruptDrift(100).random_seed == 1


@pytest.mark.parametrize("cls", [Drift, GradualDrift])
def test_drift_no_longer_accepts_alpha(cls):
    """``alpha`` was removed: it never shaped the transition.

    MOA reads it once in ``prepareForUseImpl``, where a non-zero value
    *overwrites* ``width`` with ``1/tan(alpha)``. It was a second, obscure
    spelling of ``width`` rather than a grade of change, and ``GradualDrift``
    discarded it anyway.
    """
    kwargs = {"position": 100, "width": 10, "alpha": 0.1}
    with pytest.raises(TypeError, match="alpha"):
        cls(**kwargs)


def test_drift_stream_cli_has_no_alpha_option():
    """The generated MOA CLI no longer carries ``-a``, so width is honoured."""
    stream = DriftStream(
        stream=[
            RandomTreeGenerator(tree_random_seed=1),
            Drift(position=1000, width=500),
            RandomTreeGenerator(tree_random_seed=2),
        ]
    )

    # The MOA CLI is produced on demand now that composition happens in Python.
    converted = stream.to_moa_stream()
    assert "-a " not in converted._CLI
    assert "-w 500 -p 1000" in converted._CLI

    moa_stream = converted.moa_stream
    moa_stream.prepareForUse()
    assert moa_stream.widthOption.getValue() == 500


@pytest.mark.parametrize(
    ["definition", "match"],
    [
        # Trailing drift: previously reported a drift that was never composed.
        (["gen", "drift"], "must end with a concept"),
        # Two drifts in a row: previously reported both, composed only the last.
        (["gen", "drift", "drift", "gen"], "alternate"),
        # Two concepts in a row, with no drift between them.
        (["gen", "gen"], "alternate"),
        # Leading drift.
        (["drift", "gen"], "alternate"),
    ],
)
def test_drift_stream_rejects_malformed_definitions(definition, match):
    """A malformed definition must raise rather than compose a wrong stream.

    ``get_num_drifts`` used to count every ``Drift`` in the list, including
    ones the builder never composed into the MOA stream, so these produced a
    stream whose declared drifts did not match its behaviour.
    """
    components = [
        RandomTreeGenerator(tree_random_seed=1) if part == "gen" else AbruptDrift(100)
        for part in definition
    ]
    with pytest.raises(ValueError, match=match):
        DriftStream(stream=components)


def test_drift_stream_rejects_empty_definition():
    with pytest.raises(ValueError, match="non-empty"):
        DriftStream(stream=[])


def test_recurrent_concept_drift_stream_accepts_gradual_start_end():
    stream = RecurrentConceptDriftStream(
        concept_list=recurrent_drift_concepts(),
        max_recurrences_per_concept=1,
        transition_type_template=GradualDrift(start=95, end=105),
    )

    assert stream.get_num_drifts() == 1
    assert str(stream.get_drifts()[0]) == (
        "GradualDrift(position=100, start=95, end=105, width=10)"
    )
    # The MOA CLI is now produced on demand rather than being the implementation.
    assert "-w 10 -p 100" in stream.to_moa_stream()._CLI


def test_recurrent_concept_drift_stream_rejects_base_drift_template():
    with pytest.raises(ValueError, match="Unsupported drift transition type"):
        RecurrentConceptDriftStream(
            concept_list=recurrent_drift_concepts(),
            max_recurrences_per_concept=1,
            transition_type_template=Drift(position=100),
        )


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
        (TorchStream.from_classification(DATASET_C1, 3), "cat1", 5),  # type: ignore
        (TorchStream.from_classification(DATASET_C2, 2), "cat2", 5),  # type: ignore
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
    if isinstance(stream, (NumpyStream, TorchStream)):
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
        (TorchStream.from_regression(DATASET_N1), "num1"),
        (TorchStream.from_regression(DATASET_N2), "num2"),
    ],
)
def test_regression_stream(stream: Stream[RegressionInstance], target: str):
    numeric_attributes = NUMERIC_ATTRS.copy()
    numeric_attributes.remove(target)
    nominal_attributes = NOMINAL_ATTRS.copy()
    num_attributes = len(numeric_attributes) + len(nominal_attributes)

    # Stream treats nominal attributes as numeric
    if isinstance(stream, (NumpyStream, TorchStream)):
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
