from capymoa.datasets._datasets import ElectricityTiny, CovtypeTiny

import pytest
from capymoa.evaluation.evaluation import prequential_ssl_evaluation
from capymoa.base import ClassifierSSL
from capymoa.stream import Stream
from functools import partial


def _make_osnn(**kwargs):
    pytest.markskip("torch")
    from capymoa.ssl import OSNN

    return OSNN(**kwargs)


def _make_sleade(**kwargs):
    pytest.markskip("torch")
    from capymoa.ssl import SLEADE

    return SLEADE(**kwargs)


def assert_ssl_evaluation(
    learner: ClassifierSSL,
    stream: Stream,
    expectation: float,
    label_probability: float = 0.01,
    max_instances: int = 1000,
):
    results = prequential_ssl_evaluation(
        stream=stream,
        learner=learner,
        label_probability=label_probability,
        window_size=10,
        max_instances=max_instances,
    )

    assert results["cumulative"].accuracy() == pytest.approx(expectation), (
        f"Expected accuracy of {expectation} but got {results['cumulative'].accuracy()}"
        + f" for learner {learner} on stream {stream}"
    )


@pytest.mark.parametrize(
    "learner_constructor, stream_constructor, expectation, label_probability",
    [
        pytest.param(
            partial(_make_osnn, optim_steps=10),
            ElectricityTiny,
            46.1,
            None,
            marks=pytest.mark.torch,
        ),
        pytest.param(
            partial(_make_osnn, optim_steps=10),
            CovtypeTiny,
            26.3,
            None,
            marks=pytest.mark.torch,
        ),
        pytest.param(
            partial(_make_sleade, ensemble_size=3),
            ElectricityTiny,
            50.8,
            None,
            marks=pytest.mark.torch,
        ),
        pytest.param(
            partial(_make_sleade, ensemble_size=3),
            CovtypeTiny,
            43.0,
            None,
            marks=pytest.mark.torch,
        ),
    ],
    ids=[
        "OSNN_ElectricityTiny",
        "OSNN_CovtypeTiny",
        "SLEADE_ElectricityTiny",
        "SLEADE_CovtypeTiny",
    ],
)
def test_ssl_classifiers(
    learner_constructor, stream_constructor, expectation, label_probability
):
    # The optimizer steps are set to 10 to speed up the test
    stream = stream_constructor()
    learner = learner_constructor(schema=stream.get_schema())

    if label_probability is None:
        label_probability = 0.01

    assert_ssl_evaluation(
        learner,
        stream,
        expectation,
        label_probability=label_probability,
    )


def test_ssl_delay_length_in_python_loop():
    """``delay_length`` works without the MOA fast path.

    It used to be implemented only in MOA, so the Python loop raised. A
    ``DriftStream`` composed in Python cannot use the fast path, which made the
    feature unreachable for drifting streams.
    """
    from capymoa.classifier import HoeffdingTree
    from capymoa.evaluation import prequential_ssl_evaluation
    from capymoa.stream.drift import AbruptDrift, DriftStream
    from capymoa.stream.generator import SEA

    class _CountingHoeffdingTree(HoeffdingTree):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.labeled_train_calls = 0

        def train(self, instance):
            self.labeled_train_calls += 1
            super().train(instance)

    stream = DriftStream(
        stream=[SEA(function=1), AbruptDrift(position=2000), SEA(function=3)]
    )

    calls = {}
    for delay in (0, 2000):
        stream.restart()
        learner = _CountingHoeffdingTree(schema=stream.get_schema())
        results = prequential_ssl_evaluation(
            stream=stream,
            learner=learner,
            max_instances=4000,
            window_size=1000,
            label_probability=0.1,
            delay_length=delay,
            optimise=False,
        )
        assert results["cumulative"].accuracy() > 0
        calls[delay] = learner.labeled_train_calls

    # Labels queued within ``delay`` of the end are never delivered, so a delay
    # strictly reduces the number of labeled training calls.
    assert calls[2000] < calls[0]


def test_ssl_rejects_negative_delay():
    from capymoa.classifier import HoeffdingTree
    from capymoa.evaluation import prequential_ssl_evaluation
    from capymoa.stream.generator import SEA

    stream = SEA(function=1)
    with pytest.raises(ValueError, match="delay_length must be zero or positive"):
        prequential_ssl_evaluation(
            stream=stream,
            learner=HoeffdingTree(schema=stream.get_schema()),
            max_instances=100,
            delay_length=-1,
            optimise=False,
        )
