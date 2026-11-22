"""Tests for :mod:`capymoa.events` and the events drift detectors emit."""

import inspect
from dataclasses import fields

import numpy as np
import pytest

from capymoa.drift import detectors
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN, DDM, KolmogorovSmirnov
from capymoa.drift.events import DriftEvent, DriftOrWarningEvent, WarningEvent
from capymoa.events import Dispatcher, Event


class Parent(Event):
    pass


class Child(Parent):
    pass


class Other(Event):
    pass


def _record(log: list, name: str):
    return lambda event: log.append((name, type(event).__name__))


def test_subscribing_to_a_parent_receives_subclasses():
    log = []
    dispatcher = Dispatcher()
    dispatcher.subscribe(Parent, _record(log, "parent"))
    dispatcher.subscribe(Child, _record(log, "child"))
    dispatcher.notify(Child())
    dispatcher.notify(Parent())
    assert log == [
        ("parent", "Child"),
        ("child", "Child"),
        ("parent", "Parent"),
    ]


def test_callbacks_run_in_subscription_order_across_types():
    log = []
    dispatcher = Dispatcher()
    dispatcher.subscribe(Child, _record(log, "first"))
    dispatcher.subscribe(None, _record(log, "second"))
    dispatcher.subscribe(Parent, _record(log, "third"))
    dispatcher.notify(Child())
    assert [name for name, _ in log] == ["first", "second", "third"]


def test_none_receives_every_event():
    log = []
    dispatcher = Dispatcher()
    dispatcher.subscribe(None, _record(log, "all"))
    dispatcher.notify(Other())
    dispatcher.notify(Child())
    assert log == [("all", "Other"), ("all", "Child")]


def test_unsubscribe():
    log = []
    callback = _record(log, "x")
    dispatcher = Dispatcher()
    dispatcher.subscribe(Parent, callback)
    dispatcher.unsubscribe(Parent, callback)
    dispatcher.notify(Parent())
    assert log == []
    with pytest.raises(ValueError, match="not subscribed to Parent"):
        dispatcher.unsubscribe(Parent, callback)


def test_a_callback_may_unsubscribe_itself_while_notified():
    log = []
    dispatcher = Dispatcher()

    def once(event):
        log.append("once")
        dispatcher.unsubscribe(Parent, once)

    dispatcher.subscribe(Parent, once)
    dispatcher.subscribe(Parent, _record(log, "after"))
    dispatcher.notify(Parent())
    dispatcher.notify(Parent())
    assert log == ["once", ("after", "Parent"), ("after", "Parent")]


def test_subscribe_rejects_a_type_that_is_never_emitted():
    dispatcher = Dispatcher(emits=[Child])
    with pytest.raises(ValueError, match="Other is never emitted here"):
        dispatcher.subscribe(Other, print)
    # A parent of an emitted type, and None, are accepted.
    dispatcher.subscribe(Parent, print)
    dispatcher.subscribe(Event, print)
    dispatcher.subscribe(None, print)


def test_subscribe_rejects_a_non_event_type():
    with pytest.raises(TypeError, match="Event subclass"):
        Dispatcher().subscribe("DriftEvent", print)  # type: ignore[arg-type]


def test_ocl_events_module_is_an_alias():
    import capymoa.events
    import capymoa.ocl.events

    for name in ("Dispatcher", "Event", "Handler"):
        assert getattr(capymoa.ocl.events, name) is getattr(capymoa.events, name)


@pytest.mark.torch
def test_ocl_task_events_do_not_subclass_each_other():
    """Strategies subscribe to ``TrainTaskBegin`` to run once per task.

    With subclass delivery, any event subclassing it would also call them.
    """
    pytest.markskip("torch")
    from capymoa.ocl.evaluate import events

    groups = {
        events.TrainTaskEvent: [
            events.TrainTaskBegin,
            events.TrainBatchPredict,
            events.TrainTaskEnd,
        ],
        events.TestTaskEvent: [
            events.TestTaskBegin,
            events.EvalBatchPredict,
            events.TestTaskEnd,
        ],
    }
    for base, members in groups.items():
        for a in members:
            assert issubclass(a, base)
            for b in members:
                assert a is b or not issubclass(a, b), f"{a} subclasses {b}"
        # Moving the fields to the base must not drop or reorder them.
        shared = [f.name for f in fields(base)]
        for member in members:
            assert [f.name for f in fields(member)][: len(shared)] == shared


def _abrupt_change(n: int = 1000, seed: int = 0) -> np.ndarray:
    """Error indicators whose rate jumps from 0.1 to 0.6 halfway."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.random(n) < 0.1, rng.random(n) < 0.6]).astype(float)


# Data drift detectors need a window size; STUDD needs a student learner.
_REQUIRED = {"window_size": 50}
# Fewer permutations keep MMD fast enough for a unit test.
_OVERRIDES = {"MMD": {"n_permutations": 5}}


def _detector_names():
    for name in detectors.__all__:
        cls = getattr(detectors, name)
        if not (inspect.isclass(cls) and issubclass(cls, BaseDriftDetector)):
            continue
        if inspect.isabstract(cls):
            continue
        parameters = inspect.signature(cls).parameters.values()
        required = {p.name for p in parameters if p.default is inspect.Parameter.empty}
        if not required <= _REQUIRED.keys():
            continue
        yield name


def _build(name: str) -> BaseDriftDetector:
    cls = getattr(detectors, name)
    parameters = inspect.signature(cls).parameters
    kwargs = {k: v for k, v in _REQUIRED.items() if k in parameters}
    return cls(**kwargs, **_OVERRIDES.get(name, {}))


@pytest.mark.parametrize("name", sorted(_detector_names()))
def test_events_match_the_detector_history(name: str):
    detector = _build(name)
    if detector.REQUIRES_FIT:
        # Data drift detectors compare a window on every element: keep it short.
        stream = _abrupt_change(n=300)
        detector.fit(stream[:300].reshape(-1, 1))
    else:
        stream = _abrupt_change()

    warnings, drifts = [], []
    if WarningEvent in detector.emits:
        detector.events.subscribe(WarningEvent, lambda e: warnings.append(e.index))
    detector.events.subscribe(DriftEvent, lambda e: drifts.append(e.index))
    both = []
    detector.events.subscribe(DriftOrWarningEvent, both.append)

    for value in stream:
        detector.add_element(value)

    assert drifts == detector.detection_index
    assert warnings == detector.warning_index
    assert len(both) == len(drifts) + len(warnings)
    assert all(event.detector is detector for event in both)


def test_events_fire_after_the_state_is_updated():
    detector = ADWIN()
    seen = []
    detector.events.subscribe(
        DriftEvent,
        lambda e: seen.append(
            (e.detector.detected_change(), e.detector.detection_index[-1] == e.index)
        ),
    )
    for value in [0.0] * 1000 + [1.0] * 1000:
        detector.add_element(value)
    assert seen and all(state == (True, True) for state in seen)


def test_a_subclass_calling_super_emits_once():
    class Wrapped(ADWIN):
        def add_element(self, element: float) -> None:
            super().add_element(element)

    detector = Wrapped()
    drifts = []
    detector.events.subscribe(DriftEvent, lambda e: drifts.append(e.index))
    for value in [0.0] * 1000 + [1.0] * 1000:
        detector.add_element(value)
    assert drifts == detector.detection_index != []


def test_reset_keeps_subscriptions():
    detector = DDM()
    drifts = []
    detector.events.subscribe(DriftEvent, lambda e: drifts.append(e.index))
    detector.reset(clean_history=True)
    for value in _abrupt_change():
        detector.add_element(value)
    assert drifts == detector.detection_index != []


def test_each_detector_has_its_own_dispatcher():
    a, b = ADWIN(), ADWIN()
    assert a.events is not b.events


def test_data_drift_detectors_do_not_offer_warnings():
    detector = KolmogorovSmirnov(window_size=100)
    assert detector.emits == (DriftEvent,)
    with pytest.raises(ValueError, match="WarningEvent is never emitted"):
        detector.events.subscribe(WarningEvent, print)
    detector.events.subscribe(DriftOrWarningEvent, print)
