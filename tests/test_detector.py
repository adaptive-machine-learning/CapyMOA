import inspect

import numpy as np
import pytest

from capymoa.core.moa._cli import cli_str_drift_detector
from capymoa.drift import detectors
from capymoa.drift.base_detector import BaseDriftDetector, MOADriftDetector
from capymoa.drift.detectors.data_drift import BaseDataDriftDetector


def test_from_cli():
    cli = "-a 0.01"
    detector = detectors.ADWIN.from_cli(cli)
    assert isinstance(detector, detectors.ADWIN)
    assert cli_str_drift_detector(detector) == "(ADWINChangeDetector -a 0.01)"


def test_concept_and_data_drift_share_namespace():
    from capymoa.drift.detectors import ADWIN, MMD, KolmogorovSmirnov

    assert ADWIN is detectors.ADWIN
    assert KolmogorovSmirnov is detectors.KolmogorovSmirnov
    assert MMD is detectors.MMD


@pytest.mark.parametrize("detector_name", detectors.__all__)
def test_constructors(detector_name: str):
    detector_cls = getattr(detectors, detector_name)

    if not inspect.isclass(detector_cls) or not issubclass(
        detector_cls, BaseDriftDetector
    ):
        pytest.skip(f"{detector_name} is not a drift detector.")
    if inspect.isabstract(detector_cls):
        pytest.skip(f"{detector_name} is abstract.")

    # Skip any that require positional arguments
    parameters = inspect.signature(detector_cls).parameters
    if any(p.default is inspect.Parameter.empty for p in parameters.values()):
        pytest.skip(f"{detector_name} has required positional arguments.")

    detector = detector_cls()
    assert isinstance(detector, BaseDriftDetector)

    # Test only MOA drift detectors
    if isinstance(detector, MOADriftDetector):
        assert isinstance(detector_cls.from_cli(""), detector_cls)
        assert detector._moa_detector_type is not None, (
            "MOADriftDetector MUST set _moa_detector_type appropriately."
        )
        assert isinstance(detector.moa_detector, detector._moa_detector_type), (
            "MOA detector instance must be of type specified by _moa_detector_type."
        )


def test_requires_fit_flag():
    from capymoa.drift.detectors import ADWIN, KolmogorovSmirnov

    assert ADWIN.REQUIRES_FIT is False
    assert ADWIN().REQUIRES_FIT is False
    assert KolmogorovSmirnov.REQUIRES_FIT is True
    assert KolmogorovSmirnov(window_size=10).REQUIRES_FIT is True


def test_is_fitted_after_explicit_fit():
    import numpy as np

    from capymoa.drift.detectors import KolmogorovSmirnov

    detector = KolmogorovSmirnov(window_size=10)
    assert detector.is_fitted is False
    detector.fit(np.zeros((20, 2)))
    assert detector.is_fitted is True


def test_auto_fit_via_add_element():
    import numpy as np

    from capymoa.drift.detectors import KolmogorovSmirnov

    rng = np.random.default_rng(0)
    detector = KolmogorovSmirnov(window_size=5, auto_fit_samples=10)
    assert detector.REQUIRES_FIT is True
    assert detector.is_fitted is False
    for x in rng.normal(size=(9, 2)):
        detector.add_element(x)
        assert detector.is_fitted is False
    detector.add_element(rng.normal(size=2))
    assert detector.is_fitted is True


def test_add_element_without_reference_raises():
    import numpy as np

    from capymoa.drift.detectors import KolmogorovSmirnov

    detector = KolmogorovSmirnov(window_size=10)
    with pytest.raises(RuntimeError, match="requires reference data"):
        detector.add_element(np.zeros(2))


def test_compare_without_reference_raises():
    import numpy as np

    from capymoa.drift.detectors import KolmogorovSmirnov

    detector = KolmogorovSmirnov(window_size=10)
    with pytest.raises(RuntimeError, match="requires reference data"):
        detector.compare(np.zeros((10, 2)))


# ---------------------------------------------------------------------------
# Data drift detector tests
# ---------------------------------------------------------------------------

# Sanity-check parameters: small windows for speed, lenient thresholds.
_DATA_DRIFT_DETECTORS = [
    ("AndersonDarling", {"window_size": 30}),
    ("BNDM", {"window_size": 30, "threshold": 0.1, "max_depth": 3}),
    ("ChiSquare", {"window_size": 30}),
    ("CramerVonMises", {"window_size": 30}),
    ("D3", {"window_size": 30, "threshold": 0.6, "seed": 0}),
    ("EnergyDistance", {"window_size": 30, "threshold": 0.5}),
    ("Hellinger", {"window_size": 30, "num_bins": 5, "threshold": 0.2}),
    ("JensenShannon", {"window_size": 30, "num_bins": 5, "threshold": 0.4}),
    ("KLDivergence", {"window_size": 30, "num_bins": 5, "threshold": 0.3}),
    ("KolmogorovSmirnov", {"window_size": 30}),
    ("MMD", {"window_size": 30, "n_permutations": 20, "sigma": 1.0}),
    ("PSI", {"window_size": 30, "num_bins": 5, "threshold": 0.3}),
    ("Wasserstein", {"window_size": 30, "threshold": 0.3}),
]


def _make_detector(name, kwargs):
    cls = getattr(detectors, name)
    return cls(**kwargs)


@pytest.mark.parametrize(
    "name,kwargs", _DATA_DRIFT_DETECTORS, ids=[n for n, _ in _DATA_DRIFT_DETECTORS]
)
def test_data_drift_instantiation(name, kwargs):
    """Every data drift detector can be instantiated and has REQUIRES_FIT=True."""
    det = _make_detector(name, kwargs)
    assert isinstance(det, BaseDataDriftDetector)
    assert det.REQUIRES_FIT is True
    assert det.is_fitted is False


@pytest.mark.parametrize(
    "name,kwargs", _DATA_DRIFT_DETECTORS, ids=[n for n, _ in _DATA_DRIFT_DETECTORS]
)
def test_data_drift_fit_and_detect(name, kwargs):
    """Fit on stable data, stream shifted data, check drift is detected."""
    rng = np.random.default_rng(42)
    n_features = 2
    ws = kwargs["window_size"]

    if name == "ChiSquare":
        ref = rng.choice(["a", "b", "c"], size=(100, n_features), p=[0.5, 0.3, 0.2])
        # Stream enough to fill the window with heavily shifted distribution
        shifted = rng.choice(
            ["a", "b", "c"], size=(ws + 10, n_features), p=[0.05, 0.05, 0.9]
        )
    else:
        ref = rng.normal(0, 1, size=(100, n_features))
        # Large shift (mean=10) to reliably trigger detection
        shifted = rng.normal(10, 1, size=(ws + 10, n_features))

    det = _make_detector(name, kwargs)
    det.fit(ref)
    assert det.is_fitted is True

    for x in shifted:
        det.add_element(x)

    assert det.detection_index, f"{name} should detect drift on shifted data"


@pytest.mark.parametrize(
    "name,kwargs", _DATA_DRIFT_DETECTORS, ids=[n for n, _ in _DATA_DRIFT_DETECTORS]
)
def test_data_drift_get_params_has_auto_fit(name, kwargs):
    """get_params() must include auto_fit_samples."""
    det = _make_detector(name, kwargs)
    params = det.get_params()
    assert "auto_fit_samples" in params
    assert "window_size" in params


@pytest.mark.parametrize(
    "name,kwargs", _DATA_DRIFT_DETECTORS, ids=[n for n, _ in _DATA_DRIFT_DETECTORS]
)
def test_data_drift_warning_zone_is_bool(name, kwargs):
    """detected_warning() must return False (not None) for data drift detectors."""
    det = _make_detector(name, kwargs)
    assert det.detected_warning() is False


# ---------------------------------------------------------------------------
# Reset-then-replay contract: reset() must restore the detector to its
# as-constructed state, so replaying the same input reproduces the same
# flag trace. The MOA-backed detectors inherit a reset() that re-creates
# the MOA object; the pure-Python detectors must clear their own state.
# ---------------------------------------------------------------------------


_RESET_REPLAY_DETECTORS = [
    ("ABCD", {"model_id": "pca", "maximum_absolute_value": 0.2}),
    ("ADWIN", {}),
    ("CUSUM", {}),
    ("DDM", {}),
    ("EDDM", {}),
    ("EWMAChart", {}),
    ("GeometricMovingAverage", {}),
    ("HDDMAverage", {}),
    ("HDDMWeighted", {}),
    ("OPTWIN", {"w_length_max": 200}),
    ("PageHinkley", {}),
    ("RDDM", {}),
    ("SEED", {}),
    ("STEPD", {}),
]


def _replay_trace(detector, stream):
    out = []
    for x in stream:
        detector.add_element(float(x))
        out.append(
            (bool(detector.detected_change()), bool(detector.detected_warning()))
        )
    return out


@pytest.mark.parametrize(
    "name,kwargs", _RESET_REPLAY_DETECTORS, ids=[n for n, _ in _RESET_REPLAY_DETECTORS]
)
def test_reset_then_replay_reproduces_trace(name, kwargs):
    """reset() followed by the same input must reproduce the same trace."""
    rng = np.random.default_rng(1234)
    stream = np.concatenate([rng.normal(0, 1, 600), rng.normal(3, 1, 600)]).astype(
        np.float64
    )

    detector = getattr(detectors, name)(**kwargs)
    first = _replay_trace(detector, stream)
    detector.reset()
    second = _replay_trace(detector, stream)

    assert first == second, (
        f"{name}: replay after reset() diverges from the first pass "
        f"(first difference at index "
        f"{next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)}) - "
        "reset() left stale detector state behind"
    )


@pytest.mark.parametrize(
    "name,kwargs", _RESET_REPLAY_DETECTORS, ids=[n for n, _ in _RESET_REPLAY_DETECTORS]
)
def test_repeated_reset_replay_is_stable(name, kwargs):
    """Replay traces must be identical across successive reset() calls."""
    rng = np.random.default_rng(1234)
    stream = rng.normal(0, 1, 300).astype(np.float64)

    detector = getattr(detectors, name)(**kwargs)
    first_pass = _replay_trace(detector, stream)
    detector.reset()
    second_pass = _replay_trace(detector, stream)
    detector.reset()
    third_pass = _replay_trace(detector, stream)

    assert second_pass == third_pass, (
        f"{name}: replay after a second reset() differs from the replay "
        "after the first - state survives reset() and accumulates"
    )
    assert first_pass == second_pass, (
        f"{name}: replay after reset() differs from the as-constructed trace"
    )
