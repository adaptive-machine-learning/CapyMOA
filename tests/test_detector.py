import inspect
import json

import pytest

from capymoa.core.moa._cli import cli_str_drift_detector
from capymoa.drift import detectors
from capymoa.drift.base_detector import BaseDriftDetector, MOADriftDetector


def test_from_cli():
    cli = "-a 0.01"
    detector = detectors.ADWIN.from_cli(cli)
    assert isinstance(detector, detectors.ADWIN)
    assert cli_str_drift_detector(detector) == "(ADWINChangeDetector -a 0.01)"


@pytest.mark.parametrize("detector_name", detectors.__all__)
def test_get_params_uses_native_python_types(detector_name: str):
    detector_cls = getattr(detectors, detector_name)

    # Skip any that require positional arguments
    parameters = inspect.signature(detector_cls).parameters
    if any(p.default is inspect.Parameter.empty for p in parameters.values()):
        pytest.skip(f"{detector_name} has required positional arguments.")

    params = detector_cls().get_params()

    if detector_name == "EDDM" and not params:
        pytest.xfail(
            "EDDM exposes no MOA options yet; being added upstream (Waikato/moa#335)"
        )

    # Keys and values must be native Python types, not JPype java.lang.String.
    # JString keys/values break **kwargs unpacking, json serialization, and any
    # interop that hands the dict to non-CapyMOA code.
    assert params, f"{detector_name}.get_params() returned an empty dict"
    for key, value in params.items():
        assert type(key) is str, (
            f"{detector_name}: key {key!r} is {type(key).__name__}, not str"
        )
        is_native_scalar = isinstance(value, str | int | float | bool) or (
            value is None
        )
        assert is_native_scalar, (
            f"{detector_name}: value for {key!r} is {type(value).__name__}, "
            "not a native scalar"
        )
    json.dumps(params)  # full dict must be JSON-serializable


@pytest.mark.parametrize("detector_name", detectors.__all__)
def test_get_params_supports_dict_interop(detector_name: str):
    detector_cls = getattr(detectors, detector_name)

    # Skip any that require positional arguments
    parameters = inspect.signature(detector_cls).parameters
    if any(p.default is inspect.Parameter.empty for p in parameters.values()):
        pytest.skip(f"{detector_name} has required positional arguments.")

    params = detector_cls().get_params()

    # Standard-library interop must work without manual conversion.
    assert json.dumps(params)  # not just truthy: must not raise
    unpacked = {**params}
    assert unpacked == params


@pytest.mark.parametrize("detector_name", detectors.__all__)
def test_constructors(detector_name: str):
    detector_cls = getattr(detectors, detector_name)

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
