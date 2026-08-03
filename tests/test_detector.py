from capymoa.drift import detectors
from capymoa._cli import cli_str_drift_detector
from capymoa.drift.base_detector import BaseDriftDetector, MOADriftDetector
import inspect
import pytest


def test_from_cli():
    cli = "-a 0.01"
    detector = detectors.ADWIN.from_cli(cli)
    assert isinstance(detector, detectors.ADWIN)
    assert cli_str_drift_detector(detector) == "(ADWINChangeDetector -a 0.01)"


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
