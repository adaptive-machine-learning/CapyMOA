import inspect

import pytest

from capymoa.core.moa._cli import cli_str_drift_detector
from capymoa.drift import detectors
from capymoa.drift.base_detector import BaseDriftDetector, MOADriftDetector


def test_from_cli():
    cli = "-a 0.01"
    detector = detectors.ADWIN.from_cli(cli)
    assert isinstance(detector, detectors.ADWIN)
    assert cli_str_drift_detector(detector) == "(ADWINChangeDetector -a 0.01)"


def test_concept_and_data_drift_share_namespace():
    from capymoa.drift.detectors import ADWIN, KolmogorovSmirnov, MMD

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
