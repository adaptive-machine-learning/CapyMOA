import numpy as np
import pytest

from capymoa.drift.eval_detector import EvaluateDriftDetector


def test_ndt_is_mdt_over_max_delay():
    """``ndt`` restates ``mdt`` as a fraction of the acceptable delay."""
    metrics = EvaluateDriftDetector(max_delay=200).calc_performance(
        trues=np.array([1000, 2000]),
        preds=np.array([500, 1200, 1250, 2100]),
        tot_n_instances=2500,
    )

    assert metrics.mdt == pytest.approx(150.0)
    assert metrics.ndt == pytest.approx(0.75)


@pytest.mark.parametrize("max_delay", [50, 200, 400])
def test_ndt_rescales_while_mdt_does_not(max_delay: int):
    """The same detections give one delay in instances but different fractions of it."""
    trues = np.array([1000])
    preds = np.array([1150])

    metrics = EvaluateDriftDetector(max_delay=max_delay).calc_performance(
        trues=trues, preds=preds, tot_n_instances=2000
    )

    if max_delay < 150:  # the detection now falls outside the acceptable window
        assert np.isnan(metrics.mdt)
        assert np.isnan(metrics.ndt)
    else:
        assert metrics.mdt == pytest.approx(150.0)
        assert metrics.ndt == pytest.approx(150.0 / max_delay)


def test_ndt_is_nan_when_nothing_is_detected():
    """Delay is undefined with no successful detection, and must not read as zero."""
    metrics = EvaluateDriftDetector(max_delay=50).calc_performance(
        trues=np.array([1000]),
        preds=np.array([5]),
        tot_n_instances=2000,
    )

    assert metrics.recall == 0.0
    assert np.isnan(metrics.mdt)
    assert np.isnan(metrics.ndt)


def test_ndt_is_zero_for_an_immediate_detection():
    metrics = EvaluateDriftDetector(max_delay=200).calc_performance(
        trues=np.array([1000]),
        preds=np.array([1000]),
        tot_n_instances=2000,
    )

    assert metrics.mdt == pytest.approx(0.0)
    assert metrics.ndt == pytest.approx(0.0)


def test_ndt_reaches_one_at_the_deadline():
    metrics = EvaluateDriftDetector(max_delay=200).calc_performance(
        trues=np.array([1000]),
        preds=np.array([1200]),
        tot_n_instances=2000,
    )

    assert metrics.mdt == pytest.approx(200.0)
    assert metrics.ndt == pytest.approx(1.0)


def test_ndt_exceeds_one_for_a_gradual_drift():
    """Delay is measured from the drift start, but the window closes at ``end + max_delay``."""
    metrics = EvaluateDriftDetector(max_delay=200).calc_performance(
        trues=np.array([(1000, 1500)]),
        preds=np.array([1700]),
        tot_n_instances=2000,
    )

    assert metrics.tp == 1
    assert metrics.mdt == pytest.approx(700.0)
    assert metrics.ndt == pytest.approx(3.5)


def test_ndt_is_negative_for_an_early_detection():
    """``max_early_detection`` admits detections before the drift, so delays can be negative."""
    metrics = EvaluateDriftDetector(
        max_delay=200, max_early_detection=100
    ).calc_performance(
        trues=np.array([1000]),
        preds=np.array([950]),
        tot_n_instances=2000,
    )

    assert metrics.tp == 1
    assert metrics.mdt == pytest.approx(-50.0)
    assert metrics.ndt == pytest.approx(-0.25)


def test_ndt_averages_over_detected_drifts_only():
    """A missed drift lowers recall but must not be counted as a delay of zero."""
    metrics = EvaluateDriftDetector(max_delay=100).calc_performance(
        trues=np.array([1000, 2000, 3000]),
        preds=np.array([1020, 3080]),
        tot_n_instances=4000,
    )

    assert metrics.fn == 1
    assert metrics.mdt == pytest.approx(50.0)  # mean of 20 and 80, not of 20, 0 and 80
    assert metrics.ndt == pytest.approx(0.5)
