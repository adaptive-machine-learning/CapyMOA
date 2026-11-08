"""Tests to ensure progress bars work correctly."""

import pytest
from pytest import CaptureFixture
from tqdm import tqdm

from capymoa.anomaly import HalfSpaceTrees, evaluate_anomaly
from capymoa.classifier import NoChange, evaluate_classifier
from capymoa.datasets import ElectricityTiny
from capymoa.ssl import evaluate_ssl
from capymoa.stream.generator import WaveformGenerator


def assert_pbar(capfd: CaptureFixture, startswith: str):
    _, err = capfd.readouterr()
    err: str = err.splitlines()[-1]
    assert err.startswith(startswith)


@pytest.mark.parametrize(
    "max_instances,instances",
    [
        (100, 100),
        (None, 2000),
        (3000, 2000),
    ],
)
def test_default(
    max_instances: int | None, instances: int, capfd: CaptureFixture
) -> None:
    stream = ElectricityTiny()
    classifier = NoChange(schema=stream.get_schema())
    evaluate_classifier(
        stream,
        classifier,
        optimise=False,
        max_instances=max_instances,
        progress_bar=True,
    )
    assert_pbar(capfd, "Eval 'NoChange' on 'ElectricityTiny':")


def test_ssl(capfd: CaptureFixture) -> None:
    stream = ElectricityTiny()
    classifier = NoChange(schema=stream.get_schema())
    evaluate_ssl(
        stream, classifier, optimise=False, progress_bar=True, max_instances=100
    )
    assert_pbar(capfd, "SSL Eval 'NoChange' on 'ElectricityTiny':")


def test_anomaly(capfd: CaptureFixture) -> None:
    stream = ElectricityTiny()
    classifier = HalfSpaceTrees(schema=stream.get_schema())
    evaluate_anomaly(
        stream, classifier, optimise=False, progress_bar=True, max_instances=100
    )
    assert_pbar(capfd, "AD Eval 'HalfSpaceTrees' on 'ElectricityTiny':")


def test_multiple_learners(capfd: CaptureFixture) -> None:
    stream = ElectricityTiny()
    classifiers = {
        "a": NoChange(schema=stream.get_schema()),
        "b": NoChange(schema=stream.get_schema()),
    }
    evaluate_classifier(stream, classifiers, progress_bar=True, max_instances=100)
    assert_pbar(capfd, "Eval 2 learners on ElectricityTiny:")


def test_no_length(capfd: CaptureFixture) -> None:
    generator = WaveformGenerator()
    classifier = NoChange(schema=generator.get_schema())
    evaluate_classifier(
        generator, classifier, optimise=False, max_instances=100, progress_bar=True
    )
    assert_pbar(capfd, "Eval 'NoChange' on 'WaveformGenerator':")


def test_disabled_progress_bar(capfd: CaptureFixture) -> None:
    stream = ElectricityTiny()
    classifier = NoChange(schema=stream.get_schema())
    evaluate_classifier(stream, classifier, optimise=False, progress_bar=False)
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


def test_tqdm(capfd: CaptureFixture) -> None:
    stream = ElectricityTiny()
    classifier = NoChange(schema=stream.get_schema())
    with tqdm(desc="Custom Message") as progress_bar:
        evaluate_classifier(
            stream, classifier, optimise=False, progress_bar=progress_bar
        )
    assert_pbar(capfd, "Custom Message:")
