"""Tests to ensure progress bars work correctly."""

from typing import Optional
from pytest import CaptureFixture
from capymoa.stream.generator import WaveformGenerator
from capymoa.datasets import ElectricityTiny
from capymoa.classifier import NoChange
from capymoa.anomaly import HalfSpaceTrees
from capymoa.evaluation import (
    prequential_evaluation,
    prequential_ssl_evaluation,
    prequential_evaluation_anomaly,
    prequential_evaluation_multiple_learners,
)
import pytest
from tqdm import tqdm


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
    max_instances: Optional[int], instances: int, capfd: CaptureFixture
) -> None:
    stream = ElectricityTiny()
    classifier = NoChange(schema=stream.get_schema())
    prequential_evaluation(
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
    prequential_ssl_evaluation(
        stream, classifier, optimise=False, progress_bar=True, max_instances=100
    )
    assert_pbar(capfd, "SSL Eval 'NoChange' on 'ElectricityTiny':")


def test_anomaly(capfd: CaptureFixture) -> None:
    stream = ElectricityTiny()
    classifier = HalfSpaceTrees(schema=stream.get_schema())
    prequential_evaluation_anomaly(
        stream, classifier, optimise=False, progress_bar=True, max_instances=100
    )
    assert_pbar(capfd, "AD Eval 'HalfSpaceTrees' on 'ElectricityTiny':")


def test_multiple_learners(capfd: CaptureFixture) -> None:
    stream = ElectricityTiny()
    classifiers = {
        "a": NoChange(schema=stream.get_schema()),
        "b": NoChange(schema=stream.get_schema()),
    }
    prequential_evaluation_multiple_learners(
        stream, classifiers, progress_bar=True, max_instances=100
    )
    assert_pbar(capfd, "Eval 2 learners on ElectricityTiny:")


def test_no_length(capfd: CaptureFixture) -> None:
    generator = WaveformGenerator()
    classifier = NoChange(schema=generator.get_schema())
    prequential_evaluation(
        generator, classifier, optimise=False, max_instances=100, progress_bar=True
    )
    assert_pbar(capfd, "Eval 'NoChange' on 'WaveformGenerator':")


def test_disabled_progress_bar(capfd: CaptureFixture) -> None:
    stream = ElectricityTiny()
    classifier = NoChange(schema=stream.get_schema())
    prequential_evaluation(stream, classifier, optimise=False, progress_bar=False)
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


def test_tqdm(capfd: CaptureFixture) -> None:
    stream = ElectricityTiny()
    classifier = NoChange(schema=stream.get_schema())
    with tqdm(desc="Custom Message") as progress_bar:
        prequential_evaluation(
            stream, classifier, optimise=False, progress_bar=progress_bar
        )
    assert_pbar(capfd, "Custom Message:")
