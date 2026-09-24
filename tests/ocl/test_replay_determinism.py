"""Seed-reproducibility contract for replay-buffer sampling.

``ReplayBuffer.sample`` draws batch indices from the global torch RNG even
though replay buffers are constructed with a seeded private generator (which
``ReservoirSampler.update`` already uses for reservoir replacement). Strategies
that replay during training - ExperienceReplay, RAR - therefore consume
ambient RNG state, so identical seeds and identical data produce different
training trajectories depending on unrelated code drawing from the global RNG
in the same process. This test pins the contract: replay sampling must be
determined by the buffer's seed alone.
"""

import pytest

pytestmark = pytest.markskip("torch")

import torch

from capymoa.classifier import Finetune
from capymoa.core.torch.ann import Perceptron
from capymoa.ocl.strategy import ExperienceReplay
from capymoa.stream import Schema

N_FEATURES = 20
N_CLASSES = 4
BATCH = 32
N_TEST = 200


def _schema() -> Schema:
    return Schema.from_custom(
        features=[f"f{i}" for i in range(N_FEATURES)] + ["y"],
        target="y",
        categories={"y": [str(i) for i in range(N_CLASSES)]},
    )


def _train_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, N_FEATURES, generator=generator)
    y = torch.randint(0, N_CLASSES, (n,), generator=generator)
    return x, y


def _new_strategy() -> ExperienceReplay:
    return ExperienceReplay(Finetune(_schema(), Perceptron), buffer_size=200)


def _prediction_trace(
    ambient_seed: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
) -> torch.Tensor:
    """Train a fresh ExperienceReplay strategy and return quantized predictions.

    Construction and the first half of training run under a fixed ambient RNG
    state so both traces start from identical model and buffer state; only the
    ambient state during the second half of training (``ambient_seed``)
    differs between calls. Predictions are quantized to integers so the
    contract asserts reproducible sampling without requiring bitwise-identical
    floats.
    """
    n_fill = 6 * BATCH
    n_total = 10 * BATCH
    torch.manual_seed(0)
    strategy = _new_strategy()
    for i in range(0, n_fill, BATCH):
        strategy.batch_train(train_x[i : i + BATCH], train_y[i : i + BATCH])
    torch.manual_seed(ambient_seed)
    for i in range(n_fill, n_total, BATCH):
        strategy.batch_train(train_x[i : i + BATCH], train_y[i : i + BATCH])
    with torch.no_grad():
        proba = strategy.learner.batch_predict_proba(test_x)
    return (proba * 100).round().to(torch.int16)


def test_replay_sampling_identical_regardless_of_ambient_rng():
    """Same seed must give the same training trajectory even when the global
    torch RNG state differs during training."""
    train_x, train_y = _train_data(10 * BATCH, 42)
    test_x = torch.randn(
        N_TEST, N_FEATURES, generator=torch.Generator().manual_seed(123)
    )

    first = _prediction_trace(111, train_x, train_y, test_x)
    second = _prediction_trace(999, train_x, train_y, test_x)

    mismatches = (first != second).any(dim=1).sum().item()
    assert mismatches == 0, (
        "ExperienceReplay: same seed and same data produced different "
        f"predictions ({mismatches}/{N_TEST} mismatched) when the global torch "
        "RNG state differed during training - replay sampling is drawing from "
        "the global RNG instead of the buffer's seeded generator"
    )
