"""Seed-reproducibility contract for GDumb's offline fit.

GDumb seeds its coreset sampler with a private generator, but its offline
``gdumb_fit`` step trains on a shuffled DataLoader. If that shuffle is drawn
from the global torch RNG, two instances built and fitted identically produce
different predictions depending on unrelated ambient RNG state (e.g. another
model initialised earlier in the same process), silently breaking run
reproducibility. These tests pin the contract: the documented ``seed``
argument must determine the fit outcome.
"""

import pytest

pytestmark = pytest.markskip("torch")

import torch
from torch import nn

from capymoa.ocl.strategy import GDumb
from capymoa.stream import Schema

N_FEATURES = 20
N_CLASSES = 4
N_TEST = 200


def _schema() -> Schema:
    return Schema.from_custom(
        features=[f"f{i}" for i in range(N_FEATURES)] + ["y"],
        target="y",
        categories={"y": [str(i) for i in range(N_CLASSES)]},
    )


def _model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(N_FEATURES, 16),
        nn.ReLU(),
        nn.Linear(16, N_CLASSES),
    )


def _new_gdumb(seed: int, capacity: int) -> GDumb:
    return GDumb(
        schema=_schema(),
        model=_model(),
        epochs=3,
        batch_size=32,
        capacity=capacity,
        lr=0.01,
        seed=seed,
    )


def _train_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, N_FEATURES, generator=generator)
    y = torch.randint(0, N_CLASSES, (n,), generator=generator)
    return x, y


def _prediction_trace(
    seed: int,
    capacity: int,
    fit_ambient_seed: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
) -> torch.Tensor:
    """Fill the coreset, fit offline, and return quantized test predictions.

    Construction and coreset filling happen under a fixed ambient RNG state so
    the model initialisation is identical across calls; only the ambient state
    at fit time (``fit_ambient_seed``) varies between calls. Predictions are
    quantized to integers so the contract asserts reproducible ordering
    without requiring bitwise-identical floats.
    """
    torch.manual_seed(0)
    learner = _new_gdumb(seed, capacity)
    for i in range(len(train_y)):
        learner.batch_train(train_x[i : i + 1], train_y[i : i + 1])
    torch.manual_seed(fit_ambient_seed)
    learner.gdumb_fit()
    with torch.no_grad():
        proba = learner.model(test_x).softmax(dim=1)
    return (proba * 100).round().to(torch.int16)


def test_same_seed_identical_predictions_regardless_of_ambient_rng():
    """Identical fits must agree even when ambient RNG states differ.

    The global torch RNG is reseeded before each fit to simulate unrelated
    code drawing from it between GDumb fits.
    """
    n_train = 400
    train_x, train_y = _train_data(n_train, 42)
    test_x = torch.randn(
        N_TEST, N_FEATURES, generator=torch.Generator().manual_seed(123)
    )

    # capacity == n_train: the coreset never replaces anything, so the seed
    # cannot influence the fit through the sampler; only the shuffle can.
    first = _prediction_trace(7, n_train, 0, train_x, train_y, test_x)
    second = _prediction_trace(7, n_train, 999, train_x, train_y, test_x)

    mismatches = (first != second).any(dim=1).sum().item()
    assert mismatches == 0, (
        "GDumb: same seed and same data produced different predictions "
        f"({mismatches}/{N_TEST} mismatched) when the global torch RNG state "
        "differed at fit time - the offline fit shuffle is not controlled "
        "by the seed"
    )


def test_different_seeds_change_stochastic_fit():
    """The coreset replacement sampler consumes the seed: on an overflowing
    buffer, different seeds must lead to different fits."""
    n_train = 400
    capacity = 200  # half the examples arrive into a full buffer
    train_x, train_y = _train_data(n_train, 1)
    test_x = torch.randn(
        N_TEST, N_FEATURES, generator=torch.Generator().manual_seed(123)
    )

    first = _prediction_trace(1, capacity, 0, train_x, train_y, test_x)
    second = _prediction_trace(42, capacity, 0, train_x, train_y, test_x)

    mismatches = (first != second).any(dim=1).sum().item()
    assert mismatches > 0, (
        "GDumb: predictions identical under seeds 1 and 42 despite buffer "
        "replacements - the seed is not reaching the coreset sampler"
    )
