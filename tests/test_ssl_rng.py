"""Random-state isolation for the torch-backed SSL learners.

CapyMOA learners are libraries: constructing one must not reseed the process-wide
generators, because the calling program -- not CapyMOA -- owns that state.

Scope note: this currently covers the ``numpy.random`` reseed in
:mod:`capymoa.ssl._osnn`, which reseeds a generator the module never uses. The
``random`` and ``torch`` reseeds in the same constructor are still load-bearing --
OSNN samples centers and initialises weights *during training* -- so isolating them
changes the interleaving of the shared stream and measurably moves the recorded
accuracies in ``test_ssl_classifiers``. That needs a maintainer decision, not a
drive-by change; see the comment in ``_osnn.py``.
"""

from __future__ import annotations

import random
from itertools import islice

import numpy as np
import pytest

from capymoa.stream.generator import SEA


def _train_osnn(seed: int, n_instances: int = 30):
    """Train an OSNN on a fixed data stream and return its final linear-layer weights."""
    pytest.markskip("torch")
    import torch

    from capymoa.ssl import OSNN

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    stream = SEA()
    learner = OSNN(
        schema=stream.get_schema(),
        num_center=3,
        window_size=5,
        optim_steps=1,
        seed=seed,
    )
    for instance in islice(stream, n_instances):
        learner.train(instance)

    return [p.detach().clone() for p in learner.Network.linear.parameters()]


def test_constructing_osnn_leaves_the_numpy_rng_untouched():
    """Regression: OSNN.__init__ reseeded numpy.random even though the module never uses it."""
    pytest.markskip("torch")
    from capymoa.ssl import OSNN

    np.random.seed(99)
    without = [float(np.random.rand()) for _ in range(3)]

    np.random.seed(99)
    OSNN()
    with_osnn = [float(np.random.rand()) for _ in range(3)]

    assert without == with_osnn, (
        "constructing OSNN shifted the caller's numpy.random stream: "
        f"{without} != {with_osnn}"
    )


def test_osnn_is_still_reproducible():
    """The removed reseed was doing no work, so reproducibility must be unchanged."""
    pytest.markskip("torch")
    import torch

    first = _train_osnn(seed=7)
    second = _train_osnn(seed=7)
    other = _train_osnn(seed=8)

    assert first, "expected OSNN to have trainable weights"
    assert all(torch.equal(a, b) for a, b in zip(first, second)), (
        "the same seed produced different weights"
    )
    assert not all(torch.equal(a, b) for a, b in zip(first, other)), (
        "different seeds produced identical weights"
    )
