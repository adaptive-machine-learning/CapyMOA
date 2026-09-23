"""Default-construction reproducibility for anomaly detectors.

Two detectors constructed with identical (default) arguments must produce
identical score traces. This guards against entropy-seeded randomness
leaking in behind a seed=None default: `random.Random(None)` seeds from
system entropy, so an unseeded default silently makes every run
non-reproducible despite a documented "Random seed for reproducibility"
contract.

Warmup note: several detectors emit constant scores until their sliding
window fills (e.g. RSHash initializes its ensemble at instance
`window_size`). Comparisons therefore start after the largest default
warmup among the tested detectors.
"""

import pytest

from capymoa.anomaly import (
    AdaptiveIsolationForest,
    HalfSpaceTrees,
    OnlineIsolationForest,
    RobustRandomCutForest,
    StreamingIsolationForest,
)
from capymoa.anomaly.datasets import TinyBlobs

# Largest default warmup among tested detectors (RSHash window_size=1000).
WARMUP = 1001
N_SCORES = 300


def _score_trace(learner_constructor):
    stream = TinyBlobs()
    detector = learner_constructor(schema=stream.get_schema())
    scores = []
    for i, instance in enumerate(stream):
        if i >= WARMUP + N_SCORES:
            break
        score = detector.score_instance(instance)
        if i >= WARMUP:
            scores.append(score)
        detector.train(instance)
    return scores


@pytest.mark.parametrize(
    "learner_constructor",
    [
        HalfSpaceTrees,
        OnlineIsolationForest,
        StreamingIsolationForest,
        AdaptiveIsolationForest,
        RobustRandomCutForest,
    ],
)
def test_default_construction_is_reproducible(learner_constructor):
    first = _score_trace(learner_constructor)
    second = _score_trace(learner_constructor)

    assert first == second, (
        f"{learner_constructor.__name__}: two identically-constructed "
        f"detectors produced different score traces (first difference at "
        f"offset {next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)} "
        f"after warmup). The seed=None default is likely entropy-seeded "
        f"(random.Random(None) draws from system entropy)."
    )
