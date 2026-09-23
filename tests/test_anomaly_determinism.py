"""Default-seed reproducibility for anomaly detectors.

Two detectors constructed with the same (default) seed must produce
identical score traces. This guards against entropy-seeded randomness
leaking in behind a seed=None default: `random.Random(None)` seeds from
system entropy, so an unseeded default silently makes every run
non-reproducible despite a documented "Random seed for reproducibility"
contract (the bug this file was written for lived exactly there in
StreamingIsolationForest).

Traces start after each detector's window-build point, where seeded
randomness first takes effect; sizing parameters (window sizes, tree
counts) are minimized purely to keep CI fast - the seed default itself,
which is what is under test, is left untouched.
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

N_SCORES = 80

# (constructor, instances to skip before comparing = window-build point)
CASES = [
    (lambda **kw: HalfSpaceTrees(window_size=64, **kw), 64),
    (lambda **kw: OnlineIsolationForest(window_size=64, num_trees=8, **kw), 64),
    (lambda **kw: StreamingIsolationForest(window_size=64, n_trees=10, **kw), 64),
    (lambda **kw: AdaptiveIsolationForest(window_size=64, n_trees=10, **kw), 64),
    (lambda **kw: RobustRandomCutForest(tree_size=64, n_trees=10, **kw), 64),
]


def _score_trace(make, skip):
    stream = TinyBlobs()
    detector = make(schema=stream.get_schema())
    scores = []
    for i, instance in enumerate(stream):
        if i >= skip + N_SCORES:
            break
        score = detector.score_instance(instance)
        if i >= skip:
            scores.append(score)
        detector.train(instance)
    return scores


@pytest.mark.parametrize(
    "make,skip",
    CASES,
    ids=[
        "HalfSpaceTrees",
        "OnlineIsolationForest",
        "StreamingIsolationForest",
        "AdaptiveIsolationForest",
        "RobustRandomCutForest",
    ],
)
def test_default_seed_is_reproducible(make, skip):
    first = _score_trace(make, skip)
    second = _score_trace(make, skip)

    assert first == second, (
        "two identically-constructed detectors produced different score "
        f"traces (first difference at offset "
        f"{next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)} "
        "after the window-build point). The seed default is likely "
        "entropy-seeded (random.Random(None) draws from system entropy)."
    )
