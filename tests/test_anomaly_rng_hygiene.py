"""Contract: StreamRHF must not mutate the global numpy RNG state.

``choose_split_attribute`` used to call ``np.random.seed(...)`` - reseeding
the *process-global* numpy generator - once per tree node during forest
construction and every window rebuild, and the per-node split-value draw
rode the global stream afterwards. Any other code sharing the process
(numpy users, other learners, the test suite itself) would see its random
sequence rewritten mid-run. The forest's own outputs were already
determined by per-node seeds; this test pins that fitting the detector
leaves the global generator untouched.
"""

import numpy as np

from capymoa.anomaly import StreamRHF
from capymoa.anomaly.datasets import TinyBlobs


def _numpy_state(state) -> tuple:
    """Comparable representation of a numpy global RNG state."""
    return (state[0], state[1].tobytes(), state[2], state[3], state[4])


def test_stream_rhf_training_does_not_touch_global_numpy_rng():
    stream = TinyBlobs()
    learner = StreamRHF(
        schema=stream.get_schema(), num_trees=5, max_height=3, window_size=10
    )

    state_before = _numpy_state(np.random.get_state())
    for i, instance in enumerate(stream):
        if i >= 30:  # enough to cross the window rebuild path
            break
        learner.score_instance(instance)
        learner.train(instance)
    state_after = _numpy_state(np.random.get_state())

    assert state_before == state_after, (
        "StreamRHF training mutated the global numpy RNG state "
        "(np.random.seed / np.random.uniform use inside tree construction); "
        "use node-local RandomState/Generator instances instead"
    )
