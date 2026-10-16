"""Cross-process reproducibility contract for RSHash.

``RSHashCountMinSketch._indices`` maps a discretised vector to one slot per
sketch table. That mapping used to go through the builtin ``hash()``, which
CPython salts for ``bytes``/``tuple`` inputs from ``PYTHONHASHSEED``. The
salt is randomized per process, so two runs of the same seeded detector on the
same stream produced different scores in different processes, while the
in-process test kept passing: within one process the salt is constant, so the
only observable effect is a sparse set of slot collisions.

The assertion therefore has to cross a process boundary. Each child is launched
with an explicit ``PYTHONHASHSEED`` and returns a digest of its score trace.

Note on cost: the config below is deliberately small (``m=20`` components would
not expose the bug at all). The number of sketch tables examined grows with
``m * w * n_instances``, and a differing hash seed only changes a score when
two vectors land in the same slot somewhere in that set, so the divergence rate
at small ``m`` is too low to catch reliably. ``m=300, w=4`` is the smallest
configuration that separates every ``PYTHONHASHSEED`` tried during development.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from capymoa.anomaly import RSHash
from capymoa.anomaly._rs_hash import RSHashCountMinSketch
from capymoa.anomaly.datasets import TinyBlobs

# Kept in sync with the parameters of the cross-process child script below.
SEED = 42
COMPONENTS = 300
WINDOW = 64
INSTANCES = 300

CHILD = f"""
import hashlib, sys

from capymoa.anomaly import RSHash
from capymoa.anomaly.datasets import TinyBlobs

stream = TinyBlobs()
learner = RSHash(
    stream.get_schema(), seed={SEED}, m={COMPONENTS}, s={WINDOW}, w=4, p=10000
)

scores = []
for index, instance in enumerate(stream):
    if index >= {INSTANCES}:
        break
    scores.append(learner.score_instance(instance))
    learner.train(instance)

sys.stdout.write(hashlib.sha256(repr([repr(s) for s in scores]).encode()).hexdigest())
"""


def _score_digest_in_subprocess(python_hash_seed: str) -> str:
    """Return the digest of a seeded RSHash score trace from a fresh process.

    The child inherits this process's environment apart from
    ``PYTHONHASHSEED``, so Java discovery and the editable install work exactly
    as they do for the test process itself.
    """
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = python_hash_seed
    result = subprocess.run(
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    digest = result.stdout.strip()
    # Guard against a vacuous comparison: two children that both printed
    # nothing would otherwise compare equal and the test would pass for free.
    assert len(digest) == 64, f"child printed {digest!r}, expected a sha256 digest"
    return digest


def _discretized_vectors(count: int = 500):
    """Yield the kind of int64 vectors ``_discretize`` hands to a sketch."""
    rng = np.random.default_rng(0)
    for _ in range(count):
        yield rng.integers(-4, 5, size=2, dtype=np.int64)


@pytest.mark.parametrize("python_hash_seed", ["1", "12345"])
def test_rshash_is_reproducible_across_processes(python_hash_seed):
    """The same seed must give the same scores in any process.

    Guarded in-process this defect is invisible: the hash salt is fixed for the
    lifetime of a process, so the trace is self-consistent either way. Only a
    second process with a different ``PYTHONHASHSEED`` can expose it.
    """
    reference = _score_digest_in_subprocess("0")

    assert _score_digest_in_subprocess(python_hash_seed) == reference, (
        "RSHash: seed 42 produced different scores under "
        f"PYTHONHASHSEED={python_hash_seed} than under PYTHONHASHSEED=0. "
        "Slot assignment depends on the interpreter's per-process hash salt, "
        "so results are not reproducible across processes."
    )


def test_table_keys_keep_the_tables_independent():
    """The sketch's ``w`` keys must each steer their own table.

    ``hash()`` took the key as part of the tuple it hashed, so dropping the key
    would collapse every table onto the same slots and quietly turn the sketch
    into ``w`` copies of one table. ``zlib.crc32`` takes the key as its initial
    register, which preserves that, and this test is what keeps it that way:
    a seed-blind ``zlib.crc32(payload, 0)`` passes both other tests in this
    file, so only this assertion can see the difference.
    """
    sketch = RSHashCountMinSketch(p=10_000, w=4, rng=np.random.default_rng(42))

    vectors = list(_discretized_vectors())
    tuples = [tuple(sketch._indices(vector)) for vector in vectors]

    # No two tables of the same sketch may land on one slot.
    assert all(len(set(slots)) > 1 for slots in tuples), (
        "RSHash: a table key mapped a vector to the same slot as another "
        "table's key, the keys are not steering their own tables"
    )

    # Distinct payloads must mostly stay distinct, otherwise the sketch is
    # over-counting by more than its estimate assumes. ``_discretize`` maps to
    # a small integer lattice, so compare against the distinct payloads rather
    # than the number of vectors drawn.
    distinct_payloads = {vector.tobytes() for vector in vectors}
    collisions = len(distinct_payloads) - len(set(tuples))
    assert collisions <= 0.01 * len(distinct_payloads), (
        f"RSHash: {collisions} of {len(distinct_payloads)} distinct payloads "
        "collapsed onto a shared slot tuple, slot assignment is far more "
        "collision-prone than the sketch assumes"
    )


def test_rshash_still_consumes_its_seed():
    """Cross-process stability must not be bought by dropping the seed.

    A detector that ignores its seed is trivially reproducible, so assert the
    opposite property here: a different seed must still change the scores.
    """
    scores = {}
    for seed in (SEED, 1):
        stream = TinyBlobs()
        learner = RSHash(stream.get_schema(), seed=seed, m=COMPONENTS, s=WINDOW)
        trace = []
        for index, instance in enumerate(stream):
            if index >= INSTANCES:
                break
            trace.append(learner.score_instance(instance))
            learner.train(instance)
        scores[seed] = trace

    assert scores[SEED] != scores[1], (
        "RSHash: seed 42 and seed 1 produced identical scores, the seed is no "
        "longer reaching slot assignment"
    )
