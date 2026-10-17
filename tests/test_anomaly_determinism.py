"""Seed handling for the anomaly detectors.

This file pins the part of the seeding contract that holds whatever a
``None`` seed is decided to mean, so it stays valid whichever way that
question is settled:

* an explicit integer seed makes a detector reproducible, in-process and
  in a fresh process;
* the generator a detector draws from is derived from the instance
  attribute ``self.random_seed``, not from the raw constructor argument;
* no detector rewrites numpy's process-global generator.

It deliberately makes **no claim** about ``random_seed=None``. Under the
policy discussed in #451, ``None`` draws from system entropy, so a test
here asserts only that ``None`` is stored and used unchanged, and that the
detector still runs. Asserting that ``None`` reproduces would encode the
opposite contract, which is exactly what this file does not do.

Out of scope here:

* ``StreamRHF`` still rewrites numpy's process-global seed inside
  ``choose_split_attribute`` and still passes its raw ``random_seed``
  argument into the forest. Both are #441 and #442, so it is excluded from
  the two tests that would fail on it.
* ``RSHash`` is excluded from the fresh-process test only. Its
  ``RSHashCountMinSketch._indices`` addresses cells with the builtin
  ``hash()`` of a tuple containing a ``bytes`` payload, and Python
  randomises that per process, so RSHash does not reproduce across
  interpreters whatever seed it is given. That is a real defect but it is
  a hashing bug rather than a seed-derivation one, and changing the hash
  function changes the sketch's collision rate, so it is reported rather
  than fixed here. Run with ``PYTHONHASHSEED`` fixed to see the rest of the
  suite agree.
* ``MOARegressor.__init__`` seeds the learner from its own raw
  ``random_seed`` argument instead of ``self.random_seed``, the same shape
  of defect fixed here for ``MOAAnomalyDetector``. It is a regressor, not
  an anomaly detector, so it is not touched by this change.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from capymoa.anomaly import (
    AdaptiveIsolationForest,
    HalfSpaceTrees,
    IForestASD,
    Loda,
    OnlineIsolationForest,
    RobustRandomCutForest,
    RSHash,
    StreamingIsolationForest,
    StreamRHF,
)
from capymoa.anomaly.datasets import TinyBlobs
from capymoa.base import AnomalyDetector
from capymoa.base._base import MOAAnomalyDetector

# Sizing is kept small so the suite stays CI-fast. The seed handling under
# test is not affected by these values.
WARMUP = 64
N_SCORES = 60

# Stands in for "the base class resolved the seed to something else". Any
# value works; it only has to be an int every detector accepts, and to be
# different from the seed passed to the constructor.
RESOLVED_SEED = 90210


class _Case:
    """One detector, built either with a seed or with ``None``."""

    def __init__(self, name, build, *, scope=()):
        self.name = name
        self.build = build
        # Names of the tests this case is excluded from, with the reason.
        self.scope = frozenset(scope)

    def __repr__(self):
        return self.name


def _schema():
    return TinyBlobs().get_schema()


def _trace(make, schema, n_scores=N_SCORES, warmup=WARMUP):
    """Score trace over one pass of TinyBlobs, past each window build."""
    stream = TinyBlobs()
    detector = make(schema)
    scores = []
    for i, instance in enumerate(stream):
        if i >= warmup + n_scores:
            break
        score = detector.score_instance(instance)
        if i >= warmup:
            scores.append(round(float(score), 10))
        detector.train(instance)
    return scores


def _cases():
    """(detector name, seed kwarg) pairs, sized small for CI."""

    def hstrees(schema, seed):
        return HalfSpaceTrees(schema, window_size=64, random_seed=seed)

    def oif(schema, seed):
        return OnlineIsolationForest(
            schema, window_size=64, num_trees=8, random_seed=seed
        )

    def sif(schema, seed):
        return StreamingIsolationForest(schema, window_size=64, n_trees=10, seed=seed)

    def aif(schema, seed):
        return AdaptiveIsolationForest(schema, window_size=64, n_trees=10, seed=seed)

    def rrcf(schema, seed):
        return RobustRandomCutForest(
            schema, tree_size=64, n_trees=10, random_state=seed
        )

    def ifasd(schema, seed):
        return IForestASD(
            schema, window_size=64, n_trees=4, sample_size=16, random_state=seed
        )

    def loda(schema, seed):
        return Loda(schema, n_projections=5, window_size=64, random_state=seed)

    def rshash(schema, seed):
        return RSHash(schema, m=5, s=64, w=2, p=1000, seed=seed)

    return [
        _Case("HalfSpaceTrees", hstrees),
        _Case("OnlineIsolationForest", oif),
        _Case("StreamingIsolationForest", sif),
        _Case("AdaptiveIsolationForest", aif),
        _Case("RobustRandomCutForest", rrcf),
        _Case("IForestASD", ifasd),
        _Case("Loda", loda),
        # Both exclusions trace to #441 / #442, not to this file.
        _Case(
            "StreamRHF",
            lambda schema, seed: StreamRHF(
                schema, num_trees=5, max_height=3, random_seed=seed
            ),
            scope={"global_rng", "from_attribute"},
        ),
        # Excluded from the fresh-process test only, see the module docstring.
        _Case("RSHash", rshash, scope={"fresh_process"}),
    ]


ALL_CASES = _cases()


def _ids(cases):
    return [case.name for case in cases]


# --------------------------------------------------------------------------
# 1. An explicit seed reproduces. Holds under any policy for None.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids(ALL_CASES))
@pytest.mark.parametrize("seed", [1, 42], ids=["seed1", "seed42"])
def test_explicit_seed_is_reproducible(case, seed):
    """Same explicit seed, two constructions, same scores.

    The seed given to the constructor must be the one stored on the
    instance, otherwise "seed 42" does not describe what the detector did.
    """
    schema = _schema()
    detector = case.build(schema, seed)
    assert detector.random_seed == seed, (
        f"{case.name} stored random_seed={detector.random_seed!r} for a "
        f"detector constructed with seed={seed!r}. The attribute is the "
        "seed the caller asked for; a component must not substitute another."
    )

    first = _trace(lambda s: case.build(s, seed), schema)
    second = _trace(lambda s: case.build(s, seed), schema)

    assert first == second, (
        f"{case.name} is not reproducible under an explicit seed={seed}. "
        "First difference at offset "
        f"{next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)}."
    )


# --------------------------------------------------------------------------
# 2. The generator comes from the attribute, not the constructor argument.
# --------------------------------------------------------------------------


_ATTR_CASES = [c for c in ALL_CASES if "from_attribute" not in c.scope]


@pytest.mark.parametrize("case", _ATTR_CASES, ids=_ids(_ATTR_CASES))
def test_generator_is_derived_from_the_seed_attribute(case, monkeypatch):
    """A resolved seed must reach the generator.

    ``AnomalyDetector.__init__`` is wrapped so the seed it resolves is not
    the one passed to the constructor. This cuts the argument channel: a
    detector that reads ``self.random_seed`` follows the resolved value, and
    one that reads its own parameter keeps the original and drifts.

    The reference trace is built by passing ``RESOLVED_SEED`` straight to
    the constructor, so the two must agree.
    """
    original_init = AnomalyDetector.__init__

    def resolving_init(self, schema, random_seed=1):
        original_init(self, schema=schema, random_seed=random_seed)
        self.random_seed = RESOLVED_SEED

    monkeypatch.setattr(AnomalyDetector, "__init__", resolving_init)

    schema = _schema()
    resolved = _trace(lambda s: case.build(s, 42), schema)
    assert case.build(schema, 42).random_seed == RESOLVED_SEED, (
        "the test setup did not take effect; AnomalyDetector.__init__ was "
        "not the method this detector resolved its seed through"
    )

    monkeypatch.undo()
    reference = _trace(lambda s: case.build(s, RESOLVED_SEED), schema)

    assert resolved == reference, (
        f"{case.name} ignored the seed on the instance and built its "
        "generator from the raw constructor argument, so it is not "
        "reproducible from the seed it reports. The difference appears from "
        f"offset {next(i for i, (a, b) in enumerate(zip(resolved, reference)) if a != b)}."
    )


# --------------------------------------------------------------------------
# 3. A reused path reproduces given a seed. Fresh interpreter, same trace.
# --------------------------------------------------------------------------

_XPROC_CASES = [c for c in ALL_CASES if "fresh_process" not in c.scope]

_XPROC_SOURCE = textwrap.dedent(
    """
    import sys
    from capymoa.anomaly import (
        AdaptiveIsolationForest, HalfSpaceTrees, IForestASD, Loda,
        OnlineIsolationForest, RobustRandomCutForest, RSHash,
        StreamingIsolationForest, StreamRHF,
    )
    from capymoa.anomaly.datasets import TinyBlobs

    BUILDERS = {
        "HalfSpaceTrees": lambda s, k: HalfSpaceTrees(s, window_size=64, random_seed=k),
        "OnlineIsolationForest": lambda s, k: OnlineIsolationForest(s, window_size=64, num_trees=8, random_seed=k),
        "StreamingIsolationForest": lambda s, k: StreamingIsolationForest(s, window_size=64, n_trees=10, seed=k),
        "AdaptiveIsolationForest": lambda s, k: AdaptiveIsolationForest(s, window_size=64, n_trees=10, seed=k),
        "RobustRandomCutForest": lambda s, k: RobustRandomCutForest(s, tree_size=64, n_trees=10, random_state=k),
        "IForestASD": lambda s, k: IForestASD(s, window_size=64, n_trees=4, sample_size=16, random_state=k),
        "Loda": lambda s, k: Loda(s, n_projections=5, window_size=64, random_state=k),
        "RSHash": lambda s, k: RSHash(s, m=5, s=64, w=2, p=1000, seed=k),
        "StreamRHF": lambda s, k: StreamRHF(s, num_trees=5, max_height=3, random_seed=k),
    }

    name, seed, warmup, n_scores = (int(a) if i else a for i, a in enumerate(sys.argv[1:]))
    stream = TinyBlobs()
    detector = BUILDERS[name](stream.get_schema(), seed)
    scores = []
    for i, instance in enumerate(stream):
        if i >= warmup + n_scores:
            break
        score = detector.score_instance(instance)
        if i >= warmup:
            scores.append(round(float(score), 10))
        detector.train(instance)
    print(repr(scores))
    """
)


@pytest.mark.parametrize("case", _XPROC_CASES, ids=_ids(_XPROC_CASES))
@pytest.mark.parametrize("seed", [1, 42], ids=["seed1", "seed42"])
def test_seeded_detector_reproduces_in_a_fresh_process(case, seed, tmp_path):
    """A seeded detector reproduces in a new interpreter.

    This is the reuse case: a detector saved to disk, reloaded, or run
    from the command line in another job. In-process equality does not
    cover it, because a module-level generator or a cached global state can
    make one process reproducible and the next not.
    """
    script = tmp_path / "trace.py"
    script.write_text(_XPROC_SOURCE, encoding="utf-8")

    def run():
        done = subprocess.run(
            [
                sys.executable,
                str(script),
                case.name,
                str(seed),
                str(WARMUP),
                str(N_SCORES),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    first, second = run(), run()

    assert first == second and first, (
        f"{case.name} did not reproduce with seed={seed} across two fresh "
        "processes, so a reused or reloaded detector cannot be relied on. "
        "Run each output through repr() differences to compare."
    )


# --------------------------------------------------------------------------
# 4. No component rewrites numpy's process-global generator.
# --------------------------------------------------------------------------


def _global_state():
    name, keys, position, has_gauss, cached = np.random.get_state()
    return (name, keys.tobytes(), position, int(has_gauss), int(cached))


_GLOBAL_CASES = [c for c in ALL_CASES if "global_rng" not in c.scope]


@pytest.mark.parametrize("case", _GLOBAL_CASES, ids=_ids(_GLOBAL_CASES))
@pytest.mark.parametrize("seed", [1, 42], ids=["seed1", "seed42"])
def test_does_not_rewrite_the_global_numpy_rng(case, seed):
    """Constructing and training must not move the process-global stream.

    ``np.random.seed()`` inside a library rewrites the stream for
    everything else in the process, including components the caller seeded
    deliberately, so their reproducibility is lost as a side effect.
    """
    schema = _schema()
    np.random.seed(1234)
    before = _global_state()

    _trace(lambda s: case.build(s, seed), schema)

    after = _global_state()
    assert before == after, (
        f"{case.name} rewrote numpy's process-global generator while "
        "building or training. It must use its own Generator instead."
    )


# --------------------------------------------------------------------------
# 5. None is stored and used unchanged. No claim about reproducibility.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids(ALL_CASES))
def test_none_seed_is_stored_unchanged(case):
    """An explicit ``None`` must reach the instance as ``None``.

    This is the only thing asserted about ``None``. Whether the result
    reproduces is not asserted, because that is the open question: with
    ``None`` meaning "draw entropy" (see #451) it must not, and a component
    that quietly turned ``None`` into a fixed seed would be the other half
    of the same mistake.
    """
    schema = _schema()
    detector = case.build(schema, None)

    assert detector.random_seed is None, (
        f"{case.name} turned an explicit random_seed=None into "
        f"{detector.random_seed!r}. None has to survive to the attribute, "
        "so a caller can see that nothing was seeded."
    )

    scores = _trace(lambda s: case.build(s, None), schema)
    assert scores, f"{case.name} produced no scores with random_seed=None"
    assert all(np.isfinite(scores)), (
        f"{case.name} produced a non-finite score with random_seed=None: "
        f"{[s for s in scores if not np.isfinite(s)][:5]}"
    )


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids(ALL_CASES))
def test_none_seed_differs_from_a_seeded_run(case):
    """``None`` and an explicit seed must not be the same thing.

    Two runs of ``None`` may agree or may not, so they are not compared.
    What is checked is that collapsing ``None`` onto a fixed seed is
    visible: under a fixed-seed collapse every ``None`` run equals the
    run made with that seed.
    """
    schema = _schema()
    unseeded = _trace(lambda s: case.build(s, None), schema)
    for seed in (1, 42):
        seeded = _trace(lambda s, k=seed: case.build(s, k), schema)
        if unseeded != seeded:
            return
    pytest.fail(
        f"{case.name} produces the same trace for random_seed=None as for "
        "seed=1 and seed=42, so None is being collapsed onto a fixed seed "
        "rather than being left unseeded"
    )


# --------------------------------------------------------------------------
# 6. The MOA wrappers seed the learner from the attribute.
# --------------------------------------------------------------------------


class _RecordingLearner:
    """Stands in for a MOA learner and records the seed it was given."""

    def __init__(self):
        self.seeds = []

    def setRandomSeed(self, seed):
        self.seeds.append(seed)

    def setModelContext(self, header):
        pass

    def getOptions(self):
        raise AssertionError("CLI is None, so options are never touched")

    def prepareForUse(self):
        pass

    def resetLearning(self):
        pass


def _resolving_base(monkeypatch, resolved):
    """Make ``AnomalyDetector`` resolve the seed to something else."""
    original_init = AnomalyDetector.__init__

    def resolving_init(self, schema, random_seed=1):
        original_init(self, schema=schema, random_seed=random_seed)
        self.random_seed = resolved

    monkeypatch.setattr(AnomalyDetector, "__init__", resolving_init)


@pytest.mark.parametrize("seed", [1, 42], ids=["seed1", "seed42"])
def test_moa_detector_seeds_the_learner_from_the_attribute(monkeypatch, seed):
    """``MOAAnomalyDetector`` must pass ``self.random_seed`` on.

    Guarding on the constructor argument reads the un-resolved seed, so a
    base class that resolves the seed has its decision ignored.
    """
    _resolving_base(monkeypatch, RESOLVED_SEED)

    schema = _schema()
    learner = _RecordingLearner()
    detector = MOAAnomalyDetector(
        schema=schema, CLI=None, random_seed=seed, moa_learner=learner
    )

    assert detector.random_seed == RESOLVED_SEED, "test setup did not take effect"
    assert learner.seeds == [RESOLVED_SEED], (
        "MOAAnomalyDetector seeded the learner with "
        f"{learner.seeds!r} but the resolved seed on the instance was "
        f"{RESOLVED_SEED}. It must read self.random_seed, not the raw "
        "constructor argument."
    )


def test_moa_detector_skips_seeding_when_the_seed_is_none():
    """``None`` must not be forwarded to the MOA learner."""
    schema = _schema()
    learner = _RecordingLearner()
    MOAAnomalyDetector(schema=schema, CLI=None, random_seed=None, moa_learner=learner)

    assert learner.seeds == [], (
        f"MOAAnomalyDetector called setRandomSeed with {learner.seeds!r} "
        "for an unseeded detector. None has to leave the learner's own "
        "seeding alone."
    )
