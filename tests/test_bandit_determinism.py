"""Seed-reproducibility contract for BanditClassifier's exploration.

The epsilon-greedy policy explored via the *global* ``random`` module, so
two BanditClassifiers with the same ``random_seed`` selected different
models to train whenever the ambient global RNG state differed (e.g.
another library drawing from ``random`` earlier in the process), silently
breaking the documented reproducibility contract. The policy now draws
from an injectable ``random.Random`` instance seeded by the classifier.
"""

import random

from capymoa.automl import BanditClassifier, EpsilonGreedy
from capymoa.classifier import HoeffdingTree, NoChange
from capymoa.datasets import ElectricityTiny

N_STEPS = 250


class CountingRandom(random.Random):
    """A caller's own generator, wrapped to count the draws it serves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draws = 0

    def random(self, *args, **kwargs):
        self.draws += 1
        return super().random(*args, **kwargs)


def _schema():
    stream = ElectricityTiny()
    stream.restart()
    return stream.get_schema()


def _build(policy: EpsilonGreedy, random_seed: int):
    return BanditClassifier(
        schema=_schema(),
        random_seed=random_seed,
        base_classifiers=[HoeffdingTree, NoChange],
        policy=policy,
    )


def _train(policy: EpsilonGreedy, random_seed: int) -> tuple:
    """Train a BanditClassifier on ``policy``; return comparable policy state.

    ``policy`` is used as given, so the same policy object can be handed to
    several classifiers.
    """
    stream = ElectricityTiny()
    stream.restart()
    learner = _build(policy=policy, random_seed=random_seed)
    for i, instance in enumerate(stream):
        if i >= N_STEPS:
            break
        learner.train(instance)
    return (
        tuple(learner.policy.arm_counts),
        tuple(round(r, 9) for r in learner.policy.arm_rewards),
    )


def _arm_history(ambient_seed: int, random_seed: int) -> tuple:
    """Train a fresh BanditClassifier; return comparable policy state.

    The ambient global RNG is reseeded before the run to simulate unrelated
    code drawing from it between runs.
    """
    random.seed(ambient_seed)
    return _train(
        policy=EpsilonGreedy(epsilon=0.1, burn_in=50), random_seed=random_seed
    )


def test_same_seed_same_arm_history_regardless_of_ambient_rng():
    first = _arm_history(ambient_seed=111, random_seed=7)
    second = _arm_history(ambient_seed=999, random_seed=7)

    assert first == second, (
        f"same random_seed produced different bandit histories "
        f"(counts {first[0]} vs {second[0]}) when the global RNG state "
        "differed - exploration is drawing from the global RNG instead "
        "of the classifier's seed"
    )


def test_different_seeds_diverge():
    """A stochastic policy must actually consume its seed."""
    first = _arm_history(ambient_seed=0, random_seed=1)
    second = _arm_history(ambient_seed=0, random_seed=42)

    assert first != second, (
        "arm histories identical under seeds 1 and 42 - the policy "
        "never explores, so the seed is unused"
    )


def test_reused_policy_follows_the_new_classifiers_seed():
    """A policy passed to a second classifier must be seeded again.

    ``EpsilonGreedy.initialize()`` resets the arm statistics, so a classifier
    that reuses a policy starts from scratch. The generator has to be reset
    with it: kept as it was, the policy would explore from draws the earlier
    run made and the second classifier would no longer follow its own
    ``random_seed``.
    """
    # What each seed yields with a policy of its own.
    baseline_seed_1 = _arm_history(ambient_seed=0, random_seed=1)
    baseline_seed_42 = _arm_history(ambient_seed=0, random_seed=42)

    shared = EpsilonGreedy(epsilon=0.1, burn_in=50)
    first = _train(policy=shared, random_seed=1)
    second = _train(policy=shared, random_seed=1)
    third = _train(policy=shared, random_seed=42)

    assert first == baseline_seed_1, (
        f"a fresh policy did not follow its own random_seed "
        f"(counts {first[0]} vs {baseline_seed_1[0]})"
    )
    assert second == baseline_seed_1, (
        f"the second classifier on a reused policy did not reproduce "
        f"random_seed=1 (counts {second[0]} vs {baseline_seed_1[0]}) - the "
        "policy kept the generator left over from the first run while "
        "initialize() reset its statistics"
    )
    assert third == baseline_seed_42, (
        f"the third classifier on a reused policy did not follow its own "
        f"random_seed=42 (counts {third[0]} vs {baseline_seed_42[0]}) - the "
        "re-seed did not take effect"
    )


def test_caller_supplied_generator_is_never_overwritten():
    """A generator the caller passed in stays the caller's, seed or not.

    ``CountingRandom`` subclasses ``random.Random``, which is an ordinary
    thing for a caller to do to log or count exploration draws, and is the
    case most at risk from a classifier that replaces generators by type.
    The draw count also shows the generator is left in use, not merely
    still referenced.
    """
    caller_rng = CountingRandom(123)
    policy = EpsilonGreedy(epsilon=0.1, burn_in=50, rng=caller_rng)

    assert _build(policy=policy, random_seed=1).policy.rng is caller_rng, (
        "BanditClassifier replaced the generator the caller supplied"
    )

    # A second classifier on the same policy must leave it alone too.
    assert _build(policy=policy, random_seed=42).policy.rng is caller_rng, (
        "BanditClassifier replaced the caller's generator when the policy "
        "was reused - re-seeding must not reach a generator it does not own"
    )
    assert caller_rng.draws == 0, (
        f"constructing a classifier drew from the caller's generator "
        f"({caller_rng.draws} draws)"
    )

    _train(policy=policy, random_seed=1)

    assert caller_rng.draws > 0, (
        "the policy stopped drawing from the caller's generator - it was "
        "replaced by one seeded from the classifier's random_seed"
    )


def test_generator_replaced_after_a_run_is_treated_as_the_callers():
    """A generator the caller installs later is the caller's from then on.

    Ownership is settled by comparing against the generator the policy
    recorded, so swapping ``rng`` after a run hands it to the caller. A bare
    "a classifier seeded this policy" flag could not tell this case from a
    reused policy, and the next classifier would overwrite a generator it
    never installed.
    """
    caller_rng = random.Random(999)
    policy = EpsilonGreedy(epsilon=0.1, burn_in=50)
    # A first classifier installs a generator of its own.
    _build(policy=policy, random_seed=1)
    assert policy.rng is not caller_rng

    policy.rng = caller_rng

    assert _build(policy=policy, random_seed=42).policy.rng is caller_rng, (
        "BanditClassifier overwrote a generator the caller installed after an "
        "earlier run - the marker it reads no longer describes this "
        "generator"
    )
