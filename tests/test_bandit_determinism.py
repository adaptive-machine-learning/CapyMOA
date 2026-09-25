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


def _arm_history(ambient_seed: int, random_seed: int) -> tuple:
    """Train a fresh BanditClassifier; return comparable policy state.

    The ambient global RNG is reseeded before the run to simulate unrelated
    code drawing from it between runs.
    """
    stream = ElectricityTiny()
    stream.restart()
    random.seed(ambient_seed)
    learner = BanditClassifier(
        schema=stream.get_schema(),
        random_seed=random_seed,
        base_classifiers=[HoeffdingTree, NoChange],
        policy=EpsilonGreedy(epsilon=0.1, burn_in=50),
    )
    for i, instance in enumerate(stream):
        if i >= N_STEPS:
            break
        learner.train(instance)
    return (
        tuple(learner.policy.arm_counts),
        tuple(round(r, 9) for r in learner.policy.arm_rewards),
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
