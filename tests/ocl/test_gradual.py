from collections.abc import Sequence

import pytest

pytestmark = pytest.markskip("torch")

import torch  # noqa: E402
from torch import BoolTensor, IntTensor  # noqa: E402
from torch.utils.data import Dataset, TensorDataset  # noqa: E402

from capymoa.ocl.datasets.gradual import (  # noqa: E402
    SigmoidFn,
    TransitionFn,
    _gradual_task_idx,
    _idx_interleave,
    gradual_task_transitions,
)


def _make_synthetic_tasks(lengths: Sequence[int]) -> list[Dataset]:
    """Return list of datasets containing incrementing numbers."""
    tasks: list[Dataset] = []
    start = 0
    for length in lengths:
        x = torch.arange(start, start + length, dtype=torch.float32).unsqueeze(1)
        y = torch.arange(start, start + length, dtype=torch.long)
        tasks.append(TensorDataset(x, y))
        start += length
    return tasks


class FixedMaskTransitionFn(TransitionFn):
    """Dummy transition function"""

    def __init__(self, masks: Sequence[BoolTensor]):
        self._masks = masks

    def __call__(self, task: int, left_len: int, right_len: int) -> BoolTensor:
        mask = self._masks[task]
        assert len(mask) == left_len + right_len
        return mask


def test_idx_interleave_no_duplicate_indices() -> None:
    """Assert interleaved tasks contain no duplicates."""
    tasks = _make_synthetic_tasks([4, 4])
    left_idx = torch.arange(0, len(tasks[0]), dtype=torch.int64)
    right_idx = torch.arange(
        len(tasks[0]), len(tasks[0]) + len(tasks[1]), dtype=torch.int64
    )
    mask = torch.tensor(
        [False, True, False, True, False, True, True, False], dtype=torch.bool
    )

    new_left, new_right = _idx_interleave(left_idx, right_idx, mask)
    merged = torch.cat([new_left, new_right]).tolist()

    assert len(merged) == len(set(merged))
    assert set(merged).issubset(set(left_idx.tolist() + right_idx.tolist()))


def test_gradual_task_idx_preserves_within_task_order() -> None:
    """Assert the order of tasks is preserved"""
    tasks = _make_synthetic_tasks([4, 4, 3])
    boundaries = [0]
    for task in tasks:
        boundaries.append(boundaries[-1] + len(task))

    task_idx: list[IntTensor] = [
        torch.arange(start, end, dtype=torch.int64)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    transition_fn = FixedMaskTransitionFn(
        masks=[
            torch.tensor([False, False, True, True, False, True, True, False]),
            torch.tensor([False, True, False, True, False, True, True]),
        ]
    )

    transitioned = _gradual_task_idx(task_idx, transition_fn)
    flattened = [int(v) for idx in transitioned for v in idx.tolist()]

    assert len(flattened) == len(set(flattened))
    assert [v for v in flattened if 0 <= v <= 3] == sorted(
        [v for v in flattened if 0 <= v <= 3]
    )
    assert [v for v in flattened if 4 <= v <= 7] == sorted(
        [v for v in flattened if 4 <= v <= 7]
    )
    assert [v for v in flattened if 8 <= v <= 10] == sorted(
        [v for v in flattened if 8 <= v <= 10]
    )


def test_gradual_task_transitions_seeded_deterministic_and_ordered() -> None:
    """Assert determinism and order preservation with sigmoid fn transition func."""
    lengths = [6, 5, 4]
    tasks = _make_synthetic_tasks(lengths)

    transitioned_a = gradual_task_transitions(tasks, SigmoidFn(width=0.5, seed=0))
    transitioned_b = gradual_task_transitions(tasks, SigmoidFn(width=0.5, seed=0))

    indices_a = [list(task.indices) for task in transitioned_a]
    indices_b = [list(task.indices) for task in transitioned_b]

    assert indices_a == indices_b

    flattened = [idx for task_indices in indices_a for idx in task_indices]
    assert len(flattened) == len(set(flattened))

    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        original_task_indices = [v for v in flattened if start <= v < end]
        assert original_task_indices == sorted(original_task_indices)
