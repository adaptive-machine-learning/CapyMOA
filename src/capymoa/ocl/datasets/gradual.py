"""Simulate gradual transition between tasks for online continual learning.

>>> from capymoa.ocl.datasets import TinySplitMNIST
>>> from capymoa.ocl.datasets.gradual import gradual_task_transitions
>>> scenario = TinySplitMNIST()
>>> tasks = gradual_sigmoid_transitions(scenario.train_tasks)

"""

from torch.utils.data import Dataset, ConcatDataset, Subset
from typing import Tuple, Sequence, cast
from torch import BoolTensor, IntTensor
import torch
from torch.nn.functional import sigmoid
from abc import ABC, abstractmethod


class TransitionFn(ABC):
    """Callable transition strategy.

    Implementations must return a boolean mask of length ``left_len + right_len``
    where ``False`` prefers sampling from the left stream and ``True`` prefers
    sampling from the right stream.
    """

    @abstractmethod
    def __call__(self, task: int, left_len: int, right_len: int) -> BoolTensor:
        """Generate a mask for transitioning between two tasks.

        :param task: The index of the current task transition (0 for the first
            transition, 1 for the second, etc.)
        :param left_len: The number of samples in the left task.
        :param right_len: The number of samples in the right task.
        :return: A boolean tensor of length ``left_len + right_len`` where ``False``
            indicates a preference for sampling from the left task and ``True``
            indicates a preference for sampling from the right task.
        """


class SigmoidFn(TransitionFn):
    r"""Transition function that simulates a gradual transition between tasks using a
    sigmoid curve.

    We may define :math:`M` using sigmoid-based transition probabilities:

    .. math::

        p(M_t = 1) = \sigma\left( \frac{4}{\min(n_L, n_R) \cdot w} (t - n_L)\right)

    where :math:`\sigma` is the standard sigmoid function, :math:`w` is a scalar
    that controls the width of the transition, and the sigmoid is centred at the
    boundary between the left and right task.
    """

    def __init__(self, width: float, seed: int | None = None) -> None:
        super().__init__()
        if not (0.0 <= width <= 1.0):
            raise ValueError("width must be between 0.0 and 1.0")
        self.width = width
        self.rng = torch.Generator().manual_seed(seed) if seed is not None else None

    def __call__(self, task: int, left_len: int, right_len: int) -> BoolTensor:
        if left_len == 0:
            return torch.ones(right_len, dtype=torch.bool)  # type: ignore

        if right_len == 0:
            return torch.zeros(left_len, dtype=torch.bool)  # type: ignore

        if self.width == 0.0:
            return torch.cat(
                [
                    torch.zeros(left_len, dtype=torch.bool),
                    torch.ones(right_len, dtype=torch.bool),
                ]
            )  # type: ignore

        total_len = left_len + right_len
        t = torch.arange(total_len, dtype=torch.float32)
        transition_point = left_len
        transition_width = self.width * min(left_len, right_len)
        transition_fn = sigmoid((4 / transition_width) * (t - transition_point))
        return transition_fn > torch.rand_like(transition_fn, generator=self.rng)  # type: ignore


def _idx_interleave(
    left_idx: IntTensor,
    right_idx: IntTensor,
    mask: BoolTensor,
) -> Tuple[IntTensor, IntTensor]:
    """Transition from the left task to the right task by interleaving the indices based
    on the provided mask.

    Returns two tensors: one for the newly gradual left task and one for the newly
    gradual right task. The transition point is determined by the relative size of
    the left and right tasks.

    >>> left_idx = torch.tensor([0, 1, 2, 3])
    >>> right_idx = torch.tensor([4, 5, 6, 7])
    >>> mask = torch.tensor([False, False, True, True, False, False, True, True])
    >>> _idx_interleave(left_idx, right_idx, mask)
    (tensor([0, 1, 4, 5]), tensor([2, 3, 6, 7]))
    """
    if len(mask) != len(left_idx) + len(right_idx):
        raise ValueError("mask length must equal len(left_idx) + len(right_idx)")

    indices = torch.arange(len(mask))
    left_i, right_i = 0, 0
    n_left, n_right = len(left_idx), len(right_idx)

    for i in range(len(mask)):
        if not mask[i] and left_i < n_left:
            indices[i] = left_idx[left_i]
            left_i += 1
        elif right_i < n_right:
            indices[i] = right_idx[right_i]
            right_i += 1
        else:
            break

    # Calculate the transition point by preserving the relative size of left and right indices
    stream_len = left_i + right_i
    transition_point = int((n_left / (n_left + n_right)) * stream_len)

    return (
        cast(IntTensor, indices[:transition_point]),
        cast(IntTensor, indices[transition_point:stream_len]),
    )


def _gradual_task_idx(
    task_idx: list[IntTensor], transition_fn: TransitionFn
) -> list[IntTensor]:
    n_tasks = len(task_idx)
    # copy task_idx to avoid modifying the original list
    task_idx = [cast(IntTensor, idx.clone()) for idx in task_idx]

    for i in range(n_tasks - 1):
        left_idx, right_idx = task_idx[i], task_idx[i + 1]
        mask = transition_fn(i, len(left_idx), len(right_idx))
        gradual_left_idx, gradual_right_idx = _idx_interleave(left_idx, right_idx, mask)
        task_idx[i] = gradual_left_idx
        task_idx[i + 1] = gradual_right_idx
    return task_idx


def gradual_sigmoid_transitions(
    tasks: Sequence[Dataset], width: float = 0.5, seed: int | None = None
) -> Sequence[Dataset]:
    r"""Apply sigmoid-based gradual task transitions to a sequence of tasks.

    This is a convenience wrapper around :func:`gradual_task_transitions` that uses
    :class:`SigmoidFn` as the transition function.
    """
    return gradual_task_transitions(tasks, SigmoidFn(width=width, seed=seed))


def gradual_task_transitions(
    tasks: Sequence[Dataset], transition_fn: TransitionFn
) -> Sequence[Dataset]:
    r"""Apply gradual task transitions to a sequence of tasks using the provided transition
    function.

    We define a combined stream with a gradual boundary :math:`S` of length :math:`n`
    given two source streams :math:`L` and :math:`R` of lengths :math:`n_L` and
    :math:`n_R`.
    Our gradual boundary preserves the relative order of elements within their
    original streams and does not duplicate any data, but it may discard some
    data from the end of :math:`L`.
    A binary mask :math:`M \in \{0,1\}^N`, where :math:`N = n_L + n_R`, controls
    the simulation of a gradual transition between tasks and may come from a
    random process.

    We initialise the indices :math:`l_1 = r_1 = 1`.
    Then, for each time-step :math:`t \in \{1, ..., N\}`, the interleaved stream
    element :math:`S_t` is determined iteratively by:

    .. math::

        (S_t, l_{t+1}, r_{t+1}) =
        \begin{cases}
        (L_{l_t}, l_t + 1, r_t) & \text{if}\ M_t = 0\ \text{and}\ l_t \leq n_L \\
        (R_{r_t}, l_t, r_t + 1) & \text{if}\ (M_t = 1\ \text{or}\ l_t > n_L)\ \text{and}\ r_t \leq n_R \\
        \text{(end of stream)} & \text{otherwise}
        \end{cases}

    The binary mask vector :math:`M` controls which source is used, unless the
    left stream :math:`L` has been exhausted, in which case we switch to the
    right stream :math:`R` until it is also exhausted.
    In practice, :math:`M` should contain mostly zeros before the boundary and
    mostly ones after the boundary to model a smooth task shift.

    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> stream = TinySplitMNIST(shuffle_tasks=False)
    >>> int(stream.train_tasks[0][-1][1])
    0
    >>> tasks = gradual_task_transitions(stream.train_tasks, SigmoidFn(width=0.5, seed=0))
    >>> int(tasks[0][-1][1]) # Task 2 now transitions gradually into Task 1
    3

    """
    stream = ConcatDataset(tasks)
    cumulative_sizes = stream.cumulative_sizes
    task_idx: list[IntTensor] = [
        cast(IntTensor, torch.arange(start, end))
        for start, end in zip([0] + cumulative_sizes, cumulative_sizes)
    ]
    return [
        Subset(stream, idx.tolist())
        for idx in _gradual_task_idx(task_idx, transition_fn)
    ]  # type: ignore
