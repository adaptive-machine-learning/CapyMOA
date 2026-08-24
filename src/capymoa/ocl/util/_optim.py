from collections import defaultdict

from torch.optim import Optimizer


def reset_optimizer_state(optimizer: Optimizer) -> None:
    """Reset an optimizer's per-parameter state (e.g. momentum buffers) in-place."""
    optimizer.__setstate__({"state": defaultdict(dict)})
