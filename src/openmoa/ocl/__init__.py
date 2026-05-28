# SPDX-License-Identifier: BSD-3-Clause
"""Online Continual Learning (OCL) module.

OCL is a setting where learners train on a sequence of tasks. A task is a
specific concept or data distribution. After training the learner on each task,
we evaluate the learner on all tasks.

Continual learning is an important problem to deep learning because these models
suffer from catastrophic forgetting, which occurs when a model forgets how to
perform well after training on a new task. This is a consequence of a neural
network's distributed representation. The term Continual Learning is often
synonymous with overcoming catastrophic forgetting. Non-deep learning methods do
not suffer from catastrophic forgetting. Care should be taken to distinguish
between online continual learning with and without deep learning.

Online continual learning (OCL) differs from data stream learning because the
objective is performance on historic tasks rather than adaptation. Unlike
traditional continual learning, OCL restricts training to a single data pass.
"""

from importlib import import_module

from . import base

_LAZY_SUBMODULES = {"datasets", "evaluation", "util", "strategy", "ann"}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        try:
            module = import_module(f".{name}", package=__name__)
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "torchvision"}:
                raise ImportError(
                    f"openmoa.ocl.{name} requires PyTorch and TorchVision. "
                    "Install them before importing this OCL submodule."
                ) from exc
            raise
        globals()[name] = module
        return module
    raise AttributeError(f"module 'openmoa.ocl' has no attribute {name!r}")


__all__ = ["base", "evaluation", "datasets", "strategy", "util", "ann"]