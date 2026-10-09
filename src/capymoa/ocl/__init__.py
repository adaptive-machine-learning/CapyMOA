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

>>> from capymoa.classifier import HoeffdingTree
>>> from capymoa.ocl.datasets import TinySplitMNIST
>>> from capymoa.ocl import evaluate_ocl
>>> import numpy as np
>>> scenario = TinySplitMNIST()
>>> learner = HoeffdingTree(scenario.schema)
>>> results = evaluate_ocl(learner, scenario.train_loaders(32), scenario.test_loaders(32))

The final accuracy is the accuracy on all tasks after finishing training on all
tasks:

>>> print(f"Final Accuracy: {results['accuracy_final']:0.2f}")
Final Accuracy: 0.69

The accuracy on each task after training on each task:

>>> with np.printoptions(precision=2):
...     print(results['accuracy_matrix'])
[[0.9  0.05 0.05 0.05 0.08]
 [0.88 0.9  0.   0.   0.05]
 [0.77 0.82 0.62 0.   0.03]
 [0.77 0.82 0.6  0.52 0.03]
 [0.77 0.85 0.57 0.52 0.75]]

Notice that the accuracies in the upper triangle are close to zero because the
learner has not trained on those tasks yet. The diagonal contains the accuracy
on each task after training on that task. The lower triangle contains the
accuracy on each task after training on all tasks.

>>> print(f"Forward Transfer: {results['forward_transfer']:0.2f}")
Forward Transfer: 0.03

>>> print(f"Backward Transfer: {results['backward_transfer']:0.2f}")
Backward Transfer: -0.07
"""

# PyTorch is an optional extra; this whole module requires it.
try:
    from . import datasets, evaluation, events, plot, strategy, util
    from .evaluation._loop import evaluate_ocl
except ModuleNotFoundError as _err:  # pragma: no cover
    if (_err.name or "").split(".")[0] in ("torch", "torchvision"):
        from capymoa.exception import OptionalDependencyError

        raise OptionalDependencyError("PyTorch", "capymoa.ocl") from _err
    raise

__all__ = [
    "datasets",
    "evaluate_ocl",
    "evaluation",
    "events",
    "plot",
    "strategy",
    "util",
]
