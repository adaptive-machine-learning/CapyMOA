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
>>> from capymoa.ocl.evaluation import ocl_train_eval_loop
>>> import numpy as np
>>> scenario = TinySplitMNIST()
>>> learner = HoeffdingTree(scenario.schema)
>>> metrics = ocl_train_eval_loop(learner, scenario.train_loaders(32), scenario.test_loaders(32))

The final accuracy is the accuracy on all tasks after finishing training on all
tasks:

>>> print(f"Final Accuracy: {metrics.accuracy_final:0.2f}")
Final Accuracy: 0.69

The accuracy on each task after training on each task:

>>> with np.printoptions(precision=2):
...     print(metrics.accuracy_matrix)
[[0.9  0.   0.   0.3  0.  ]
 [0.88 0.9  0.   0.12 0.  ]
 [0.77 0.82 0.62 0.12 0.  ]
 [0.77 0.82 0.6  0.52 0.  ]
 [0.77 0.82 0.57 0.52 0.75]]

Notice that the accuracies in the upper triangle are close to zero because the
learner has not trained on those tasks yet. The diagonal contains the accuracy
on each task after training on that task. The lower triangle contains the
accuracy on each task after training on all tasks.

>>> print(f"Forward Transfer: {metrics.forward_transfer:0.2f}")
Forward Transfer: 0.05

>>> print(f"Backward Transfer: {metrics.backward_transfer:0.2f}")
Backward Transfer: -0.07
"""

from . import base, datasets, evaluation, util, strategy

__all__ = ["evaluation", "datasets", "strategy", "base", "util"]
