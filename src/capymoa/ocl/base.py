"""Base classes for online continual learning algorithms.

All OCL learners inherit from :class:`capymoa.base.Classifier` this module
contains additional base classes for OCL learners that are aware of the task
boundaries and/or the task identities during training and evaluation.
"""


class TrainTaskAware:
    """Interface for learners that are aware of the transition between tasks.

    Knowing the transition between tasks is required by some algorithms, but is
    a relaxation of the online continual learning setting. A researcher should
    be mindful and communicate when a learner is task-aware.

    >>> from capymoa.classifier import NoChange
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.base import TrainTaskAware
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop

    >>> class MyTaskBoundaryAware(TrainTaskAware, NoChange):
    ...     def on_train_task(self, train_task_id: int):
    ...         print(f"Training task {train_task_id}")

    >>> scenario = TinySplitMNIST()
    >>> learner = MyTaskBoundaryAware(scenario.schema)
    >>> _ = ocl_train_eval_loop(learner, scenario.train_loaders(32), scenario.test_loaders(32))
    Training task 0
    Training task 1
    Training task 2
    Training task 3
    Training task 4
    """

    def on_train_task(self, task_id: int):
        """Called when a new training task starts."""


class TestTaskAware:
    """Interface for learners that are aware of the task during evaluation.

    Knowing the task during inference greatly simplifies the learning problem.
    When using this interface your problem becomes a task-incremental online
    continual learning problem.

    >>> from capymoa.classifier import NoChange
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.base import TrainTaskAware, TestTaskAware
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop

    >>> class MyTaskAware(TestTaskAware, TrainTaskAware, NoChange):
    ...     def on_train_task(self, train_task_id: int):
    ...         print(f"Training task {train_task_id}")
    ...
    ...     def on_test_task(self, test_task_id: int):
    ...         print(f"Testing task {test_task_id}")

    >>> scenario = TinySplitMNIST()
    >>> learner = MyTaskAware(scenario.schema)
    >>> ocl_train_eval_loop(learner, scenario.train_loaders(32), scenario.test_loaders(32))
    Training task 0
    Testing task 0
    Testing task 1
    Testing task 2
    Testing task 3
    Testing task 4
    Training task 1
    Testing task 0
    Testing task 1
    ...
    """

    def on_test_task(self, task_id: int):
        """Called when testing on a task starts."""
