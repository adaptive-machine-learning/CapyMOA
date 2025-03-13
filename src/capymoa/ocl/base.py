from abc import ABC, abstractmethod


class TaskBoundaryAware(ABC):
    """Interface for learners that are aware of the transition between tasks.

    Knowing the transition between tasks is required by some algorithms, but is
    a relaxation of the online continual learning setting. A researcher should
    be mindful and communicate when a learner is task-aware.

    >>> from capymoa.classifier import NoChange
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.base import TaskBoundaryAware
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop

    >>> class MyTaskBoundaryAware(TaskBoundaryAware, NoChange):
    ...     def set_train_task(self, train_task_id: int):
    ...         print(f"Training task {train_task_id}")

    >>> scenario = TinySplitMNIST()
    >>> learner = MyTaskBoundaryAware(scenario.schema)
    >>> ocl_train_eval_loop(learner, scenario.train_streams, scenario.test_streams)
    Training task 0
    Training task 1
    Training task 2
    Training task 3
    Training task 4
    """

    @abstractmethod
    def set_train_task(self, train_task_id: int):
        """Called when a new training task starts.

        :param task_id: The ID of the new task.
        """


class TaskAware(TaskBoundaryAware):
    """Interface for learners that are aware of the task during evaluation.

    Knowing the task during inference greatly simplifies the learning problem.
    When using this interface your problem becomes a task-incremental online
    continual learning problem.

    >>> from capymoa.classifier import NoChange
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.base import TaskAware
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop

    >>> class MyTaskAware(TaskAware, NoChange):
    ...     def set_train_task(self, train_task_id: int):
    ...         print(f"Training task {train_task_id}")
    ...
    ...     def set_test_task(self, test_task_id: int):
    ...         print(f"Testing task {test_task_id}")

    >>> scenario = TinySplitMNIST()
    >>> learner = MyTaskAware(scenario.schema)
    >>> ocl_train_eval_loop(learner, scenario.train_streams, scenario.test_streams)
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

    @abstractmethod
    def set_test_task(self, test_task_id: int):
        """Called when testing on a task starts.

        :param task_id: The ID of the task.
        """
