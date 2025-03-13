from typing import Sequence, Union

from capymoa.base import Classifier
from capymoa.instance import LabeledInstance
from capymoa.ocl.base import TaskAware, TaskBoundaryAware
from capymoa.stream import Stream
from dataclasses import dataclass

import torch


@dataclass
class OCLTrainingResults:
    pass


@torch.no_grad()
def ocl_train_eval_loop(
    learner: Union[TaskBoundaryAware, TaskAware, Classifier],
    train_streams: Sequence[Stream[LabeledInstance]],
    test_streams: Sequence[Stream[LabeledInstance]],
):
    n_tasks = len(train_streams)
    if n_tasks != len(test_streams):
        raise ValueError("Number of train and test tasks must be equal")
    if not isinstance(learner, Classifier):
        raise ValueError("Learner must be a classifier")

    for train_task_id, train_stream in enumerate(train_streams):
        train_stream.restart()
        assert len(train_stream)

        if isinstance(learner, TaskBoundaryAware):
            learner.set_train_task(train_task_id)

        # train and evaluation loop
        with torch.enable_grad():
            for instance in train_stream:
                learner.predict(instance)
                learner.train(instance)

        # evaluate the learner on past and future tasks
        for test_task_id, test_stream in enumerate(test_streams):
            test_stream.restart()
            assert len(test_stream)

            if isinstance(learner, TaskAware):
                learner.set_test_task(test_task_id)

            # predict instances in the current task
            for instance in test_stream:
                pass
