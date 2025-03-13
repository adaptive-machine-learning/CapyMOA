from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.evaluation import ocl_train_eval_loop
from capymoa.ocl.base import TaskBoundaryAware

from capymoa.classifier import NoChange


class MyTaskAware(TaskBoundaryAware, NoChange):
    def set_train_task(self, train_task_id: int):
        print(f"Training task {train_task_id}")


def test_ocl_train_eval_loop():
    scenario = TinySplitMNIST()
    learner = MyTaskAware(scenario.schema)
    ocl_train_eval_loop(learner, scenario.train_streams, scenario.test_streams)
