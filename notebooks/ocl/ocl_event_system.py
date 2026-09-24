# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: capymoa
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Online Continual Learning and Event Handlers
#
# The OCL model is experimenting with an event-based system to facilitate communication
# between different objects. This allows objects to hook into stages of training, evaluation, and inference. A common use is to implement custom metrics or loggers.

# %% [markdown]
# This block shows how to use the event system in isolation:

# %%
from dataclasses import dataclass

from capymoa.ocl.events import Dispatcher, Event


@dataclass
class MyEvent(Event):
    value: int


dispatcher = Dispatcher()


def print_event_type(event):
    print(type(event))


dispatcher.subscribe(MyEvent, print_event_type)
dispatcher.notify(MyEvent(42))

# %% [markdown]
# This block shows how to use the event system within the OCL training loop. You can
# subscribe to the `None` event to get notified about every event emitted.

# %%
from capymoa.classifier import NoChange
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.evaluation import ocl_train_eval_loop

dispatcher = Dispatcher()
# Subscribe to all events by using None as the event type
dispatcher.subscribe(None, print_event_type)

scenario = TinySplitMNIST()
_ = ocl_train_eval_loop(
    NoChange(scenario.schema),
    train_streams=scenario.train_loaders(64)[:1],
    test_streams=scenario.test_loaders(64)[:1],
    dispatcher=dispatcher,
)

# %% [markdown]
# We can use the `TestTaskBegin` and `TrainTaskBegin` events to implement a
# task-incremental learner. To standardize subscription to a dispatcher, our method
# implements `attach_with` from the `Handler` class.

# %%
from torch import Tensor

from capymoa.base import BatchClassifier
from capymoa.classifier import Finetune
from capymoa.core.torch.ann import Perceptron
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin
from capymoa.ocl.events import Dispatcher, Handler


class PerceptronTI(BatchClassifier, Handler):
    def __init__(self, schema, n_tasks: int) -> None:
        super().__init__(schema)
        self._classifiers = [Finetune(schema, Perceptron) for _ in range(n_tasks)]
        self._train_task = 0
        self._test_task = 0

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self._classifiers[self._train_task].batch_train(x, y)

    def batch_predict_proba(self, x: Tensor) -> Tensor:
        return self._classifiers[self._test_task].batch_predict_proba(x)

    def _on_train_task_begin(self, event: TrainTaskBegin) -> None:
        print(f"Train task {event.train_task} has begun.")
        self._train_task = event.train_task

    def _on_test_task_begin(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    def attach_with(self, dispatcher: Dispatcher) -> Handler:
        dispatcher.subscribe(TrainTaskBegin, self._on_train_task_begin)
        dispatcher.subscribe(TestTaskBegin, self._on_test_task_begin)
        return super().attach_with(dispatcher)


learner = PerceptronTI(scenario.schema, n_tasks=5)
results = ocl_train_eval_loop(
    learner,  # If the learner is a Handler, it will be automatically subscribed.
    train_streams=scenario.train_loaders(32),
    test_streams=scenario.test_loaders(32),
)
print(f"Accuracy {results.accuracy_seen_avg * 100:.2f}")

# %% [markdown]
# If you use a custom train-test loop with a `Classifier` that is also a `Handler` you
# will need to manually notify the classifier as required.
