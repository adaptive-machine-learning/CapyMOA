import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from capymoa.base import BatchClassifier
from capymoa.ocl.evaluation.events import TestTaskBegin
from capymoa.ocl.events import Dispatcher, Handler
from capymoa.ocl.util._module import TorchMixin
from capymoa.ocl.util._replay import GreedySampler
from capymoa.stream import Schema


class _GDumbModule(nn.Module):
    """Torch state of GDumb."""

    def __init__(self, model: nn.Module, coreset: GreedySampler) -> None:
        super().__init__()
        self.model = model
        self.coreset = coreset


class GDumb(TorchMixin[_GDumbModule], BatchClassifier, Handler):
    """Greedy sampler and a dumb learner.

    Greedy sampler and a dumb learner (GDumb) [#f0]_ is a baseline replay strategy. It
    works by down sampling the dataset and training offline. Since online learners do
    not have an inference time, GDumb is an offline algorithm, but GDumb remains a
    useful baseline.

    .. [#f0] `Prabhu, A., Torr, P. H. S., & Dokania, P. K. (2020). GDumb: A Simple
              Approach that Questions Our Progress in Continual Learning. In A. Vedaldi,
              H. Bischof, T. Brox, & J.-M. Frahm (Eds.), Computer Vision – ECCV 2020
              (pp. 524–540). Springer International Publishing.
              <https://doi.org/10.1007/978-3-030-58536-5_31>`_
    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        epochs: int,
        batch_size: int,
        capacity: int,
        lr: float = 0.001,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ):
        super().__init__(schema)
        self.schema = schema
        self.x_dtype = next(model.parameters(), torch.empty(0)).dtype
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.original_state_dict = model.state_dict()
        self._seed = seed
        coreset = GreedySampler(
            capacity,
            {
                "x": ((schema.get_num_attributes(),), self.x_dtype),
                "y": ((), torch.long),
            },
            torch.Generator().manual_seed(seed),
        )
        self.loss_func = nn.CrossEntropyLoss()
        self._module = _GDumbModule(model, coreset).to(device)

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self._module.coreset.update(x=x, y=y)

    def batch_predict_proba(self, x: Tensor) -> Tensor:
        return self._module.model(x).softmax(dim=1)

    def gdumb_fit(self) -> None:
        """
        Fit the model on the coreset.
        """
        # Assemble a dataset from the buffer
        dataset = self._module.coreset.dataset_view()

        self._module.model.load_state_dict(self.original_state_dict)
        self._module.model.train()
        optimizer = torch.optim.Adam(self._module.model.parameters(), lr=self.lr)

        # Derive the shuffling generator from the seed so the offline fit is
        # reproducible independently of the global torch RNG state.
        generator = torch.Generator().manual_seed(self._seed)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self._module.model(batch_x)
                loss = self.loss_func(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def on_test_task(self, task_id: int) -> None:
        if task_id == 0:
            self.gdumb_fit()

    def attach_with(self, source: Dispatcher) -> "GDumb":
        source.subscribe(TestTaskBegin, self._on_test_task_start)
        return self

    def _on_test_task_start(self, event: TestTaskBegin) -> None:
        self.on_test_task(event.test_task)
