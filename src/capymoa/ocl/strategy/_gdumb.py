from capymoa.ocl.util._coreset import GreedySampler
import torch
from capymoa.base import BatchClassifier
from capymoa.ocl.base import TestTaskAware
from capymoa.stream import Schema
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class GDumb(BatchClassifier, TestTaskAware):
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
        self.model = model.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.fit_device = torch.device(device)

        self.original_state_dict = model.state_dict()
        self.coreset = GreedySampler(
            capacity, schema.get_num_attributes(), torch.Generator().manual_seed(seed)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self.coreset.update(x, y)

    def batch_predict_proba(self, x: Tensor) -> Tensor:
        return self.model(x).softmax(dim=1)

    def gdumb_fit(self) -> None:
        """
        Fit the model on the coreset.
        """
        # Assemble a dataset from the buffer
        dataset = TensorDataset(*self.coreset.array())

        self.model.load_state_dict(self.original_state_dict)
        self.model.to(self.fit_device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.fit_device)
                batch_y = batch_y.to(self.fit_device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.loss_func(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def on_test_task(self, task_id: int) -> None:
        if task_id == 0:
            self.gdumb_fit()
