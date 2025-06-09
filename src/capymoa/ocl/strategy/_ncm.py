import torch
from typing import Tuple
from capymoa.base import BatchClassifier
from capymoa.stream import Schema
from torch import Tensor, nn


def _batch_cumulative_mean(
    batch: Tensor, count: int, mean: Tensor
) -> Tuple[int, Tensor]:
    """Update cumulative mean and count.

    :param batch: Current batch of data. Shape (batch_size, num_features).
    :param count: Current count of samples processed.
    :param mean: Current cumulative mean of the data. Shape (num_features,).
    :return: Updated count and cumulative mean.
    """
    batch_size = batch.size(0)
    if batch_size == 0:
        return count, mean
    new_count = count + batch_size
    updated_mean = (count * mean + batch.sum(0)) / new_count
    return new_count, updated_mean


class NCM(BatchClassifier):
    """Nearest Class Mean (NCM).

    NCM is a simple classifier that uses the mean of each class as a prototype.
    It calculates the distance from each input to the class means and assigns
    the class with the closest mean as the predicted class.
    """

    _dtype = torch.float32

    def __init__(
        self,
        schema: Schema,
        pre_processor: nn.Module = nn.Identity(),
        num_features: int | None = None,
        device: torch.device | str = torch.device("cpu"),
    ):
        """Initialize a NCM classifier head.

        :param schema: Describes the shape and type of the data.
        :param pre_processor: A pre-processing module to apply to the input
            data, defaults to an identity module.
        :param num_features: Number of features once pre-processed, defaults to
            the number of attributes in the schema.
        :param device: Device to run the model on, defaults to CPU.
        """
        super().__init__(schema)
        n_classes = schema.get_num_classes()
        n_feats = num_features or schema.get_num_attributes()
        self._device = device
        self._pre_processor = pre_processor.to(device)
        self._class_counts = torch.zeros((n_classes,), device=device, dtype=torch.int64)
        self._class_means = torch.zeros((n_classes, n_feats), device=device)

    @torch.no_grad()
    def batch_train(self, x: Tensor, y: Tensor) -> None:
        x = self._pre_processor(x)

        # Update mean and count
        for i in range(self.schema.get_num_classes()):
            mask = y == i
            self._class_counts[i], self._class_means[i] = _batch_cumulative_mean(
                batch=x[mask],
                count=int(self._class_counts[i].item()),
                mean=self._class_means[i],
            )

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        assert x.ndim == 2, "Input must be a 2D array (batch_size, features)"
        x = self._pre_processor(x)

        # Calculate distances to class means
        distances = torch.cdist(x.unsqueeze(0), self._class_means.unsqueeze(0))
        distances = distances.squeeze(0)

        # Convert distances to pseudo-probabilities. Using the inverse weighted
        # distance method.
        inv_distances = 1 / (1 + distances)
        probabilities = inv_distances / inv_distances.sum(dim=1, keepdim=True)
        return probabilities
