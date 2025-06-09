import torch
from typing import Tuple
from capymoa.base import BatchClassifier
from capymoa.stream import Schema
from torch import Tensor, nn
from ._ncm import _batch_cumulative_mean


def _batch_cumulative_covariance(
    batch: Tensor, count: int, mean: Tensor, covariance: Tensor
) -> Tuple[int, Tensor, Tensor]:
    """Update cumulative count, mean, and covariance with batch **Welford's** algorithm.

    See:
    * "Algorithms for calculating variance"
      https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

    * "python (numpy) implementation of welford algorithm"
      https://github.com/nickypro/welford-torch

    :param batch: A batch of features ``(batch size, num features)``
    :param count: Current count of samples. Initially zero.
    :param mean: Current mean of samples ``(num_features,)``. Initially zero.
    :param covariance: Covariance matrix ``(num_features, num_features)``. Initially zero.
    :return: Updated count, mean, and covariance.
    """
    batch_size = batch.size(0)
    if batch_size == 0:
        return count, mean, covariance
    new_count = count + batch_size
    new_mean = (count * mean + batch.sum(0)) / new_count
    cov_update = torch.einsum("ni,nj->ij", batch - mean, batch - new_mean)
    covariance = (count * covariance + cov_update) / new_count
    return new_count, new_mean, covariance


class SLDA(BatchClassifier):
    """Streaming Linear Discriminant Analysis (SLDA).

    SLDA incrementally accumulates the mean of each class and a joint mean and
    covariance for all classes. Note that this method does not gracefully forget
    and may not handle concept drift well.

    [Hayes20]_ uses SLDA ontop of a pre-trained model to perform continual
    learning. See [WikipediaILDA]_ and [Ghassabeh2015]_ for more details on
    incremental LDA outside of continual learning.

    ..  [Hayes20] Hayes, T. L., & Kanan, C. (2020). Lifelong Machine Learning with Deep
        Streaming Linear Discriminant Analysis. CLVision Workshop at CVPR 2020, 1–15.
    ..  [Ghassabeh2015] Ghassabeh, Y. A., Rudzicz, F., & Moghaddam, H. A. (2015). Fast incremental
        LDA feature extraction. Pattern Recognition, 48(6), 1999-2012.
    ..  [WikipediaILDA] https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Incremental_LDA
    """

    def __init__(
        self,
        schema: Schema,
        pre_processor: nn.Module = nn.Identity(),
        num_features: int | None = None,
        ridge: float = 1e-6,
        device: torch.device | str = torch.device("cpu"),
    ):
        """Initialize a SLDA classifier.

        :param schema: Describes the shape and type of the data.
        :param pre_processor: A pre-processing module to apply to the input
            data, defaults to an identity module.
        :param num_features: Number of features once pre-processed, defaults to
            the number of attributes in the schema.
        :param ridge: Ridge regularization term to avoid singular covariance matrix,
            defaults to 1e-6.
        :param device: Device to run the model on, defaults to CPU.
        """
        super().__init__(schema)
        self._pre_processor = pre_processor.to(device)
        self._n_classes = schema.get_num_classes()
        self._n_feats = num_features or schema.get_num_attributes()
        self.device = torch.device(device)

        # Class means and counts
        self._class_counts = torch.zeros((self._n_classes,), device=device)
        self._class_means = torch.zeros((self._n_classes, self._n_feats), device=device)

        # Joint cumulative mean and covariance
        self._count = 0
        self._mean = torch.zeros(self._n_feats, device=device)
        self._covariance = torch.eye(self._n_feats, device=device)
        self._ridge = torch.eye(self._n_feats, device=device) * ridge

    @torch.no_grad()
    def batch_train(self, x: Tensor, y: Tensor) -> None:
        x = self._pre_processor(x)

        # Update class means and counts
        for i in range(self.schema.get_num_classes()):
            self._class_counts[i], self._class_means[i] = _batch_cumulative_mean(
                x[y == i], int(self._class_counts[i]), self._class_means[i]
            )

        # Update joint cumulative mean and covariance
        self._count, self._mean, self._covariance = _batch_cumulative_covariance(
            x, self._count, self._mean, self._covariance
        )

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        # Return uniform probabilities if no training has been done
        if self._count == 0:
            return torch.full(
                (x.size(0), self._n_classes),
                1.0 / self._n_classes,
                dtype=self.x_dtype,
                device=self.device,
            )

        x = self._pre_processor(x)
        covariance = self._covariance / self._count + self._ridge
        weights: Tensor = torch.linalg.solve(covariance, self._class_means.T).T
        bias = -0.5 * (self._class_means @ weights.T).diagonal()
        bias += torch.log(self._class_counts / self._count)
        scores = x @ weights.T
        return torch.softmax(scores, dim=1)
