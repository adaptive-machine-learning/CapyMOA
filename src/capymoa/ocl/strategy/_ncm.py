import torch
from torch import Tensor, nn

from capymoa.base import BatchClassifier
from capymoa.ocl.util._module import TorchMixin
from capymoa.stream import Schema


def _batch_cumulative_mean(
    batch: Tensor, count: int, mean: Tensor
) -> tuple[int, Tensor]:
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


class _NCMModule(nn.Module):
    """Torch state of NCM."""

    def __init__(self, pre_processor: nn.Module, n_classes: int, n_feats: int) -> None:
        super().__init__()
        self.pre_processor = pre_processor
        self.class_counts = nn.Buffer(torch.zeros((n_classes,), dtype=torch.int64))
        self.class_means = nn.Buffer(torch.zeros((n_classes, n_feats)))


class NCM(TorchMixin, BatchClassifier):
    """Nearest Class Mean.

    Nearest Class Mean (NCM) [#f0]_ is a prototype based classifier that uses the mean
    of each class as a prototype. It calculates the distance from each input to
    the class means and assigns class with the closest mean as the predicted
    class.

    >>> from capymoa.ocl.strategy import NCM
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl import evaluate_ocl
    >>> scenario = TinySplitMNIST()
    >>> learner = NCM(scenario.schema)
    >>> results = evaluate_ocl(
    ...     learner,
    ...     scenario.train_loaders(32),
    ...     scenario.test_loaders(32),
    ... )
    >>> print(f"{results['accuracy_final']*100:.1f}%")
    71.5%

    .. [#f0] `Rebuffi, S.-A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017, July).
             iCaRL: Incremental Classifier and Representation Learning. The IEEE
             Conference on Computer Vision and Pattern Recognition (CVPR).
             <https://arxiv.org/abs/1611.07725>`_
    """

    def __init__(
        self,
        schema: Schema,
        pre_processor: nn.Module | None = None,
        num_features: int | None = None,
        device: torch.device | str = "cpu",
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
        if pre_processor is None:
            pre_processor = nn.Identity()
        n_classes = schema.get_num_classes()
        n_feats = num_features or schema.get_num_attributes()
        self._module = _NCMModule(pre_processor, n_classes, n_feats).to(device)

    @torch.no_grad()
    def batch_train(self, x: Tensor, y: Tensor) -> None:
        x = self._module.pre_processor(x)

        # Update mean and count
        for i in range(self.schema.get_num_classes()):
            mask = y == i
            self._module.class_counts[i], self._module.class_means[i] = (
                _batch_cumulative_mean(
                    batch=x[mask],
                    count=int(self._module.class_counts[i].item()),
                    mean=self._module.class_means[i],
                )
            )

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        assert x.ndim == 2, "Input must be a 2D array (batch_size, features)"
        x = self._module.pre_processor(x)

        # Calculate distances to class means
        distances = torch.cdist(x.unsqueeze(0), self._module.class_means.unsqueeze(0))
        distances = distances.squeeze(0)

        # Convert distances to pseudo-probabilities. Using the inverse weighted
        # distance method.
        inv_distances = 1 / (1 + distances)
        probabilities = inv_distances / inv_distances.sum(dim=1, keepdim=True)
        return probabilities
