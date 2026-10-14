import torch
from torch import Tensor, nn

from capymoa.base import BatchClassifier
from capymoa.stream import Schema

from ._ncm import _batch_cumulative_mean


def _batch_cumulative_covariance(
    batch: Tensor, count: int, mean: Tensor, covariance: Tensor
) -> tuple[int, Tensor, Tensor]:
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
    r"""Streaming Linear Discriminant Analysis.

    Streaming Linear Discriminant Analysis (SLDA) [#f0]_ is a prototype classifier.
    It keeps a running mean per class and one shared covariance matrix for all
    classes. Used on top of a frozen pre-trained model, it does continual learning
    with no forgetting [#f0]_. It does not forget old data, so it may not handle
    concept drift well.

    Other streaming Gaussian discriminant methods for continual learning are
    RanPAC [#f1]_ (see :class:`RanPAC`), FeCAM [#f2]_, RanDumb [#f3]_, and the
    kernel method of Momeni et al. [#f4]_. They mostly differ in how they
    pre-process the embeddings. For incremental LDA outside continual learning see
    [#f5]_ and [#f6]_.

    **Training.** For a batch :math:`B` of :math:`b` samples, after :math:`n`
    samples seen, the shared mean and covariance use the batch form of Welford's
    algorithm [#f7]_:

    ..  math::

        n' &= n + b \\
        \mu' &= \frac{n \mu + \sum_{x \in B} x}{n'} \\
        \Sigma' &= \frac{n \Sigma + \sum_{x \in B} (x - \mu)(x - \mu')^\top}{n'}

    Each class mean :math:`\mu_k` and count :math:`n_k` use the same mean update
    with only the samples of class :math:`k`. The result does not depend on the
    order of the data.

    **Prediction.** With ridge term :math:`\lambda` (``ridge``):

    ..  math::

        \Lambda &= (\Sigma + \lambda I)^{-1} \\
        w_k &= \Lambda \mu_k \\
        b_k &= -\tfrac{1}{2} \mu_k^\top \Lambda \mu_k + \log \frac{n_k}{n} \\
        p(y = k \mid x) &= \operatorname{softmax}_k \left( w_k^\top x + b_k \right)

    >>> from capymoa.ocl.strategy import SLDA
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop
    >>> scenario = TinySplitMNIST()
    >>> learner = SLDA(scenario.schema)
    >>> results = ocl_train_eval_loop(
    ...     learner,
    ...     scenario.train_loaders(32),
    ...     scenario.test_loaders(32),
    ... )
    >>> print(f"{results.accuracy_final*100:.1f}%")
    80.5%

    .. [#f0] `Hayes, T. L., & Kanan, C. (2020). Lifelong Machine Learning with Deep
              Streaming Linear Discriminant Analysis. CLVision Workshop at CVPR 2020,
              1–15. <https://arxiv.org/abs/1909.01520>`_
    .. [#f1] `McDonnell, M. D., Gong, D., Parvaneh, A., Abbasnejad, E., & van den
              Hengel, A. (2023). RanPAC: Random Projections and Pre-trained Models for
              Continual Learning. NeurIPS 2023. <https://arxiv.org/abs/2307.02251>`_
    .. [#f2] `Goswami, D., Liu, Y., Twardowski, B., & van de Weijer, J. (2023). FeCAM:
              Exploiting the Heterogeneity of Class Distributions in Exemplar-Free
              Continual Learning. NeurIPS 2023. <https://arxiv.org/abs/2309.14062>`_
    .. [#f3] `Prabhu, A., Sinha, S., Kumaraguru, P., Torr, P. H. S., Sener, O., &
              Dokania, P. K. (2024). RanDumb: A Simple Approach that Questions the
              Efficacy of Continual Representation Learning. <https://arxiv.org/abs/2402.08823>`_
    .. [#f4] `Momeni, S., Mazumder, S., & Liu, B. (2025). Continual Learning Using a
              Kernel-Based Method Over Foundation Models. AAAI 2025, 19528–19536.
              <https://doi.org/10.1609/aaai.v39i18.34150>`_
    .. [#f5] `Ghassabeh, Y. A., Rudzicz, F., & Moghaddam, H. A. (2015). Fast incremental
              LDA feature extraction. Pattern Recognition, 48(6), 1999-2012.
              <https://doi.org/10.1016/j.patcog.2014.12.012>`_
    .. [#f6] `Linear discriminant analysis. (2025). In Wikipedia.
              <https://en.wikipedia.org/w/index.php?title=Linear_discriminant_anal
              ysis&oldid=1295914652#Incremental_LDA>`_
    .. [#f7] `Welford, B. P. (1962). Note on a Method for Calculating Corrected Sums
              of Squares and Products. Technometrics, 4(3), 419–420.
              <https://doi.org/10.1080/00401706.1962.10490022>`_
    """

    def __init__(
        self,
        schema: Schema,
        pre_processor: nn.Module | None = None,
        num_features: int | None = None,
        ridge: float = 1e-3,
        device: torch.device | str | None = None,
    ):
        """Initialize a SLDA classifier.

        :param schema: Describes the shape and type of the data.
        :param pre_processor: A pre-processing module to apply to the input
            data, defaults to an identity module.
        :param num_features: Number of features once pre-processed, defaults to
            the number of attributes in the schema.
        :param ridge: Added to the covariance diagonal so it can be inverted,
            defaults to 1e-3.
        :param device: Device to run the model on, defaults to CPU.
        """
        super().__init__(schema)
        if pre_processor is None:
            pre_processor = nn.Identity()
        if device is None:
            device = torch.device("cpu")
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
        covariance = self._covariance + self._ridge
        weights: Tensor = torch.linalg.solve(covariance, self._class_means.T).T
        bias = -0.5 * (self._class_means * weights).sum(1)
        bias += torch.log(self._class_counts / self._count)
        scores = x @ weights.T + bias
        return torch.softmax(scores, dim=1)


class RanPAC(SLDA):
    r"""Random Projections and Pre-trained Models (RanPAC).

    RanPAC [#f0]_ is a variant of :class:`SLDA`. It maps each input :math:`x \in
    \mathbb{R}^d` to :math:`M` random features, then fits SLDA on them:

    ..  math::

        h = \operatorname{ReLU}(W x), \quad W \in \mathbb{R}^{M \times d},
        \quad W_{ij} \sim \mathcal{N}(0, 1)

    :math:`W` is fixed and never trained. With :math:`M \gg d` the classes are
    more linearly separable, which helps the linear SLDA head [#f0]_. Tune
    ``ridge`` together with ``projection_dim``.

    Unlike the original, this version does not adapt the backbone on the first
    task. The head also differs. The original solves ridge regression with the
    uncentred Gram matrix of the random features :math:`H` and one-hot labels
    :math:`Y`:

    ..  math::

        W_o = (H^\top H + \lambda I)^{-1} H^\top Y

    This version uses the SLDA head, which inverts the shared covariance
    :math:`\Sigma + \lambda I` and adds a class prior bias.

    >>> from capymoa.ocl.strategy import RanPAC
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop
    >>> scenario = TinySplitMNIST()
    >>> learner = RanPAC(scenario.schema, projection_dim=512)
    >>> results = ocl_train_eval_loop(
    ...     learner,
    ...     scenario.train_loaders(32),
    ...     scenario.test_loaders(32),
    ... )
    >>> print(f"{results.accuracy_final*100:.1f}%")
    80.0%

    .. [#f0] `McDonnell, M. D., Gong, D., Parvaneh, A., Abbasnejad, E., & van den
              Hengel, A. (2023). RanPAC: Random Projections and Pre-trained Models for
              Continual Learning. NeurIPS 2023. <https://arxiv.org/abs/2307.02251>`_
    """

    def __init__(
        self,
        schema: Schema,
        projection_dim: int = 5000,
        ridge: float = 1e-3,
        device: torch.device | str | None = None,
        random_seed: int = 1,
    ):
        """Initialize a RanPAC classifier.

        :param schema: Describes the shape and type of the data.
        :param projection_dim: Number of random features :math:`M`, defaults to 5000.
        :param ridge: Added to the covariance diagonal so it can be inverted,
            defaults to 1e-3.
        :param device: Device to run the model on, defaults to CPU.
        :param random_seed: Seed for the random projection, defaults to 1.
        """
        generator = torch.Generator().manual_seed(random_seed)
        projection = nn.Linear(schema.get_num_attributes(), projection_dim, bias=False)
        with torch.no_grad():
            projection.weight.normal_(generator=generator)
        super().__init__(
            schema,
            pre_processor=nn.Sequential(projection, nn.ReLU()),
            num_features=projection_dim,
            ridge=ridge,
            device=device,
        )
        self.random_seed = random_seed
