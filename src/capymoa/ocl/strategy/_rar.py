import torch
from torch import Tensor

from capymoa.base import BatchClassifier
from capymoa.ocl.util._coreset import ReservoirSampler
from capymoa.ocl.base import TrainTaskAware, TestTaskAware

from typing import Callable


class RAR(BatchClassifier, TrainTaskAware, TestTaskAware):
    """Repeated Augmented Rehearsal.

    Repeated Augmented Rehearsal (RAR) [#f0]_ is a replay continual learning
    strategy that combines data augmentation with repeated training on each
    batch to mitigate catastrophic forgetting.

    * Coreset Selection: Reservoir sampling is used to select a fixed-size
      buffer of past examples.

    * Coreset Retrieval: During training, the learner samples uniformly from the
      buffer of past examples.

    * Coreset Exploitation: The learner trains on the current batch of examples
      and the sampled buffer examples, performing multiple optimization steps
      per-batch using random augmentations of the examples. The original paper uses
      RandAugment [#f1]_ for augmentation but any randomized augmentation can be used.
      But the choice of augmentation is important and should be chosen based on the
      problem domain.

    * Not :class:`~capymoa.ocl.base.TrainTaskAware` or
      :class:`~capymoa.ocl.base.TestTaskAware`, but will proxy it to the wrapped
      learner.

    >>> from capymoa.ann import Perceptron
    >>> from capymoa.classifier import Finetune
    >>> from capymoa.ocl.strategy import RAR
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop
    >>> import torchvision.transforms as T
    >>> import torch
    >>> _ = torch.manual_seed(0)
    >>> scenario = TinySplitMNIST()
    >>> model = Perceptron(scenario.schema)
    >>> # You should use more complex augmentations for more challenging problems.
    >>> augment = T.Compose([
    ...     T.RandomRotation(10),
    ... ])
    >>> learner = RAR(Finetune(scenario.schema, model), augment=augment, repeats=5)
    >>> results = ocl_train_eval_loop(
    ...     learner,
    ...     scenario.train_loaders(32),
    ...     scenario.test_loaders(32),
    ... )
    >>> print(f"{results.accuracy_final*100:.1f}%")
    46.0%

    Usually more complex augmentations are used such as random crops and
    rotations.

    .. [#f0] Zhang, Yaqian, Bernhard Pfahringer, Eibe Frank, Albert Bifet, Nick
       Jin Sean Lim, and Yunzhe Jia. “A Simple but Strong Baseline for Online
       Continual Learning: Repeated Augmented Rehearsal.” In Advances in Neural
       Information Processing Systems 35: Annual Conference on Neural
       Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA,
       November 28 - December 9, 2022, edited by Sanmi Koyejo, S. Mohamed, A.
       Agarwal, Danielle Belgrave, K. Cho, and A. Oh, 2022.
       https://doi.org/10.5555/3600270.3601344.

    .. [#f1] Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). Randaugment:
       Practical automated data augmentation with a reduced search space. 2020 IEEE/CVF
       Conference on Computer Vision and Pattern Recognition Workshops (CVPRW),
       3008-3017. https://doi.org/10.1109/CVPRW50498.2020.00359
    """

    def __init__(
        self,
        learner: BatchClassifier,
        augment: Callable[[Tensor], Tensor],
        coreset_size: int = 200,
        repeats: int = 1,
    ) -> None:
        """Initialize Repeated Augmented Rehearsal.

        :param learner: Underlying learner to be trained with RAR.
        :param augment: Data augmentation function to apply to the samples. Should take
            a Tensor of shape ``(batch_size, *schema.shape)`` and return a Tensor of the
            same shape. Take a look at the PyTorch torchvision transforms for some
            building blocks for your pipeline (https://docs.pytorch.org/vision/main/transforms.html).
        :param coreset_size: Size of the coreset buffer.
        :param repeats: Number of times to repeat training on each batch, defaults to 1.
        """

        super().__init__(learner.schema)
        num_features = learner.schema.get_num_attributes()
        self.learner = learner
        self.augment = augment
        self.repeats = repeats
        self.coreset = ReservoirSampler(
            coreset_size,
            num_features,
            rng=torch.Generator().manual_seed(learner.random_seed),
        )
        self.shape = learner.schema.shape

    def train_step(self, x_fresh: Tensor, y_fresh: Tensor) -> None:
        # Sample from reservoir and augment the data
        n = x_fresh.shape[0]
        x_replay, y_replay = self.coreset.sample(n)
        x = torch.cat((x_fresh, x_replay), dim=0).to(self.device, self.x_dtype)
        y = torch.cat((y_fresh, y_replay), dim=0).to(self.device, self.y_dtype)
        x = x.view(-1, *self.shape)
        x: Tensor = self.augment(x)

        # Train the learner
        x = x.to(self.learner.device, self.learner.x_dtype)
        y = y.to(self.learner.device, self.learner.y_dtype)
        self.learner.batch_train(x, y)

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self.coreset.update(x, y)
        for i in range(self.repeats):
            self.train_step(x, y)

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        x = x.to(self.learner.device, self.learner.x_dtype)
        return self.learner.batch_predict_proba(x)

    def on_test_task(self, task_id: int):
        if isinstance(self.learner, TestTaskAware):
            self.learner.on_test_task(task_id)

    def on_train_task(self, task_id: int):
        if isinstance(self.learner, TrainTaskAware):
            self.learner.on_train_task(task_id)
