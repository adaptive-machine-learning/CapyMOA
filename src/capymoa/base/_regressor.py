from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.base import RegressorMixin as _SKRegressorMixin
from torch import Tensor

from capymoa.instance import Instance, RegressionInstance
from capymoa.stream._stream import Schema
from capymoa.type_alias import TargetValue

from ._batch import Batch


class Regressor(ABC):
    def __init__(self, schema=None, random_seed=1):
        self.random_seed = random_seed
        self.schema = schema

    def __str__(self) -> str:
        return str(self.__class__.__name__)

    @abstractmethod
    def train(self, instance: RegressionInstance):
        pass

    @abstractmethod
    def predict(self, instance: RegressionInstance) -> TargetValue:
        pass


class BatchRegressor(Regressor, Batch, ABC):
    """Base class for regressor that support mini-batches.

    Supported by:

    - :func:`capymoa.evaluation.prequential_evaluation`

    Evaluators that support batch classifiers will call the :func:`batch_train`
    and :func:`batch_predict` methods instead of :func:`train` and
    :func:`predict`:

    >>> from capymoa.base import BatchRegressor
    >>> from capymoa.datasets import FriedTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> batch_size = 500
    >>> class MyBatchRegressor(BatchRegressor):
    ...     def batch_train(self, x, y):
    ...         print(f"batch_train x: {x.shape} {x.dtype}")
    ...         print(f"batch_train y: {y.shape} {y.dtype}")
    ...
    ...     def batch_predict(self, x):
    ...         print(f"batch_predict x: {x.shape} {x.dtype}")
    ...         return np.zeros((x.shape[0],))
    ...
    >>> stream = FriedTiny()
    >>> learner = MyBatchRegressor(stream.get_schema())
    >>> _ = prequential_evaluation(
    ...     stream,
    ...     learner,
    ...     batch_size=batch_size,
    ...     max_instances=721
    ... )
    batch_predict x: torch.Size([500, 10]) torch.float32
    batch_train x: torch.Size([500, 10]) torch.float32
    batch_train y: torch.Size([500]) torch.float32
    batch_predict x: torch.Size([221, 10]) torch.float32
    batch_train x: torch.Size([221, 10]) torch.float32
    batch_train y: torch.Size([221]) torch.float32

    You can manually use ``itertools.batched`` (python 3.12) function and
    ``np.stack`` to collect batches of instances as a matrix:

    >>> from itertools import islice
    >>> from capymoa._utils import batched # Not available in python < 3.12
    >>> for i, batch in enumerate(batched(stream, 100)):
    ...     x = np.stack([instance.x for instance in batch])
    ...     y = np.stack([instance.y_value for instance in batch])
    ...     x = torch.from_numpy(x).to(dtype=learner.x_dtype, device=learner.device)
    ...     y = torch.from_numpy(y).to(dtype=learner.y_dtype, device=learner.device)
    ...     learner.batch_train(x, y)
    ...     break
    batch_train x: torch.Size([100, 10]) torch.float32
    batch_train y: torch.Size([100]) torch.float32

    The default implementation of :func:`train` and :func:`predict` calls the
    batch variants with a batch of size 1. This is useful for parts of CapyMOA
    that expect a classifier to be able to train and predict on single
    instances.

    >>> instance = next(stream)
    >>> learner.train(instance)
    batch_train x: torch.Size([1, 10]) torch.float32
    batch_train y: torch.Size([]) torch.float32
    >>> learner.predict(instance)
    batch_predict x: torch.Size([1, 10]) torch.float64
    np.float64(0.0)
    """

    x_dtype: torch.dtype = torch.float32
    y_dtype: torch.dtype = torch.float32

    @abstractmethod
    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Train the classifier with a batch of instances.

        :param x: Batch of :py:attr:`x_dtype` valued feature vectors
            ``(batch_size, num_features)``
        :param y: Batch of :py:attr:`y_dtype` valued targets ``(batch_size,)``.
        """

    @abstractmethod
    def batch_predict(self, x: Tensor) -> Tensor:
        """Return probability estimates for each label in a batch.

        :param x: Batch of :py:attr:`x_dtype` valued feature vectors
            ``(batch_size, num_features)``
        :return: Predicted batch of :py:attr:`y_dtype` valued targets
            ``(batch_size,)``.
        """

    def train(self, instance: RegressionInstance) -> None:
        """Calls :func:`batch_train` with a batch of size 1."""
        x_ = torch.from_numpy(instance.x.reshape(1, -1))
        x_ = x_.to(dtype=self.x_dtype, device=self.device)
        y_ = torch.scalar_tensor(
            instance.y_value, dtype=self.y_dtype, device=self.device
        )
        return self.batch_train(x_, y_)

    def predict(self, instance: RegressionInstance) -> TargetValue:
        """Calls :func:`batch_predict` with a batch of size 1."""
        x_ = torch.from_numpy(instance.x.reshape(1, -1))
        return np.float64(self.batch_predict(x_).item())


class MOARegressor(Regressor):
    def __init__(self, schema=None, CLI=None, random_seed=1, moa_learner=None):
        super().__init__(schema=schema, random_seed=random_seed)
        self.CLI = CLI
        self.moa_learner = moa_learner

        if random_seed is not None:
            self.moa_learner.setRandomSeed(random_seed)

        if self.schema is not None:
            self.moa_learner.setModelContext(self.schema.get_moa_header())

        if self.CLI is not None:
            self.moa_learner.getOptions().setViaCLIString(CLI)

        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearning()
        self.moa_learner.setModelContext(self.schema.get_moa_header())

    def __str__(self):
        full_name = str(self.moa_learner.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name

    # def describe(self):
    #     return str(self.moa_learner)

    def cli_help(self):
        return self.moa_learner.getOptions().getHelpString()

    def train(self, instance):
        self.moa_learner.trainOnInstance(instance.java_instance)

    def predict(self, instance):
        prediction_array = self.moa_learner.getVotesForInstance(instance.java_instance)
        # The learner didn't provide a prediction, returns 0.0 (probably the learner has not been initialised.)
        if len(prediction_array) == 0:
            return 0.0
        return prediction_array[0]


class SKRegressor(Regressor):
    """A wrapper class for using scikit-learn regressors in CapyMOA.

    Some of scikit-learn's regressors that are compatible with online learning
    have been wrapped and tested already in CapyMOA (See :mod:`capymoa.regressor`).

    However, if you want to use a scikit-learn regressor that has not been
    wrapped yet, you can use this class to wrap it yourself. This requires
    that the scikit-learn regressor implements the ``partial_fit`` and
    ``predict`` methods.

    For example, the following code demonstrates how to use a scikit-learn
    regressor in CapyMOA:

    >>> from sklearn.linear_model import SGDRegressor
    >>> from capymoa.datasets import Fried
    >>> stream = Fried()
    >>> sklearner = SGDRegressor(random_state=1)
    >>> learner = SKRegressor(sklearner, stream.get_schema())
    >>> for _ in range(10):
    ...     instance = stream.next_instance()
    ...     prediction = learner.predict(instance)
    ...     if prediction is not None:
    ...         print(f"y_value: {instance.y_value}, y_prediction: {prediction:.2f}")
    ...     else:
    ...         print(f"y_value: {instance.y_value}, y_prediction: None")
    ...     learner.train(instance)
    y_value: 17.949, y_prediction: None
    y_value: 13.815, y_prediction: 0.60
    y_value: 20.766, y_prediction: 1.30
    y_value: 18.301, y_prediction: 1.86
    y_value: 22.989, y_prediction: 2.28
    y_value: 25.986, y_prediction: 2.65
    y_value: 17.15, y_prediction: 3.51
    y_value: 14.006, y_prediction: 3.25
    y_value: 18.566, y_prediction: 3.80
    y_value: 12.107, y_prediction: 3.87

    A word of caution: even compatible scikit-learn regressors are not
    necessarily designed for online learning and might require some tweaking
    to work well in an online setting.

    See also :class:`capymoa.base.SKClassifier` for scikit-learn classifiers.
    """

    sklearner: _SKRegressorMixin
    """The underlying scikit-learn object."""

    def __init__(
        self, sklearner: _SKRegressorMixin, schema: Schema = None, random_seed: int = 1
    ):
        """Construct a scikit-learn regressor wrapper.

        :param sklearner: A scikit-learn classifier object to wrap that must
            implements ``partial_fit`` and ``predict``.
        :param schema: Describes the structure of the datastream.
        :param random_seed: Random seed for reproducibility.
        :raises ValueError: If the scikit-learn algorithm does not implement
            ``partial_fit`` or ``predict``.
        """
        super().__init__(schema=schema, random_seed=random_seed)

        # Checks if it implements partial_fit and predict
        if not hasattr(sklearner, "partial_fit") or not hasattr(sklearner, "predict"):
            raise ValueError(
                "Invalid scikit-learn algorithm provided. The algorithm does not implement partial_fit or predict. "
            )

        self.sklearner = sklearner
        self._trained_at_least_once = False

    def __str__(self):
        return str(self.sklearner)

    def train(self, instance: RegressionInstance):
        self.sklearner.partial_fit(
            [instance.x],
            [instance.y_value],
        )
        self._trained_at_least_once = True

    def predict(self, instance: Instance) -> float:
        if not self._trained_at_least_once:
            # scikit-learn does not allows invoking predict in a model that was not fit before
            return None
        return self.sklearner.predict([instance.x])[0]
