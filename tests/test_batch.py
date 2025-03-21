from capymoa.datasets._datasets import ElectricityTiny
from capymoa.base import BatchClassifierSSL
from capymoa.stream._stream import Schema, NumpyStream
from capymoa.evaluation.evaluation import prequential_ssl_evaluation
import numpy as np


class _DummyBatchClassifierSSL(BatchClassifierSSL):
    def __init__(
        self,
        batch_size: int,
        schema: Schema = None,
        random_seed=1,
        class_value_type=int,
    ):
        super().__init__(batch_size=batch_size, schema=schema, random_seed=random_seed)
        self.instance_counter = 0
        self.batch_counter = 0
        self.class_value_type = class_value_type
        self.batch_size = batch_size

    def batch_train(self, x, y):
        # Check type
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

        # Check shape
        assert x.shape == (self.batch_size, self.schema.get_num_attributes())
        assert y.shape == (self.batch_size,)

        # Features must be a numpy array of floats
        assert x.dtype == np.float32
        assert y.dtype == np.int32

        self.batch_counter += 1

    def __str__(self):
        return "EmptyBatchClassifierSSL"

    def train(self, instance):
        super().train(instance)
        self.instance_counter += 1

    def train_on_unlabeled(self, instance):
        super().train_on_unlabeled(instance)
        self.instance_counter += 1

    def predict(self, instance):
        pass

    def predict_proba(self, instance):
        pass


def test_batch_basic():
    """Ensures that the batch classifier is called with the correct batch size and
    that the batch is filled with the correct instances.
    """

    n = 128 * 10
    feature_count = 10
    batch_size = 128
    x = np.arange(n).repeat(feature_count).reshape(n, feature_count)
    y = np.arange(n)
    assert x.shape == (n, feature_count)

    stream = NumpyStream(x, y, target_type="categorical")
    learner = _DummyBatchClassifierSSL(batch_size, stream.schema, class_value_type=str)
    prequential_ssl_evaluation(
        stream=stream, learner=learner, label_probability=0.01, window_size=100
    )

    assert learner.instance_counter == n
    assert learner.batch_counter == n // batch_size


def test_batch_real():
    stream = ElectricityTiny()
    assert stream.schema.get_label_values() == ["0", "1"]
    assert stream.schema.get_num_attributes() == 6

    learner = _DummyBatchClassifierSSL(128, stream.schema, class_value_type=str)
    prequential_ssl_evaluation(
        stream=stream,
        learner=learner,
        label_probability=0.01,
        window_size=100,
    )

    assert learner.instance_counter == 2000
