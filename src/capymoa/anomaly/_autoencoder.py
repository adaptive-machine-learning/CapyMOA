from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
import torch
import torch.nn as nn
import torch.optim as optim


class Autoencoder(AnomalyDetector):
    """Autoencoder anomaly detector

    This is a simple autoencoder anomaly detector that uses a single hidden layer.

    Reference:

    `Contextual One-Class Classification in Data Streams.
    Richard Hugh Moulton, Herna L. Viktor, Nathalie Japkowicz, and João Gama.
    arXiv:1907.04233, 2019.
    <https://arxiv.org/pdf/1907.04233>`_

    Example:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import Autoencoder
    >>> from capymoa.evaluation import AnomalyDetectionEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = Autoencoder(schema=schema)
    >>> evaluator = AnomalyDetectionEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.58

    """

    def __init__(
        self,
        schema=None,
        hidden_layer=2,
        learning_rate=0.5,
        threshold=0.6,
        random_seed=1,
    ):
        """Construct an Autoencoder anomaly detector

        Parameters
        :param schema: The schema of the input data
        :param hidden_layer: Number of neurons in the hidden layer. The number should less than the number of input
        features.
        :param learning_rate: Learning rate
        :param threshold: Anomaly threshold
        :param random_seed: Random seed
        """

        super().__init__(schema, random_seed=random_seed)
        self.hidden_layer = hidden_layer
        self.learning_rate = learning_rate
        self.threshold = threshold

        if self.hidden_layer >= self.schema.get_num_attributes():
            raise ValueError(
                "The number of hidden layer should be less than the number of input features"
            )
        torch.manual_seed(self.random_seed)
        self._initialise()

    def _initialise(self):
        class _AEModel(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(_AEModel, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, hidden_size, dtype=torch.double), nn.Sigmoid()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_size, input_size, dtype=torch.double), nn.Sigmoid()
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        self.model = _AEModel(
            input_size=self.schema.get_num_attributes(), hidden_size=self.hidden_layer
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def __str__(self):
        return "Autoencoder Anomaly Detector"

    def train(self, instance: Instance):
        # Convert the input to a tensor
        input = torch.from_numpy(instance.x)

        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(input)

        # Compute the loss
        loss = self.criterion(output, input)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

    def predict(self, instance: Instance) -> int:
        if self.score_instance(instance) > 0.5:
            return 0
        else:
            return 1

    def score_instance(self, instance: Instance) -> float:
        # Convert the input to a tensor
        input = torch.from_numpy(instance.x)

        # Pass the input through the autoencoder
        output = self.model(input)

        # Compute the reconstruction error
        error = torch.mean(torch.square(input - output))

        return 2.0 ** (-(error.item() / self.threshold))
