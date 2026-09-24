# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using PyTorch with CapyMOA
# * This notebook demonstrates how use PyTorch with CapyMOA.
# * It contains examples showing:
#     * How to define a PyTorch Network to be used with CapyMOA.
#     * How a simple PyTorch model can be used in a CapyMOA `Instance` loop.
#     * How to define a PyTorch CapyMOA Classifier based on CapyMOA `Classifier` framework and how to use it with `prequential_evaluation()`.
#     * How to use a PyTorch dataset with a CapyMOA classifier.
# * `Exploring Advanced Features` (notebooks/common/advanced_API.py) includes an example using TensorBoard and a `PyTorchClassifier`.
#  
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 28/11/2025**

# %% [markdown]
# ## Setup
# * Sets random seed for reproducibility.
# * Sets PyTorch network .

# %% [markdown] jupyter={"outputs_hidden": false}
# ### Set random seeds

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, mock_datasets

if is_nb_fast():
    mock_datasets()

# %% jupyter={"outputs_hidden": false}
import random

random_seed = 1
random.seed(random_seed)

# %% [markdown] jupyter={"outputs_hidden": false}
# ### Define network structure
# * Here, the network uses the CPU device.

# %%
import torch
from torch import nn

torch.manual_seed(random_seed)
torch.use_deterministic_algorithms(True)

# Get cpu device for training.
device = "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=0, number_of_classes=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, number_of_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# %% [markdown] jupyter={"outputs_hidden": false}
# ### Using an instance loop
# * Model is initialised after receiving the first instance.

# %%
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import ClassificationEvaluator

elec_stream = ElectricityTiny()

# Creating the evaluator
evaluator = ClassificationEvaluator(schema=elec_stream.get_schema())

model = None
optimiser = None
loss_fn = nn.CrossEntropyLoss()

i = 0
while elec_stream.has_more_instances():
    i += 1
    instance = elec_stream.next_instance()
    if model is None:
        moa_instance = instance.java_instance.getData()
        # initialise the model and send it to the device
        model = NeuralNetwork(
            input_size=elec_stream.get_schema().get_num_attributes(),
            number_of_classes=elec_stream.get_schema().get_num_classes(),
        ).to(device)
        # set the optimiser
        optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)
        print(model)

    X = torch.tensor(instance.x, dtype=torch.float32)
    y = torch.tensor(instance.y_index, dtype=torch.long)
    # set the device and add a dimension to the tensor
    X, y = torch.unsqueeze(X.to(device), 0), torch.unsqueeze(y.to(device), 0)

    # turn off gradient collection for test
    with torch.no_grad():
        pred = model(X)
        prediction = torch.argmax(pred)

    # update evaluator with predicted class
    evaluator.update(instance.y_index, prediction.item())

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()

    if i % 500 == 0:
        print(f"Accuracy at {i} : {evaluator.accuracy()}")

print(f"Accuracy at {i} : {evaluator.accuracy()}")

# %% [markdown]
# ## PyTorchClassifier
# * Defining a `PyTorchClassifier` using the CapyMOA API makes it **compatible** with CapyMOA functions like `prequential_evaluation()` without losing the **flexibility** of specifying the `architecture` and the `training` method.
# * The model is initialised after receiving the first instance.
# * `PyTorchClassifier` is based on the `capymoa.base` `Classifier` abstract class.
#
# * **Important**: We can access information about the stream through any of its instances. See `set_model(self, instance)` for an example: 
#
# ```python
# ...
# moa_instance = instance.java_instance.getData()
# self.model = NeuralNetwork(input_size=moa_instance.get_num_attributes(), 
#                            number_of_classes=moa_instance.get_num_classes()).to(self.device)
# ...
# ```

# %%
import numpy as np

from capymoa.base import Classifier


class PyTorchClassifier(Classifier):
    def __init__(
        self,
        schema=None,
        random_seed=1,
        nn_model: nn.Module = None,
        optimiser=None,
        loss_fn=None,
        device=("cpu"),
        lr=1e-3,
    ):
        super().__init__(schema, random_seed)
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        self.model = None
        self.optimiser = None
        self.loss_fn = loss_fn
        self.lr = lr
        self.device = device

        torch.manual_seed(random_seed)

        if nn_model is None:
            self.set_model(None)
        else:
            self.model = nn_model.to(device)
        if optimiser is None:
            if self.model is not None:
                self.optimiser = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            self.optimiser = optimiser

    def __str__(self):
        return str(self.model)

    def cli_help(self):
        return 'schema=None, random_seed=1, nn_model: nn.Module=None, optimiser=None, loss_fn=nn.CrossEntropyLoss(), device=("cpu"), lr=1e-3'

    def set_model(self, instance):
        if self.schema is None:
            moa_instance = instance.java_instance.getData()
            self.model = NeuralNetwork(
                input_size=moa_instance.get_num_attributes(),
                number_of_classes=moa_instance.get_num_classes(),
            ).to(self.device)
        elif instance is not None:
            self.model = NeuralNetwork(
                input_size=self.schema.get_num_attributes(),
                number_of_classes=self.schema.get_num_classes(),
            ).to(self.device)

    def train(self, instance):
        if self.model is None:
            self.set_model(instance)

        X = torch.tensor(instance.x, dtype=torch.float32)
        y = torch.tensor(instance.y_index, dtype=torch.long)
        # set the device and add a dimension to the tensor
        X, y = (
            torch.unsqueeze(X.to(self.device), 0),
            torch.unsqueeze(y.to(self.device), 0),
        )

        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        self.optimiser.step()
        self.optimiser.zero_grad()

    def predict(self, instance):
        return np.argmax(self.predict_proba(instance))

    def predict_proba(self, instance):
        if self.model is None:
            self.set_model(instance)
        X = torch.unsqueeze(
            torch.tensor(instance.x, dtype=torch.float32).to(self.device), 0
        )
        # turn off gradient collection
        with torch.no_grad():
            pred = np.asarray(self.model(X).numpy(), dtype=np.double)
        return pred


# %% [markdown] jupyter={"outputs_hidden": false}
# ### Using PyTorchClassifier + prequential_evaluation
#
# * We can access information about the stream through the `schema` directly, from the example below: 
# ```python
# ...
# nn_model=NeuralNetwork(input_size=elec_stream.get_schema().get_num_attributes(),
#                        number_of_classes=elec_stream.get_schema().get_num_classes()).to(device)
# ...
# ```

# %%
from capymoa.evaluation import prequential_evaluation

## Opening a file as a stream
elec_stream = ElectricityTiny()

# Creating a learner
simple_pyTorch_classifier = PyTorchClassifier(
    schema=elec_stream.get_schema(),
    nn_model=NeuralNetwork(
        input_size=elec_stream.get_schema().get_num_attributes(),
        number_of_classes=elec_stream.get_schema().get_num_classes(),
    ).to(device),
)

evaluator = prequential_evaluation(
    stream=elec_stream,
    learner=simple_pyTorch_classifier,
    window_size=4500,
    optimise=False,
)

print(f"Accuracy: {evaluator.cumulative.accuracy()}")

# %% [markdown] jupyter={"outputs_hidden": false}
# ## How to use a PyTorch dataset with a CapyMOA classifier
# * One may want to use various PyTorch datasets with different CapyMOA classifiers.
# * In this example we use PyTorch Dataset + prequential evaluation + CapyMOA Classifier.
#
# **Observation**: *Using a learner like Online Bagging without any feature extraction is not going to yield meaningful performance*

# %%
from torchvision import datasets
from torchvision.transforms import ToTensor

from capymoa.classifier import OnlineBagging
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.visualization import plot_windowed_results
from capymoa.stream import TorchStream

pytorch_dataset = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
pytorch_stream = TorchStream.from_classification(
    dataset=pytorch_dataset, num_classes=10
)

# Creating a learner
ob_learner = OnlineBagging(schema=pytorch_stream.get_schema(), ensemble_size=5)

results_ob_learner = prequential_evaluation(
    stream=pytorch_stream, learner=ob_learner, window_size=100, max_instances=1000
)

print(f"Accuracy: {results_ob_learner.cumulative.accuracy()}")
display(results_ob_learner.windowed.metrics_per_window())
plot_windowed_results(results_ob_learner, metric="accuracy")
