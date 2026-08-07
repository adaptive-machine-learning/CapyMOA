"""The PyTorch-backed encoder-decoder for :class:`~capymoa.drift.detectors.ABCD`.

Kept apart from :mod:`.feature_extraction` so that importing ABCD does not import
PyTorch. ``AutoEncoder`` subclasses :class:`torch.nn.Module`, so the import cannot
be deferred into a method -- the class statement itself needs it -- which is why
this lives in its own module and is loaded only when ``model_id="ae"``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from .feature_extraction import EncoderDecoder


class AutoEncoder(nn.Module, EncoderDecoder):
    def __init__(self, input_size: int, eta: float):
        """
        A simple single layer autoencoder
        :param input_size: The size of the input
        :param eta: The encoding factor. Hidden layer size is eta * input_size
        """
        super(AutoEncoder, self).__init__()
        self.eta = eta
        self.input_size = input_size
        self.bottleneck_size = int(eta * input_size)
        self.encoder = nn.Linear(
            in_features=self.input_size, out_features=self.bottleneck_size
        )
        self.decoder = nn.Linear(
            in_features=self.bottleneck_size, out_features=self.input_size
        )
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x

    def update(self, window, epochs: int = 1):
        """
        Update the autoencoder on the given window
        :param window: The data
        :param epochs: The number of training epochs
        :param logger: If a logger is provided, log the reconstruction loss during training
        :return:
        """
        if len(window) == 0:
            return
        self.train()
        tensor = torch.from_numpy(window).float()
        for ep in range(epochs):
            self.optimizer.zero_grad()
            pred = self.forward(tensor)
            loss = F.mse_loss(pred, tensor)
            loss.backward()
            self.optimizer.step()

    def new_tuple(self, x):
        """
        :param x: Input instance
        :return: A new tuple containing, MSE, reconstruction, and original
        """
        tensor = torch.from_numpy(x).float()
        self.eval()
        with torch.no_grad():
            pred = self.forward(tensor)
            loss = F.mse_loss(pred, tensor)
            return loss.item(), pred.numpy()[0], x[0]
