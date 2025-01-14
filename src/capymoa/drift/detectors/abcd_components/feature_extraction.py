from __future__ import annotations

from typing import Protocol, Any, Tuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from sklearn.decomposition import PCA, KernelPCA


class EncoderDecoder(Protocol):
    def update(self, window, epochs: int):
        """
        Update the model
        :param window: the data [n_samples, n_features]
        :param epochs: The number of training epochs
        :return: nothing
        """

    def new_tuple(self, x) -> Tuple[Any, Any, Any]:
        """
        :param x: Input instance
        :return: A new tuple containing, MSE, reconstruction, and original
        """


class DummyEncoderDecoder(EncoderDecoder):
    def update(self, window, epochs: int):
        pass

    def new_tuple(self, x) -> Tuple[Any, Any, Any]:
        return 0.0, x, x


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


class PCAModel(EncoderDecoder):
    def __init__(self, input_size: int, eta: float):
        self.input_size = input_size
        self.eta = eta
        self.components = int(input_size * eta)

    def update(self, window, epochs: int):
        # n_components must be between 0 and min(n_samples, n_features) with svd_solver='full'
        max_components = min(window.shape)
        components = min(self.components, max_components)
        self.pca = PCA(n_components=components, svd_solver="full")
        self.pca.fit(window)

    def new_tuple(self, x) -> Tuple[Any, Any, Any]:
        assert len(x.shape) == 2
        enc = self.pca.transform(x)
        dec = self.pca.inverse_transform(enc)
        se = (dec - x) ** 2
        mse = np.mean(se)
        return mse, dec.flatten(), x.flatten()


class KernelPCAModel(EncoderDecoder):
    def __init__(self, input_size: int, eta: float, kernel="rbf"):
        self.input_size = input_size
        self.eta = eta
        self.kernel = kernel
        self.components = int(input_size * eta)

    def update(self, window, epochs: int):
        self.pca = KernelPCA(
            n_components=self.components, kernel=self.kernel, fit_inverse_transform=True
        )
        self.pca.fit(window)

    def new_tuple(self, x) -> Tuple[Any, Any, Any]:
        assert len(x.shape) == 2
        enc = self.pca.transform(x)
        dec = self.pca.inverse_transform(enc)
        se = (dec - x) ** 2
        mse = np.mean(se)
        return mse, dec.flatten(), x.flatten()
