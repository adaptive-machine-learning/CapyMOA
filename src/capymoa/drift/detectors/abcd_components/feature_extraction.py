from __future__ import annotations

from typing import Protocol, Any, Tuple

import numpy as np
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
