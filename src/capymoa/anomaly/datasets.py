"""Collection of built in anomaly detection datasets."""

import numpy as np
from sklearn.datasets import make_blobs

from capymoa.stream import NumpyStream


class TinyBlobs(NumpyStream):
    """A tiny stream for running unit tests for anomaly detection.

    ..  plot::

        import matplotlib.pyplot as plt
        from capymoa.anomaly.datasets import TinyBlobs

        stream = TinyBlobs()
        x, y = stream._x_data, stream._y_data
        plt.scatter(x[y == 0, 0], x[y == 0, 1], marker=".")
        plt.scatter(x[y == 1, 0], x[y == 1, 1], marker="x")

    """

    def __init__(
        self,
        in_samples: int = 1000,
        out_samples: int = 100,
        features: int = 4,
        clusters: int = 3,
        seed: int = 0,
        center_box: tuple[float, float] = (-10.0, 10.0),
        cluster_std: float | list[float] = 1.0,
    ):
        """Construct TinyBlobs.

        :param in_samples: In distribution samples.
        :param out_samples: Out of distribution samples.
        :param features: Number of features.
        :param clusters: Number of clusters.
        :param seed: Random seed for generating data.
        :param center_box: Range features may take.
        :param cluster_std: Variance of each blob center.
        """

        rng = np.random.default_rng(seed)
        in_data, _ = make_blobs(
            n_samples=in_samples,
            n_features=features,
            random_state=seed,
            centers=clusters,
            center_box=center_box,
            cluster_std=cluster_std,
        )
        out_data = rng.uniform(*center_box, size=(out_samples, features))

        # Combine in and out distributions
        data_x = np.vstack([in_data, out_data])
        data_y = np.hstack([np.zeros(in_samples), np.ones(out_samples)])

        # Random permutation of the stream
        idx = rng.permutation(in_samples + out_samples)
        data_x = data_x[idx]
        data_y = data_y[idx]

        super().__init__(
            data_x,
            data_y,
            dataset_name="TinyBlobs",
            target_name="is_anomaly",
            target_type="categorical",
        )
