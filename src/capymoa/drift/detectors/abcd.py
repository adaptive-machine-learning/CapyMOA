from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .abcd_components.feature_extraction import (
    AutoEncoder,
    EncoderDecoder,
    PCAModel,
    KernelPCAModel,
)
from .abcd_components.windowing import AdaptiveWindow, p_bernstein
from capymoa.drift.base_detector import BaseDriftDetector
from ...instance import Instance


class ABCD(BaseDriftDetector):
    def __init__(
        self,
        delta_drift: float = 0.002,
        delta_warn: float = 0.01,
        model_id: str = "ae",
        split_type: str = "ed",
        encoding_factor: float = 0.5,
        update_epochs: int = 50,
        num_splits: int = 20,
        max_size: int = np.inf,
        subspace_threshold: float = 2.5,
        n_min: int = 100,
        maximum_absolute_value: float = 1.0,
        bonferroni: bool = False,
    ):
        """
        :param delta_drift: The desired confidence level at which a drift is detected
        :param delta_warn: The desired confidence level at which a warning is detected
        :param model_id: The name of the model to use
        :param update_epochs: The number of epochs to train the AE after a change occurred
        :param split_type: Investigation of different split types
        :param subspace_threshold: Called tau in the paper
        :param bonferroni: Use bonferroni correction to account for multiple testing?
        :param encoding_factor: The relative size of the bottleneck
        :param maximum_absolute_value: The maximum absolute value that one can expect (e.g. 1.0 for normalized data). Smaller values can increase false alarms but speed up change detection
        :param num_splits: The number of time point to evaluate
        """
        self.split_type = split_type
        self.delta_drift = delta_drift
        self.delta_warn = delta_warn
        self.bonferroni = bonferroni
        self.num_splits = num_splits
        self.max_size = max_size
        self.subspace_threshold = subspace_threshold
        self.n_min = n_min
        self.maximum_absolute_value = maximum_absolute_value
        self.window = AdaptiveWindow(
            delta_drift=delta_drift,
            delta_warn=delta_warn,
            split_type=split_type,
            max_size=max_size,
            bonferroni=bonferroni,
            n_splits=num_splits,
            abs_max=maximum_absolute_value,
        )
        self.model: EncoderDecoder = None
        self.last_change_point = None
        self.last_detection_point = None
        self.last_training_point = None
        self._last_loss = np.nan
        self.drift_dimensions = None
        self.epochs = update_epochs
        self.eta = encoding_factor
        self._severity = np.nan
        self.model_id = model_id
        self._new_data = None
        self.delay = 0
        if model_id == "pca":
            self.model_class = PCAModel
        elif model_id == "kpca":
            self.model_class = KernelPCAModel
        elif model_id == "ae":
            self.model_class = AutoEncoder
        else:
            raise ValueError
        super(ABCD, self).__init__()

    def get_params(self) -> Dict[str, Any]:
        """Get the hyper-parameters of the drift detector."""
        return {
            "delta_drift": self.delta_drift,
            "delta_warn": self.delta_warn,
            "model_id": self.model_id,
            "split_type": self.split_type,
            "encoding_factor": self.eta,
            "update_epochs": self.epochs,
            "num_splits": self.num_splits,
            "max_size": self.max_size,
            "subspace_threshold": self.subspace_threshold,
            "n_min": self.n_min,
            "maximum_absolute_value": self.maximum_absolute_value,
            "bonferroni": self.bonferroni,
        }

    def pre_train(self, data):
        if self.model is None:
            self.model = self.model_class(input_size=data.shape[-1], eta=self.eta)
        self.model.update(data, epochs=self.epochs)

    def add_element(self, element: float | np.ndarray | Instance) -> None:
        """
        Add the new element and also perform change detection
        :param element: The new observation
        :return:
        """
        # transform input into 2d numpy array
        n_dims = 1
        if isinstance(element, Instance):
            n_dims = len(element.x)
            this_element = np.array([element.x])
        elif isinstance(element, np.ndarray):
            n_dims = len(element)
            this_element = np.array([element])
        else:
            this_element = np.array([[element]])
        self.idx += 1
        self.in_concept_change = False
        if n_dims > 1:
            if self.model is None:
                if self._new_data is None:
                    self._new_data = np.array(this_element)
                if len(self._new_data) < self.n_min:
                    self._new_data = np.append(self._new_data, this_element, axis=0)
                else:
                    self.pre_train(self._new_data)
                    self._new_data = None
                return
            new_tuple = self.model.new_tuple(
                this_element
            )  # A new tuple containing, MSE, reconstruction, and original
        else:
            new_tuple = (
                this_element[0, 0],
                np.zeros_like(this_element),
                this_element,
            )  # In the 1d case, we don't need an encoder-decoder model, we simply monitor the input
        self.window.grow(new_tuple)  # add new tuple to window
        self._last_loss = self.window.most_recent_loss()
        self.in_concept_change, self.in_warning_zone, detection_point = (
            self.window.has_change()
        )

        if self.in_warning_zone:
            self.warning_index.append(self.idx)

        if self.in_concept_change:
            self.detection_index.append(self.idx)

        if self.in_concept_change:
            self._evaluate_subspace()
            self._evaluate_magnitude()

            # EXPERIMENT LOGGING
            self.last_detection_point = detection_point
            self.delay = len(self.window) - self.window.t_star
            self.last_change_point = self.last_detection_point - self.delay

            self.model = None  # Remove outdated model
            self._new_data = self.window.data_new()
            self.window.reset()  # forget outdated data

    def loss(self):
        return self._last_loss

    def get_dims_p_values(self) -> np.ndarray:
        return self.drift_dimensions

    def get_drift_dims(self) -> np.ndarray:
        drift_dims = np.array(
            [
                i
                for i in range(len(self.drift_dimensions))
                if self.drift_dimensions[i] < self.subspace_threshold
            ]
        )
        return (
            np.arange(len(self.drift_dimensions))
            if len(drift_dims) == 0
            else drift_dims
        )

    def get_severity(self):
        return self._severity

    def _evaluate_subspace(self):
        data = self.window.data()
        output = self.window.reconstructions()
        error = output - data
        squared_errors = np.power(error, 2)
        window1 = squared_errors[: self.window.t_star]
        window2 = squared_errors[self.window.t_star :]
        mean1 = np.mean(window1, axis=0)
        mean2 = np.mean(window2, axis=0)
        eps = np.abs(mean2 - mean1)
        sigma1 = np.std(window1, axis=0)
        sigma2 = np.std(window2, axis=0)
        n1 = len(window1)
        n2 = len(window2)
        p = p_bernstein(
            eps,
            n1=n1,
            n2=n2,
            sigma1=sigma1,
            sigma2=sigma2,
            abs_max=self.maximum_absolute_value,
        )
        self.drift_dimensions = p if isinstance(p, np.ndarray) else np.array([p])

    def _evaluate_magnitude(self):
        drift_point = self.window.t_star
        data = self.window.data()
        recons = self.window.reconstructions()
        drift_dims = self.get_drift_dims()
        if len(drift_dims) == 0:
            drift_dims = np.arange(data.shape[-1])
        input_pre = data[:drift_point, drift_dims]
        input_post = data[drift_point:, drift_dims]
        output_pre = recons[:drift_point, drift_dims]
        output_post = recons[drift_point:, drift_dims]
        se_pre = (input_pre - output_pre) ** 2
        se_post = (input_post - output_post) ** 2
        mse_pre = np.mean(se_pre, axis=-1)
        mse_post = np.mean(se_post, axis=-1)
        mean_pre, std_pre = np.mean(mse_pre), np.std(mse_pre)
        mean_post = np.mean(mse_post)
        z_score_normalized = np.abs(mean_post - mean_pre) / std_pre
        self._severity = float(z_score_normalized)
