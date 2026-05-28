# SPDX-License-Identifier: BSD-3-Clause
"""_fesl_regressor.py"""
from __future__ import annotations
from typing import Literal
import numpy as np

from openmoa.base import Regressor
from openmoa.stream import Schema
from openmoa.instance import Instance


class FESLRegressor(Regressor):
    """Feature Evolvable Streaming Learning (FESL) Regressor.
    
    This regressor handles scenarios where features evolve over time - old features 
    disappear and new features appear. It assumes an overlapping period where both 
    feature sets are available, learns a mapping between them, and maintains two models
    to improve prediction performance.
    
    Reference:
    
    Hou, B.-J., Zhang, L., & Zhou, Z.-H. (2017). 
    Learning with Feature Evolvable Streams. 
    In Advances in Neural Information Processing Systems 30 (NIPS'17).
    
    Example:

    >>> from openmoa.regressor import FESLRegressor
    >>> FESLRegressor.__name__
    'FESLRegressor'
    """

    def __init__(
        self,
        schema: Schema,
        s1_feature_indices: list[int],
        s2_feature_indices: list[int],
        overlap_size: int = 50,
        switch_point: int = 500,
        ensemble_method: Literal["combination", "selection"] = "combination",
        learning_rate_scale: float = 1.0,
        random_seed: int = 1,
    ):
        """Initialize FESL Regressor.
        
        :param schema: The schema of the stream
        :param s1_feature_indices: Indices of features in the first feature space S1
        :param s2_feature_indices: Indices of features in the second feature space S2
        :param overlap_size: Number of instances in the overlapping period B
        :param switch_point: Instance number where feature space switches from S1 to S2
        :param ensemble_method: "combination" for FESL-c or "selection" for FESL-s
        :param learning_rate_scale: Scale factor for learning rate (tau = 1/(scale * sqrt(t)))
        :param random_seed: Random seed for reproducibility
        """
        # Call parent class __init__
        super().__init__(schema=schema, random_seed=random_seed)
        
        # Set numpy random seed
        np.random.seed(random_seed)
        
        # Feature space configuration
        self.s1_indices = np.array(s1_feature_indices)
        self.s2_indices = np.array(s2_feature_indices)
        self.d1 = len(s1_feature_indices)
        self.d2 = len(s2_feature_indices)
        
        # Temporal configuration
        self.B = overlap_size
        self.T1 = switch_point
        self.overlap_start = self.T1 - self.B
        
        # Algorithm configuration
        self.ensemble_method = ensemble_method
        self.tau_scale = learning_rate_scale
        
        # Model parameters
        self.w1 = np.zeros(self.d1)  # Model for S1
        self.w2 = np.zeros(self.d2)  # Model for S2
        self.M = None  # Mapping matrix from S2 to S1
        
        # Ensemble weights
        self.alpha1 = 0.5
        self.alpha2 = 0.5
        
        # For overlap period: collect data to learn mapping
        self.overlap_X1 = []
        self.overlap_X2 = []
        
        # Instance counter
        self.instance_count = 0
        
        # For FESL-s (selection method)
        self.v1 = 0.5
        self.v2 = 0.5

    def __str__(self):
        return (f"FESLRegressor(s1_dim={self.d1}, s2_dim={self.d2}, "
                f"overlap={self.B}, switch={self.T1}, method={self.ensemble_method})")

    def train(self, instance: Instance):
        """Train the regressor on a single instance.
        
        :param instance: The instance to train on
        """
        self.instance_count += 1
        t = self.instance_count
        
        # Extract features for both spaces
        x_full = np.array(instance.x)
        x_s1 = x_full[self.s1_indices]
        x_s2 = x_full[self.s2_indices]
        
        # Get true target value
        y = instance.y_value
        
        # Stage 1: Only S1 available
        if t <= self.overlap_start:
            self._update_model(self.w1, x_s1, y, t)
        
        # Overlap period: Both S1 and S2 available
        elif t <= self.T1:
            # Collect data for learning mapping
            self.overlap_X1.append(x_s1)
            self.overlap_X2.append(x_s2)
            
            # Continue updating w1
            self._update_model(self.w1, x_s1, y, t)
            
            # At the end of overlap, learn mapping
            if t == self.T1:
                self._learn_mapping()
        
        # Stage 2: Only S2 available
        else:
            t_new = t - self.T1
            
            # Get predictions from both models
            x_s1_recovered = self.M @ x_s2 if self.M is not None else np.zeros(self.d1)
            
            pred1 = np.dot(self.w1, x_s1_recovered)
            pred2 = np.dot(self.w2, x_s2)
            
            # Update both models
            self._update_model(self.w1, x_s1_recovered, y, t_new)
            self._update_model(self.w2, x_s2, y, t_new)
            
            # Update ensemble weights
            loss1 = self._square_loss(pred1, y)
            loss2 = self._square_loss(pred2, y)
            
            T2 = t - self.T1
            self._update_ensemble_weights(loss1, loss2, T2)

    def predict(self, instance: Instance) -> float:
        """Make a prediction on an instance.
        
        :param instance: The instance to predict
        :return: Predicted value
        """
        x_full = np.array(instance.x)
        t = self.instance_count + 1  # Predict is called before train typically
        
        # Stage 1: Use w1 with S1 features
        if t <= self.T1:
            x_s1 = x_full[self.s1_indices]
            pred = np.dot(self.w1, x_s1)
        
        # Stage 2: Ensemble predictions
        else:
            x_s2 = x_full[self.s2_indices]
            
            # Recover S1 features
            x_s1_recovered = self.M @ x_s2 if self.M is not None else np.zeros(self.d1)
            
            pred1 = np.dot(self.w1, x_s1_recovered)
            pred2 = np.dot(self.w2, x_s2)
            
            # Ensemble prediction
            if self.ensemble_method == "combination":
                pred = self.alpha1 * pred1 + self.alpha2 * pred2
            else:  # selection
                pred = pred1 if np.random.random() < self.alpha1 else pred2
        
        return pred

    def _update_model(self, w: np.ndarray, x: np.ndarray, y: float, t: int):
        """Update model weights using online gradient descent.
        
        :param w: Weight vector to update (modified in place)
        :param x: Feature vector
        :param y: True target value
        :param t: Time step for learning rate
        """
        pred = np.dot(w, x)
        tau = self.tau_scale / np.sqrt(t)  # 修正：去掉分母
        
        # Square loss gradient
        gradient = 2.0 * (pred - y) * x
        
        w -= tau * gradient

    def _learn_mapping(self):
        """Learn linear mapping M from S2 to S1 using least squares."""
        if len(self.overlap_X1) == 0:
            self.M = np.zeros((self.d1, self.d2))
            return
        
        X1 = np.array(self.overlap_X1)  # Shape: (B, d1)
        X2 = np.array(self.overlap_X2)  # Shape: (B, d2)
        
        # M = (X2^T X2)^{-1} X2^T X1
        # M^T = X1^T X2 (X2^T X2)^{-1}
        try:
            self.M = np.linalg.lstsq(X2, X1, rcond=None)[0].T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            self.M = np.zeros((self.d1, self.d2))

    def _square_loss(self, pred: float, y: float) -> float:
        """Calculate square loss.
        
        :param pred: Predicted value
        :param y: True target value
        :return: Loss value
        """
        return (y - pred) ** 2

    def _update_ensemble_weights(self, loss1: float, loss2: float, T2: int):
        """Update ensemble weights based on losses.
        
        :param loss1: Loss of model 1
        :param loss2: Loss of model 2
        :param T2: Number of instances in stage 2
        """
        if self.ensemble_method == "combination":
            # FESL-c: Exponential weights
            eta = np.sqrt(8 * np.log(2) / T2) if T2 > 0 else 0.1
            
            w1 = self.alpha1 * np.exp(-eta * loss1)
            w2 = self.alpha2 * np.exp(-eta * loss2)
            
            total = w1 + w2
            if total > 0:
                self.alpha1 = w1 / total
                self.alpha2 = w2 / total
        
        else:  # selection
            # FESL-s: Dynamic selection weights
            eta = np.sqrt(8 / T2 * (2 * np.log(2) + (T2 - 1) * self._H(1 / (T2 - 1)))) if T2 > 1 else 0.1
            delta = 1.0 / (T2 - 1) if T2 > 1 else 0.5
            
            self.v1 = self.alpha1 * np.exp(-eta * loss1)
            self.v2 = self.alpha2 * np.exp(-eta * loss2)
            
            W = self.v1 + self.v2
            self.alpha1 = delta * W / 2 + (1 - delta) * self.v1
            self.alpha2 = delta * W / 2 + (1 - delta) * self.v2
            
            total = self.alpha1 + self.alpha2
            if total > 0:
                self.alpha1 /= total
                self.alpha2 /= total

    @staticmethod
    def _H(x: float) -> float:
        """Binary entropy function.
        
        :param x: Input value in (0, 1)
        :return: Entropy value
        """
        if x <= 0 or x >= 1:
            return 0
        return -x * np.log(x) - (1 - x) * np.log(1 - x)