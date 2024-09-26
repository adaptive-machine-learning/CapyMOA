from __future__ import annotations
from typing import Literal

import numpy as np

from capymoa.base import (
    Regressor,
)

from capymoa.stream._stream import Schema
from capymoa.classifier._shrubs_ensemble import ShrubEnsembles


class ShrubsRegressor(ShrubEnsembles, Regressor):
    """ShrubsRegressor

    This class implements the ShrubEnsembles algorithm for regression, which is
    an ensemble classifier that continously adds decision trees to the ensemble by training 
    decision trees over a sliding window while pruning unnecessary drees away using proximal (stoachstic) gradient descent, 
    hence allowing for adaptation to concept drift. For regression, the MSE loss is minimized.

    Reference:
    
    `Shrub Ensembles for Online Classification
    Sebastian Buschjäger, Sibylle Hess, and Katharina Morik
    In Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22), Jan 2022.
    <https://aaai.org/papers/06123-shrub-ensembles-for-online-classification/>`_

    Example usage:

    >>> from capymoa.datasets import Fried
    >>> from capymoa.regressor import ShrubsRegressor
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = Fried()
    >>> schema = stream.get_schema()
    >>> learner = ShrubsRegressor(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].rmse()
    4.3554695106618
    """
    def __init__(self,
        schema: Schema, 
        step_size: float|Literal["adaptive"] = 1e-1,
        ensemble_regularizer: Literal["hard-L0","L0","L1","none"] = "hard-L0",
        l_ensemble_reg: float|int = 16, 
        l_l2_reg: float = 0,
        l_tree_reg: float = 0,
        normalize_weights: bool = True,
        burnin_steps: int = 0,
        update_leaves: bool = False,
        batch_size: int = 32,
        additional_tree_options: dict = {
            "splitter" : "best", 
            "max_depth": None,
            "random_state": 1234
        }
    ):
        """ Initializes the ShrubEnsemble regressor with the given parameters.

        :param step_size: float|Literal["adaptive"] - The step size (i.e. learning rate of SGD) for updating the model. Can be a float or "adaptive". Adaptive reduces the step size with more estimators, i.e. sets it to 1.0 / (n_estimators + 1.0)
        :param ensemble_regularizer: Literal["hard-L0","L0","L1","none"] - The regularizer for the weights of the ensemble. Supported values are "none", "L0", "L1", and "hard-L0". Hard-L0 refers to L0 regularization via the prox-operator, whereas L0 and L1 refer to L0/L1 regularization via projection. Projection can be viewed as a softer regularization that drives the weights of each member towards 0, whereas hard-l0 limits the number of trees in the entire ensemble. 
        :param l_ensemble_reg: float | int - The regularization strength. If `ensemble_regularizer = hard-L0`, then this parameter represent the total number of trees in the ensembles. If `ensemble_regularizer = L0` or `ensemble_regularizer = L1`, then this parameter is the regularization strength. This these cases the number of trees grow over time and only trees that do not contribute to the ensemble will be removed.
        :param l_l2_reg: float - The L2 regularization strength of the weights of each tree. 
        :param l_tree_reg: float - The regularization parameter for individual trees. Must be greater than or equal to 0. `l_tree_reg` controls the number of (overly) large trees in the ensemble by punishing the weights of each tree. Formally, the number of nodes of each tree is used as an additional regularizer. 
        :param normalize_weights: bool - Whether to normalize the weights of the ensemble, i.e. the weight sum to 1.
        :param burnin_steps: int - The number of burn-in steps before updating the model, i.e. the number of SGD steps to be take per each call of train
        :param update_leaves: bool - Whether to update the leaves of the trees as well using SGD.
        :param batch_size: int - The batch size for training each individual tree. Internally, a sliding window is stored. Must be greater than or equal to 1. 
        :param additional_tree_options: dict - Additional options for the trees, such as splitter, criterion, and max_depth. See sklearn.tree.DecisionTreeRegressor for details. An example would be additional_tree_options = {"splitter": "best", "max_depth": None}
        
        """
        Regressor.__init__(self, schema, additional_tree_options.get("random_state",0))
        ShrubEnsembles.__init__(self, schema, "mse", step_size, ensemble_regularizer, l_ensemble_reg, l_l2_reg,  l_tree_reg, normalize_weights, burnin_steps, update_leaves, batch_size, additional_tree_options)

    def __str__(self):
       return str("ShrubsRegressor")

    def _individual_proba(self, X):
        if len(X.shape) < 2:
            all_proba = np.zeros(shape=(len(self.estimators_), 1, 1), dtype=np.float32)
        else:
            all_proba = np.zeros(shape=(len(self.estimators_), X.shape[0], 1), dtype=np.float32)

        for i, e in enumerate(self.estimators_):
            all_proba[i, :, 0] += e.predict(X)

        return all_proba

    def predict(self, instance):
        if (len(self.estimators_)) == 0:
            # TODO Maybe add a more meanigful default here
            return 0
        else:
            all_proba = self._individual_proba(np.array([instance.x]))
            scaled_prob = sum([w * p for w,p in zip(all_proba, self.estimator_weights_)])
            combined_proba = np.sum(scaled_prob, axis=0)
            # combined_proba should be a (1,) array, but the remaining CapyMoa code expects scalars
            return combined_proba.item()