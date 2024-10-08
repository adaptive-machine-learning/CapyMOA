from __future__ import annotations
from typing import Literal

import numpy as np

from capymoa.base import Classifier
from capymoa.stream._stream import Schema
from capymoa.classifier._shrubs_ensemble import _ShrubEnsembles
from sklearn.tree import DecisionTreeClassifier


class ShrubsClassifier(_ShrubEnsembles, Classifier):
    """ShrubsClassifier

    This class implements the ShrubEnsembles algorithm for classification, which is
    an ensemble classifier that continuously adds decision trees to the ensemble by training new trees over a sliding window while pruning unnecessary trees away using proximal (stochastic) gradient descent, hence allowing for adaptation to concept drift.

    Reference:
    
    `Shrub Ensembles for Online Classification
    Sebastian Buschjäger, Sibylle Hess, and Katharina Morik
    In Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22), Jan 2022.
    <https://aaai.org/papers/06123-shrub-ensembles-for-online-classification/>`_

    Example usage:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import ShrubsClassifier
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = ShrubsClassifier(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    85.5...
    
    """

    def __init__(
        self,
        schema: Schema,
        loss: Literal["mse", "ce", "h2"] = "ce",
        step_size: float | Literal["adaptive"] = "adaptive",
        ensemble_regularizer: Literal["hard-L0", "L0", "L1", "none"] = "hard-L0",
        l_ensemble_reg: float | int = 32,
        l_l2_reg: float = 0,
        l_tree_reg: float = 0,
        normalize_weights: bool = True,
        burnin_steps: int = 5,
        update_leaves: bool = False,
        batch_size: int = 32,
        sk_dt: DecisionTreeClassifier = DecisionTreeClassifier(
            splitter="best", criterion="gini", max_depth=None, random_state=1234
        ),
    ):

        """Initializes the ShrubEnsemble classifier with the given parameters.

        :param loss: The loss function to be used. Supported values are ``"mse"``,
            ``"ce"``, and ``"h2"``.
        :param step_size: The step size (i.e. learning rate of SGD) for updating
            the model. Can be a float or "adaptive". Adaptive reduces the step
            size with more estimators, i.e. sets it to ``1.0 / (n_estimators +
            1.0)``
        :param ensemble_regularizer: The regularizer for the weights of the
            ensemble. Supported values are:

            * ``hard-L0``: L0 regularization via the prox-operator.
            * ``L0``: L0 regularization via projection.
            * ``L1``: L1 regularization via projection.
            * ``none``: No regularization.
            
            Projection can be viewed as a softer regularization that drives the
            weights of each member towards 0, whereas ``hard-l0`` limits the
            number of trees in the entire ensemble.
        :param l_ensemble_reg: The regularization strength. Depending on the
            value of ``ensemble_regularizer``, this parameter has different
            meanings:
        
            * ``hard-L0``: then this parameter represent the total number of
              trees in the ensembles.
            * ``L0`` or ``L1``: then this parameter is the regularization
              strength. In these cases the number of trees grow over time and
              only trees that do not contribute to the ensemble will be
              removed.
            * ``none``: then this parameter is ignored.
        :param l_l2_reg: The L2 regularization strength of the weights of each
            tree. 
        :param l_tree_reg: The regularization parameter for individual trees.
            Must be greater than or equal to 0. ``l_tree_reg`` controls the
            number of (overly) large trees in the ensemble by punishing the
            weights of each tree. Formally, the number of nodes of each tree is
            used as an additional regularizer.
        :param normalize_weights: Whether to normalize the weights of the
            ensemble, i.e. the weight sum to 1.
        :param burnin_steps: The number of burn-in steps before updating the
            model, i.e. the number of SGD steps to be take per each call of
            train
        :param update_leaves: Whether to update the leaves of the trees as well
            using SGD.
        :param batch_size: The batch size for training each individual tree.
            Internally, a sliding window is stored. Must be greater than or
            equal to 1. 
        :param sk_dt: Base object which is used to clone any new decision trees
            from. Note, that if you set random_state to an integer the exact
            same clone is used for any DT object 
        """

        Classifier.__init__(self, schema, sk_dt.random_state)
        _ShrubEnsembles.__init__(self, schema, loss, step_size, ensemble_regularizer, l_ensemble_reg, l_l2_reg, l_tree_reg, normalize_weights, burnin_steps, update_leaves, batch_size, sk_dt)

    def __str__(self):
        return str("ShrubsClassifier")

    def _individual_proba(self, X):
        # assert self.estimators_ is not None, "Call fit before calling predict_proba!"

        if len(X.shape) < 2:
            all_proba = np.zeros(shape=(len(self.estimators_), 1, self.n_classes_), dtype=np.float32)
        else:
            all_proba = np.zeros(shape=(len(self.estimators_), X.shape[0], self.n_classes_), dtype=np.float32)

        for i, e in enumerate(self.estimators_):
            if len(X.shape) < 2:
                all_proba[i, 1, e.classes_.astype(int)] += e.predict_proba(X[np.newaxis,:])
            else:
                proba = e.predict_proba(X)
                # Numpy seems to do some weird stuff when it comes to advanced indexing.
                # Basically, due to e.classes_.astype(int) the last and second-to-last dimensions of all_proba
                # are swapped when doing all_proba[i, :, e.classes_.astype(int)]. Hence, we would also need to swap
                # the shapes of proba to match this correctly. Alternatively, we use a simpler form of indexing as below.
                # Both should work fine
                # all_proba[i, :, e.classes_.astype(int)] += proba.T
                all_proba[i, :, :][:, e.classes_.astype(int)] += proba

        return all_proba

    def predict_proba(self, instance):
        if (len(self.estimators_)) == 0:
            return 1.0 / self.n_classes_ * np.ones(self.n_classes_)
        else:
            all_proba = self._individual_proba(np.array([instance.x]))
            scaled_prob = sum([w * p for w,p in zip(all_proba, self.estimator_weights_)])
            combined_proba = np.sum(scaled_prob, axis=0)
            return combined_proba

    def predict(self, instance):
        # Return the index of the class with the highest probability
        return self.predict_proba(instance).argmax(axis=0)
