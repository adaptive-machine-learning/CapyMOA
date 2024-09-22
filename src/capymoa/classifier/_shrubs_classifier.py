from __future__ import annotations

import numpy as np

from capymoa.base import (
    Classifier,
)

from capymoa.stream._stream import Schema
from capymoa.classifier._shrubensembles import ShrubEnsembles


class ShrubsClassifier(ShrubEnsembles, Classifier):
    """ShrubsClassifier

    This class implements the ShrubEnsembles algorithm for classification, which is
    an ensemble classifier that continously adds decision trees to the ensemble by training 
    decision trees over a sliding window while pruning unnecessary drees away using proximal (stoachstic) gradient descent, 
    hence allowing for adaptation to concept drift.

    Reference:

    `Shrub Ensembles for Online Classification
     Sebastian Buschjäger, Sibylle Hess, and Katharina Morik
     In Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22), Jan 2022 
    <https://aaai.org/papers/06123-shrub-ensembles-for-online-classification/>`_

    See also :py:class:`capymoa.regressor.AdaptiveRandomForestRegressor`
    See :py:class:`capymoa.base.MOAClassifier` for train, predict and predict_proba.

    Example usage:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import ShrubsClassifier
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = ShrubsClassifier(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    """
    def __init__(self,
                schema: Schema, 
                loss = "ce",
                step_size = 1e-1,
                ensemble_regularizer = "hard-L0",
                l_ensemble_reg = 32,  
                l_l2_reg = 0,
                l_tree_reg = 0,
                normalize_weights = True,
                burnin_steps = 0,
                update_leaves = False,
                batch_size = 32,
                additional_tree_options = {
                    "splitter" : "best", 
                    "criterion" : "gini",
                    "max_depth": None
                }
        ):

        """
        Initializes the ShrubsClassifier classifier with the given parameters.
        Parameters:
        -----------
        schema : Schema
            The schema of the dataset, containing information about attributes and target.
        loss : str, optional (default="ce")
            The loss function to be used. Supported values are "mse", "ce", and "h2".
        step_size : float or str, optional (default=1e-1)
            The step size (i.e. learning rate of SGD) for updating the model. Can be a float or "adaptive".
        ensemble_regularizer : str, optional (default="hard-L0")
            The regularizer for the weights of the ensemble. Supported values are "none", "L0", "L1", and "hard-L0". Hard-L0 refer to L0 regularization via the prox-operator, whereas L0 and L1 refer to L0/L1 regularization via projection. Projection can be viewed as a softer regularization that drives the weights of each member towards 0, whereas hard-l0 limits the number of trees in the entire ensemble. 
        l_ensemble_reg : int or float, optional (default=32)
            The regularization strength. If `ensemble_regularizer = hard-L0`, then this parameter represent the total number of trees in the ensembles. If `ensemble_regularizer = L0` or `ensemble_regularizer = L1`, then this parameter is the regularization strength. This these cases the number of trees grow over time and only trees that do not contribute to the ensemble will be removed.
        l_l2_reg: float, optional (default=0)
            The L2 regularization strength of the weights of each tree. 
        l_tree_reg : float, optional (default=0)
            The regularization parameter for individual trees. Must be greater than or equal to 0. `l_tree_reg` controls the number of (overly) large trees in the ensemble by punishing the weights of each tree. Formally, the number of nodes of each tree is used as an additional regularizer. 
        normalize_weights : bool, optional (default=False)
            Whether to normalize the weights of the ensemble, i.e. the weight sum to 1.
        burnin_steps : int, optional (default=0)
            The number of burn-in steps before updating the model, i.e. the number of SGD steps to be take per each call of train
        update_leaves : bool, optional (default=False)
            Whether to update the leaves of the trees as well using SGD.
        batch_size : int, optional (default=256)
            The batch size for training each individual tree. Internally, a sliding window is stored. Must be greater than or equal to 1. 
        additional_tree_options : dict, optional (default={"splitter": "best", "criterion": "gini", "max_depth": None})
            Additional options for the trees, such as splitter, criterion, and max_depth. See sklearn.tree.DecisionTreeClassifier 
        Raises:
        -------
        ValueError
            If an unsupported value is provided for loss or ensemble_regularizer.
            If l_tree_reg is less than 0.
        """
        Classifier.__init__(self, schema, additional_tree_options.get("random_state",0))
        ShrubEnsembles.__init__(self, schema, loss, step_size, ensemble_regularizer, l_ensemble_reg, l_l2_reg, l_tree_reg, normalize_weights, burnin_steps, update_leaves, batch_size, additional_tree_options)

    def __str__(self):
       return str("ShrubsClassifier")
    
    def _individual_proba(self, X):
        ''' Predict class probabilities for each individual learner in the ensemble without considering the weights.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        y : array, shape (n_samples,C)
            The predicted class probabilities for each learner.
        '''
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
                # the shapes of proba to match this correctly. Alternativley, we use a simpler form of indexing as below. 
                # Both should work fine
                #all_proba[i, :, e.classes_.astype(int)] += proba.T
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
        # TODO check if index or actual class should be returned
        return self.predict_proba(instance).argmax(axis=0)
        # return self.classes_.take(proba.argmax(axis=1), axis=0)