from __future__ import annotations

import numpy as np

from capymoa.base import (
    Regressor,
)

from capymoa.stream._stream import Schema
from capymoa.classifier._shrubensembles import ShrubEnsembles


class ShrubsRegressor(ShrubEnsembles, Regressor):

    def __init__(self,
                schema: Schema, 
                step_size = 1e-4,
                ensemble_regularizer = "hard-L0",
                l_ensemble_reg = 32,  
                l_tree_reg = 0,
                normalize_weights = False,
                burnin_steps = 0,
                update_leaves = False,
                batch_size = 256,
                additional_tree_options = {
                    "splitter" : "best", 
                    "max_depth": None
                }
        ):
        Regressor.__init__(self, schema, additional_tree_options.get("random_state",0))
        ShrubEnsembles.__init__(self, schema, "mse", step_size, ensemble_regularizer, l_ensemble_reg, l_tree_reg, normalize_weights, burnin_steps, update_leaves, batch_size, additional_tree_options)

    def __str__(self):
       return str("ShrubsRegressor")

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
            return combined_proba