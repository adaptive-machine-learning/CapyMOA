from __future__ import annotations

import numpy as np

from capymoa.base import (
    Classifier,
)

from capymoa.stream._stream import Schema
from capymoa.classifier._shrubensembles import ShrubEnsembles


class ShrubsClassifier(ShrubEnsembles, Classifier):

    def __init__(self,
                schema: Schema, 
                loss = "ce",
                step_size = 1e-1,
                ensemble_regularizer = "hard-L0",
                l_ensemble_reg = 32,  
                l_tree_reg = 0,
                normalize_weights = False,
                burnin_steps = 0,
                update_leaves = False,
                batch_size = 256,
                additional_tree_options = {
                    "splitter" : "best", 
                    "criterion" : "gini",
                    "max_depth": None
                }
        ):
        Classifier.__init__(self, schema, additional_tree_options.get("random_state",0))
        ShrubEnsembles.__init__(self, schema, loss, step_size, ensemble_regularizer, l_ensemble_reg, l_tree_reg, normalize_weights, burnin_steps, update_leaves, batch_size, additional_tree_options)

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