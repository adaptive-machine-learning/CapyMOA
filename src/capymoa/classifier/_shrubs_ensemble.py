from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from sklearn import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.special import softmax

from capymoa.stream._stream import Schema


def to_prob_simplex(x: list | np.array):
    """
    Projects a given vector `x` onto the probability simplex.
    The function takes a vector `x` and projects it onto the probability simplex,
    ensuring that the resulting vector sums to 1 and all elements are non-negative.
    If `x` is None or empty, it returns `x` as is.

    :param x (list or numpy array): The input vector to be projected.
    :return: The projected vector on the probability simplex.
    :rtype: list

    Reference:
    `Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application
    Weiran Wang, Miguel Á. Carreira-Perpiñán 2013
    https://arxiv.org/abs/1309.1541`
    """
    if x is None or len(x) == 0:
        return x
    u = np.sort(x)[::-1]

    l = None  # noqa
    u_sum = 0
    for i in range(0, len(u)):
        u_sum += u[i]
        tmp = 1.0 / (i + 1.0) * (1.0 - u_sum)
        if u[i] + tmp > 0:
            l = tmp  # noqa

    projected_x = [max(xi + l, 0.0) for xi in x]
    return projected_x


class _ShrubEnsembles(ABC):
    """Base for shrub ensembles for classification and regression.

    This class implements the ShrubEnsembles algorithm for classification and regression, which is
    an ensemble classifier that continuously adds decision trees to the ensemble by training
    new trees over a sliding window while pruning unnecessary trees away using proximal (stochastic) gradient descent,
    hence allowing for adaptation to concept drift.

    **Note:**
    This class should not be instantiated directly, but as it only implements the base algorithm. For classification tasks use :class:`capymoa.classifier.ShrubsClassifier` and for regression tasks use :class:`capymoa.regressor.ShrubsRegressor`.


    Reference:

    `Shrub Ensembles for Online Classification
    Sebastian Buschjäger, Sibylle Hess, and Katharina Morik
    In Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22), Jan 2022.
    <https://aaai.org/papers/06123-shrub-ensembles-for-online-classification/>`_

    """

    def __init__(
        self,
        schema: Schema,
        loss: Literal["mse", "ce", "h2"],
        step_size: float | Literal["adaptive"],
        ensemble_regularizer: Literal["hard-L0", "L0", "L1", "none"],
        l_ensemble_reg: float | int,
        l_l2_reg: float,
        l_tree_reg: float,
        normalize_weights: bool,
        burnin_steps: int,
        update_leaves: bool,
        batch_size: int,
        sk_dt: DecisionTreeClassifier | DecisionTreeRegressor,
    ):
        """Initializes the ShrubEnsemble ensemble with the given parameters.

        :param loss: Literal["mse","ce","h2"] - The loss function to be used. Supported values are "mse", "ce", and "h2".
        :param step_size: float|Literal["adaptive"] - The step size (i.e. learning rate of SGD) for updating the model. Can be a float or "adaptive". Adaptive reduces the step size with more estimators, i.e. sets it to 1.0 / (n_estimators + 1.0)
        :param ensemble_regularizer: Literal["hard-L0","L0","L1","none"] - The regularizer for the weights of the ensemble. Supported values are "none", "L0", "L1", and "hard-L0". Hard-L0 refers to L0 regularization via the prox-operator, whereas L0 and L1 refer to L0/L1 regularization via projection. Projection can be viewed as a softer regularization that drives the weights of each member towards 0, whereas hard-l0 limits the number of trees in the entire ensemble.
        :param l_ensemble_reg: float | int - The regularization strength. If `ensemble_regularizer = hard-L0`, then this parameter represent the total number of trees in the ensembles. If `ensemble_regularizer = L0` or `ensemble_regularizer = L1`, then this parameter is the regularization strength. This these cases the number of trees grow over time and only trees that do not contribute to the ensemble will be removed.
        :param l_l2_reg: float - The L2 regularization strength of the weights of each tree.
        :param l_tree_reg: float - The regularization parameter for individual trees. Must be greater than or equal to 0. `l_tree_reg` controls the number of (overly) large trees in the ensemble by punishing the weights of each tree. Formally, the number of nodes of each tree is used as an additional regularizer.
        :param normalize_weights: bool - Whether to normalize the weights of the ensemble, i.e. the weight sum to 1.
        :param burnin_steps: int - The number of burn-in steps before updating the model, i.e. the number of SGD steps to be take per each call of train
        :param update_leaves: bool - Whether to update the leaves of the trees as well using SGD.
        :param batch_size: int - The batch size for training each individual tree. Internally, a sliding window is stored. Must be greater than or equal to 1.
        :param additional_tree_options: dict - Additional options for the trees, such as splitter, criterion, and max_depth. See sklearn.tree.DecisionTreeClassifier and sklearn.tree.DecisionTreeRegressor for details. An example would be additional_tree_options = {"splitter": "best", "criterion": "gini", "max_depth": None}

        """

        if loss not in ["mse", "ce", "h2"]:
            raise ValueError(
                f"Currently only {{mse, ce, h2}} loss is supported, but you provided {loss}"
            )

        if loss != "mse" and schema.is_regression():
            raise ValueError(
                f"For regression tasks only the mse loss is supported, but you provided {loss}."
            )

        if ensemble_regularizer is not None and ensemble_regularizer not in [
            "none",
            "L0",
            "L1",
            "hard-L0",
        ]:
            raise ValueError(
                f"Currently only {{none, L0, L1, hard-L0}} as ensemble regularizer is supported, but you provided {ensemble_regularizer}."
            )

        if l_tree_reg < 0:
            raise ValueError(
                f"l_tree_reg must be greater or equal to 0, but your provided {l_tree_reg}."
            )

        if batch_size is None or batch_size < 1:
            print(
                f"WARNING: batch_size should be greater than 1, but was {batch_size}. Setting it to 2."
            )
            batch_size = 2

        if ensemble_regularizer == "hard-L0" and l_ensemble_reg < 1:
            print(
                f"WARNING: You set l_ensemble_reg to `{l_ensemble_reg}', but regularizer is `hard-L0'. In this mode, l_ensemble_reg should be an integer 1 <= l_ensemble_reg <= max_trees where max_trees is the maximum number of estimators you allow in the ensemble"
            )

        if l_ensemble_reg > 0 and (
            ensemble_regularizer == "none" or ensemble_regularizer is None
        ):
            print(
                f"WARNING: You set l_ensemble_reg to `{l_ensemble_reg}', but regularizer is None. Ignoring l_ensemble_reg."
            )
            l_ensemble_reg = 0

        if l_ensemble_reg == 0 and (
            ensemble_regularizer != "none" and ensemble_regularizer is not None
        ):
            print(
                "WARNING: You set l_ensemble_reg to 0, but choose regularizer {}.".format(
                    ensemble_regularizer
                )
            )

        if isinstance(step_size, str) and step_size != "adaptive":
            step_size = float(step_size)

        # Options set by the user
        self.step_size = step_size
        self.loss = loss
        self.ensemble_regularizer = ensemble_regularizer
        self.l_ensemble_reg = l_ensemble_reg
        self.l_tree_reg = l_tree_reg
        self.normalize_weights = normalize_weights
        self.update_leaves = update_leaves
        self.sk_dt = sk_dt
        self.batch_size = batch_size
        self.burnin_steps = burnin_steps
        self.l_l2_reg = l_l2_reg

        # Number of classes. For regression we simply set this to 1
        if schema.is_regression():
            self.n_classes_ = 1
        else:
            self.n_classes_ = schema.get_num_classes()

        self.l_l2_reg = l_l2_reg

        # Estimators and their corresponding weights.
        self.estimators_ = []
        self.estimator_weights_ = np.empty(shape=(0,), dtype=float)

        # Shrubs are trained over a sliding window. To enhance performance, we use a circular buffer for the data and for the targets
        self.buffer_data = np.zeros((batch_size, schema.get_num_attributes()))
        self.buffer_target = np.zeros(batch_size, dtype=np.int32)
        self.current_index = 0  # Current index for inserting
        # If false, the buffer is filled up to self.current_index. If true, it is full and we can use the entire buffer
        self.buffer_full = False

    @abstractmethod
    def _individual_proba(self, X):
        pass

    def train(self, instance):
        # Update the ring buffer for the sliding window
        self.buffer_data[self.current_index] = instance.x
        if self.n_classes_ == 1:
            self.buffer_target[self.current_index] = instance.y_value
        else:
            self.buffer_target[self.current_index] = instance.y_index

        self.current_index = (self.current_index + 1) % self.batch_size
        if self.current_index == self.batch_size - 1:
            self.buffer_full = True

        # Get the appropriate data out of the buffer
        if not self.buffer_full:
            data = self.buffer_data[: self.current_index, :]
            target = self.buffer_target[: self.current_index]
        else:
            data = self.buffer_data
            target = self.buffer_target

        if len(set(target)) > 1:
            # Fit a new tree on the current batch.
            tree = clone(self.sk_dt)
            tree.fit(data, target)

            # SKlearn stores the raw counts instead of probabilities in the classification setting.
            # For SGD its better to have the probabilities for numerical stability.
            # tree.tree_.value is not writeable, but we can modify the values inplace. Thus we
            # use [:] to copy the array into the normalized array. Also tree.tree_.value has a strange shape (batch_size, 1, n_classes)
            if self.n_classes_ > 1:
                tree.tree_.value[:] = (
                    tree.tree_.value
                    / tree.tree_.value.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]
                )
        else:
            if self.n_classes_ == 1:
                tree = DummyRegressor(strategy="constant", constant=target[0])
            else:
                tree = DummyClassifier(strategy="constant", constant=target[0])
            tree.fit(data, target)

        self.estimator_weights_ = np.append(self.estimator_weights_, 0.0)
        self.estimators_.append(tree)

        # Prepare step sizes
        if self.step_size == "adaptive":
            step_size = 1.0 / (len(self.estimators_) + 1.0)
        else:
            step_size = self.step_size

        # Get all predictions of all trees for gradient computation
        all_proba = self._individual_proba(data)

        for i in range(self.burnin_steps + 1):
            output = np.array(
                [w * p for w, p in zip(all_proba, self.estimator_weights_)]
            ).sum(axis=0)
            # Compute derivative of the loss
            if self.loss == "mse":
                if self.n_classes_ == 1:
                    loss_deriv = 2 * (output - target)
                else:
                    # Use one hot encoding in classification setting
                    target_one_hot = np.array(
                        [
                            [1.0 if y == i else 0.0 for i in range(self.n_classes_)]
                            for y in target
                        ]
                    )
                    loss_deriv = 2 * (output - target_one_hot)
                # loss = (output - target_one_hot) * (output - target_one_hot)
            elif self.loss == "ce":
                target_one_hot = np.array(
                    [
                        [1.0 if y == i else 0.0 for i in range(self.n_classes_)]
                        for y in target
                    ]
                )
                m = target.shape[0]
                loss_deriv = softmax(output, axis=1)
                loss_deriv[range(m), target_one_hot.argmax(axis=1)] -= 1
                # loss = -target_one_hot*np.log(softmax(output, axis=1) + 1e-7)
            elif self.loss == "h2":
                target_one_hot = np.array(
                    [
                        [1.0 if y == i else -1.0 for i in range(self.n_classes_)]
                        for y in target
                    ]
                )
                zeros = np.zeros_like(target_one_hot)
                loss_deriv = (
                    -2
                    * target_one_hot
                    * np.maximum(1.0 - target_one_hot * output, zeros)
                )
                # loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
            else:
                raise RuntimeError(
                    f"The loss {self.loss} is unknown and for some reason you ended up in a code-path that should not be reachable. Currently only the losses {{ce, mse, h2}} are supported for classification. For regression only mse is supported. "
                )

            # Compute the gradient for the loss
            directions = np.mean(all_proba * loss_deriv, axis=(1, 2))

            # Compute the gradient for the tree regularizer
            if self.l_tree_reg != 0:
                node_deriv = self.l_tree_reg * np.array(
                    [
                        est.tree_.node_count
                        for est in self.estimators_
                        if isinstance(est, DecisionTreeClassifier)
                    ]
                )
            else:
                node_deriv = 0

            # Perform the gradient step. Note that L0 / L1 regularizer is performed via the prox operator and thus performed _after_ this update.
            self.estimator_weights_ = (
                self.estimator_weights_
                - step_size * directions
                - step_size * node_deriv
                - step_size * self.l_l2_reg * 2 * self.estimator_weights_
            )

            # Update leaf values, if necessary
            if self.update_leaves:
                for j, h in enumerate(self.estimators_):
                    tree_grad = (self.estimator_weights_[j] * loss_deriv)[
                        :, np.newaxis, :
                    ]
                    if isinstance(h, DecisionTreeClassifier):
                        idx = h.apply(data)
                        h.tree_.value[idx] = (
                            h.tree_.value[idx]
                            - step_size * tree_grad[:, :, h.classes_.astype(int)]
                        )

        # Compute the prox step.
        if self.ensemble_regularizer == "L0":
            tmp = np.sqrt(2 * self.l_ensemble_reg * step_size)
            tmp_w = np.array(
                [0 if abs(w) < tmp else w for w in self.estimator_weights_]
            )
        elif self.ensemble_regularizer == "L1":
            sign = np.sign(self.estimator_weights_)
            tmp_w = np.abs(self.estimator_weights_) - step_size * self.l_ensemble_reg
            tmp_w = sign * np.maximum(tmp_w, 0)
        elif self.ensemble_regularizer == "hard-L0":
            top_K = np.argsort(self.estimator_weights_)[-self.l_ensemble_reg :]
            tmp_w = np.array(
                [w if i in top_K else 0 for i, w in enumerate(self.estimator_weights_)]
            )
        else:
            tmp_w = self.estimator_weights_

        # If set, normalize the weights. Note that we use the support of tmp_w for the projection onto the probability simplex
        # as described in "Sparse projections onto the simplex" by Kyrillidis et al. 2013 (http://proceedings.mlr.press/v28/kyrillidis13.pdf)
        # Thus, we first need to extract the nonzero weights, project these and then copy them back into corresponding array
        if self.normalize_weights and len(tmp_w) > 0:
            nonzero_idx = np.nonzero(tmp_w)[0]
            nonzero_w = tmp_w[nonzero_idx]
            nonzero_w = to_prob_simplex(nonzero_w)
            self.estimator_weights_ = np.zeros((len(tmp_w)))
            for i, w in zip(nonzero_idx, nonzero_w):
                self.estimator_weights_[i] = w
        else:
            self.estimator_weights_ = tmp_w

        # Keep all non-zero weighted trees and remove the rest
        new_est = []
        new_w = []
        for h, w in zip(self.estimators_, self.estimator_weights_):
            if w > 0:
                new_est.append(h)
                new_w.append(w)

        self.estimators_ = new_est
        self.estimator_weights_ = np.array(new_w)
