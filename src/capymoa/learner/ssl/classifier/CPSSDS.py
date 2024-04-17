import typing as t
from typing import Dict, Literal

import numpy as np
from river.base import Classifier
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier

from capymoa.learner.ssl.classifier.batch import BatchClassifierSSL
from capymoa.stream import Schema
from capymoa.stream.instance import Instance


def shuffle_split(
    split_proportion: float, x: np.ndarray, y: np.ndarray
) -> t.Tuple[t.Tuple[np.ndarray, np.ndarray], t.Tuple[np.ndarray, np.ndarray]]:
    """Shuffle and split the data into two parts.

    :param split_proportion: The proportion of the dataset to be included in
        the first part.
    :param x: The instances to split.
    :param y: The labels to split.
    :raises LengthMismatchError: The length of x and y must be the same.
    :return: Two tuples containing the instances and labels of the two parts.
    """
    assert len(x) == len(y), "x and y must have the same length"
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    split_index = int(len(x) * split_proportion)
    idx_a = indices[:split_index]
    idx_b = indices[split_index:]
    return (x[idx_a], y[idx_a]), (x[idx_b], y[idx_b])


def split_by_label_presence(
    x: np.ndarray, y: np.ndarray
) -> t.Tuple[t.Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Split the data into labeled and unlabeled instances.

    :param x: A batch of instances.
    :param y: A batch of labels where -1 means that the instance is unlabeled.
    :raises LengthMismatchError: The length of x and y must be the same.
    :return:
        - A tuple containing the labeled instances and labels.
        - A numpy array containing the unlabeled instances.
    """
    assert len(x) == len(y), "x and y must have the same length"
    labeled_mask = y != -1
    return (x[labeled_mask], y[labeled_mask]), x[~labeled_mask]


def Unlabeling_data(X_train, Y_train, Percentage, chunk_size, class_count):
    labeled_count = round(Percentage * chunk_size)
    TLabeled = X_train[0 : labeled_count - 1]
    Y_TLabeled = Y_train[0 : labeled_count - 1]
    X_Unlabeled = X_train[labeled_count : Y_train.shape[0] - 1]

    cal_count = round(0.3 * TLabeled.shape[0])
    X_cal = TLabeled[0 : cal_count - 1]
    Y_cal = Y_TLabeled[0 : cal_count - 1]
    X_L = TLabeled[cal_count : TLabeled.shape[0] - 1]
    Y_L = Y_TLabeled[cal_count : TLabeled.shape[0] - 1]

    return X_Unlabeled, X_L, Y_L, X_cal, Y_cal


def Prediction_by_CP(num, classifier, X, Y, X_Unlabeled, class_count, sl):
    row = X_Unlabeled.shape[0]
    col = class_count
    p_values = np.zeros([row, col])
    labels = np.ones((row, col), dtype=bool)
    alphas = NCM(num, classifier, X, Y, 1, class_count)
    for elem in range(row):
        c = []
        for o in range(class_count):
            a_test = NCM(
                num, classifier, np.array([X_Unlabeled[elem, :]]), o, 2, class_count
            )
            idx = np.argwhere(Y == o).flatten()
            temp = alphas[idx]
            p = len(temp[temp >= a_test])
            if idx.shape[0] == 0:
                s = 0
            else:
                s = p / idx.shape[0]
            c.append(s)
            if s < sl:
                labels[elem, int(o)] = False
        p_values[elem, :] = np.array(c)
    return p_values, labels


def NCM(num, classifier, X, Y, t, class_count):
    if num == 1:
        if t == 1:
            p = np.zeros([X.shape[0], 1])
            alpha = np.zeros([X.shape[0], 1])
            for g in range(X.shape[0]):
                dic_vote = classifier.predict_proba_one(np_to_dict(X[g, :]))
                vote = np.fromiter(dic_vote.values(), dtype=float)
                vote_keys = np.fromiter(dic_vote.keys(), dtype=int)
                Sum = np.sum(vote)
                keys = np.argwhere(vote_keys == int(Y[g])).flatten()
                if keys.size == 0:
                    p[g] = (1) / (Sum + class_count)
                else:
                    for key, val in dic_vote.items():
                        if key == float(Y[g]):
                            p[g] = (val + 1) / (Sum + class_count)
                alpha[g] = 1 - p[g]

        else:
            dic_vote = classifier.predict_proba_one(np_to_dict(X[0, :]))
            vote = np.fromiter(dic_vote.values(), dtype=float)
            vote_keys = np.fromiter(dic_vote.keys(), dtype=int)
            Sum = np.sum(vote)
            keys = np.argwhere(vote_keys == int(Y)).flatten()
            if keys.size == 0:
                p = (1) / (Sum + class_count)
            else:
                for key, val in dic_vote.items():
                    if key == float(Y):
                        p = (val + 1) / (Sum + class_count)
            alpha = 1 - p

    else:
        if t == 1:
            prediction = predict_many(classifier, X)
            P = np.max(prediction, axis=1)
            alpha = 1 - P
        elif t == 2:
            prediction = predict_many(classifier, X)
            # TODO: This is a hacky patch because river tries to be smart and
            # infer the number of classes from the data. This is silly because
            # CPSSDS assumes that the number of classes is known. Future work
            # will replace river with MOA.
            if prediction.shape[1] <= Y:
                P = 0
            else:
                P = prediction[0, int(Y)]
            alpha = 1 - P
    return alpha


def Informatives_selection(X_Unlabeled, p_values, labels, class_count):
    row = X_Unlabeled.shape[0]
    X = np.empty([1, X_Unlabeled.shape[1]])
    Y = np.empty([1])
    for elem in range(row):
        l = np.argwhere(labels[elem, :] == True).flatten()
        if len(l) == 1:
            pp = p_values[elem, l]
            X = np.append(X, [X_Unlabeled[elem, :]], axis=0)
            Y = np.append(Y, [l[0]], axis=0)
    Informatives = X[1 : X.shape[0], :]
    Y_Informatives = Y[1 : Y.shape[0]]
    return Informatives, Y_Informatives


def Appending_informative_to_nextchunk(
    X_Currentchunk_Labeled, Y_Currentchunk_Labeled, Informatives, Y_Informatives
):
    X = np.append(X_Currentchunk_Labeled, Informatives, axis=0)
    Y = np.append(Y_Currentchunk_Labeled, Y_Informatives, axis=0)
    return X, Y


def np_to_dict(x):
    return dict(enumerate(x))


def predict_many(classifier: Classifier, x: np.ndarray) -> np.ndarray:
    """Predict the labels of a batch of instances.

    :param classifier: The classifier to use.
    :param x: A batch of instances.
    :return: A numpy array containing the predicted labels.
    """
    if len(x) == 0:
        return np.array([])
    results = []
    for x_i in x:
        y_hat = classifier.predict_proba_one(np_to_dict(x_i))
        y_hat_skmf = np.array(list(y_hat.values()))
        results.append(y_hat_skmf)
    return np.stack(results)


class CPSSDS(BatchClassifierSSL):
    """Conformal prediction for semi-supervised classification on data streams.

    Tanha, J., Samadi, N., Abdi, Y., & Razzaghi-Asl, N. (2021). CPSSDS:
    Conformal prediction for semi-supervised classification on data streams.
    Information Sciences, 584, 212–234. https://doi.org/10.1016/j.ins.2021.10.068
    """

    def __init__(
        self,
        base_model: Literal["NaiveBayes", "HoeffdingTree"],
        batch_size: int,
        schema: Schema,
        significance_level: float = 0.98,
        calibration_split: float = 0.3,
        random_seed=1,
    ) -> None:
        """Constructor for CPSSDS.

        :param base_model: An underlying model which is augmented with
            self-labeled data from conformal prediction.
        :param batch_size: The number of instances to train on at a time.
        :param schema: The schema of the data stream.
        :param significance_level: Controls the required confidence level for
            unlabeled instances to be labeled. Must be between 0 and 1. defaults to 0.98
        :param calibration_split: The proportion of the labeled data to be used
            for calibration. defaults to 0.3
        :param random_seed: The random seed to use for reproducibility.
        :raises ValueError: `base_model` must be either NaiveBayes or HoeffdingTree
        """
        super().__init__(batch_size, schema, random_seed)
        self.significance_level: float = significance_level
        self.chunk_id = 0
        self.class_count = schema.get_num_classes()
        self.calibration_split = calibration_split

        # TODO: These classifiers should be replaced with MOA classifiers
        if base_model == "NaiveBayes":
            self.classifier = GaussianNB()
            self._num = 2
        elif base_model == "HoeffdingTree":
            self.classifier = HoeffdingTreeClassifier()
            self._num = 1
        else:
            raise ValueError("`base_model` must be either NaiveBayes or HoeffdingTree")

        # Self-labeled data, initialized as empty
        self.self_labeled_x: np.array = None
        self.self_labeled_y: np.array = None

        # Set seed for reproducibility
        np.random.seed(random_seed)

    def train_on_batch(self, x_batch, y_indices):
        (x_label, y_label), x_unlabeled = split_by_label_presence(x_batch, y_indices)
        (x_cal, y_cal), (x_train, y_train) = shuffle_split(
            self.calibration_split, x_label, y_label
        )

        # Add self-labeled data to training data
        if self.self_labeled_x is not None and self.self_labeled_y is not None:
            x_train = np.concatenate((x_train, self.self_labeled_x))
            y_train = np.concatenate((y_train, self.self_labeled_y))

        for x_one, y_one in zip(x_train, y_train):
            self.classifier.learn_one(dict(enumerate(x_one)), y_one)

        assert x_cal.shape[0] > 0, "Calibration data must not be empty"
        assert x_unlabeled.shape[0] > 0, "Unlabeled data must not be empty"
        """Issues arise when not enough labeled data is available for calibration.
        This can be solved by increasing the calibration split or increasing the
        batch size.
        """

        # Use conformal prediction to label some unlabeled data
        p_values, labels = Prediction_by_CP(
            self._num,
            self.classifier,
            x_cal,
            y_cal,
            x_unlabeled,
            self.class_count,
            self.significance_level,
        )

        # Add newly labeled data to self-labeled data
        self.self_labeled_x, self.self_labeled_y = Informatives_selection(
            x_unlabeled, p_values, labels, self.class_count
        )

    def instance_to_dict(self, instance: Instance) -> Dict[str, float]:
        """Convert an instance to a dictionary with the feature names as keys."""
        return dict(enumerate(instance.x))

    def skmf_to_river(self, x):
        return dict(enumerate(x))

    def predict(self, instance: Instance):
        class_index = self.classifier.predict_one(self.instance_to_dict(instance))
        if class_index is None:
            return None
        return class_index

    def predict_proba(self, instance):
        raise NotImplementedError()

    def __str__(self):
        return f"CPSSDS(significance_level={self.significance_level})"
