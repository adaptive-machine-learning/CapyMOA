from __future__ import annotations
import typing

from capymoa.base import MOAClassifier
from capymoa.stream import Schema

import moa.classifiers.bayes as moa_bayes


class NaiveBayes(MOAClassifier):
    """Naive Bayes incremental learner.
    Performs classic Bayesian prediction while making the naive assumption that all inputs are independent. Naive Bayes is a classifier algorithm known for its simplicity and low computational cost. Given n different classes, the trained Naive Bayes classifier predicts, for every unlabeled instance I, the class C to which it belongs with high accuracy.

    :param schema: The schema of the stream, defaults to None.
    :param random_seed: The random seed passed to the MOA learner, defaults to 0.
    """

    def __init__(self, schema: typing.Union[Schema, None] = None, random_seed: int = 0):
        super(NaiveBayes, self).__init__(
            moa_learner=moa_bayes.NaiveBayes(), schema=schema, random_seed=random_seed
        )

    def __str__(self):
        # Overrides the default class name from MOA (OzaBag)
        return "Naive Bayes CapyMOA Classifier"
