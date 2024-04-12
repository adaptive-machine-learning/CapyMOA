from __future__ import annotations

from capymoa.learner import MOAClassifier
from capymoa.stream import Schema

import moa.classifiers.bayes as moa_bayes


class NaiveBayes(MOAClassifier):
    """Naive Bayes incremental learner.

    Performs classic bayesian prediction while making naive assumption that
    all inputs are independent. Naive Bayes is a classiﬁer algorithm known
    for its simplicity and low computational cost. Given n different classes, the
    trained Naive Bayes classiﬁer predicts for every unlabelled instance I the
    class C to which it belongs with high accuracy.

    Parameters
    ----------
    schema
        The schema of the stream
    random_seed
        The random seed passed to the moa learner
    """

    def __init__(self, schema: Schema | None = None, random_seed: int = 0):
        super(NaiveBayes, self).__init__(moa_learner=moa_bayes.NaiveBayes(), 
                                        schema=schema,
                                        random_seed=random_seed)

    def __str__(self):
        # Overrides the default class name from MOA (OzaBag)
        return "Naive Bayes CapyMOA Classifier"

