from capymoa.base import MOAClassifier
from moa.classifiers.lazy import kNN as _moa_kNN


class KNN(MOAClassifier):
    """
    The default number of neighbors (k) is set to 3 instead of 10 (as in MOA)
    """

    def __init__(self, schema=None, CLI=None, random_seed=1, k=3, window_size=1000):
        # Important, should create the MOA object before invoking the super class __init__
        self.moa_learner = _moa_kNN()
        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )

        # Initialize instance attributes with default values, CLI was not set.
        if self.CLI is None:
            self.k = k
            self.window_size = window_size
            self.moa_learner.getOptions().setViaCLIString(
                f"-k {self.k} -w {self.window_size}"
            )
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()

    def __str__(self):
        # Overrides the default class name from MOA
        return "kNN"
