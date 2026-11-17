from __future__ import annotations

from moa.classifiers.meta import OnlineSmoothBoost as _MOA_OnlineSmoothBoost

from capymoa.base import Classifier, MOAClassifier
from capymoa.stream import Schema


class OnlineSmoothBoost(MOAClassifier):
    """Online Smooth Boost.

    Online Smooth Boost [#0]_ is a ensemble classifier. Incremental on-line boosting
    with Theoretical Justifications of Shang-Tse Chen.

    >>> from capymoa.classifier import OnlineSmoothBoost
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = OnlineSmoothBoost(stream.get_schema())
    >>> results = evaluate_classifier(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    87.8

    .. [#0] `An Online Boosting Algorithm with Theoretical Justifications. Shang-Tse
             Chen, Hsuan-Tien Lin, Chi-Jen Lu. ICML, 2012.
             <https://icml.cc/2012/papers/538.pdf>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        base_learner: str | Classifier = "trees.HoeffdingTree",
        boosting_iterations: int = 100,
        gamma=0.1,
        random_seed: int | None = None,
    ):
        """OnlineSmoothBoost Classifier

        :param schema: Describes the stream's structure and types.
        :param base_learner: The base learner to be trained. Default trees.HoeffdingTree.
        :param boosting_iterations: The number of boosting iterations (ensemble size).
        :param gamma: The value of the gamma parameter.
        :param random_seed: Seed for all random choices. ``None`` is a fresh random seed each run.
        """

        options = {
            "baseLearner": base_learner,
            "ensembleSize": boosting_iterations,
            "gamma": gamma,
        }
        super().__init__(
            moa_learner=_MOA_OnlineSmoothBoost,
            schema=schema,
            random_seed=random_seed,
            options=options,
        )
