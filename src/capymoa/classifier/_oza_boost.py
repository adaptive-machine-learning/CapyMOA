from __future__ import annotations

from moa.classifiers.meta import OzaBoost as _MOA_OzaBoost

from capymoa.base import Classifier, MOAClassifier
from capymoa.stream import Schema


class OzaBoost(MOAClassifier):
    """Incremental on-line boosting classifier of Oza and Russell.

    Incremental on-line boosting classifier of Oza and Russell [#0]_ is a ensemble
    classifier. For the boosting method, Oza and Russell note that the weighting
    procedure of AdaBoost actually divides the total example weight into two halves –
    half of the weight is assigned to the correctly classified examples, and the other
    half goes to the misclassified examples. They use the Poisson distribution for
    deciding the random probability that an example is used for training, only this time
    the parameter changes according to the boosting weight of the example as it is
    passed through each model in sequence.

    >>> from capymoa.classifier import OzaBoost
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = OzaBoost(stream.get_schema(), random_seed=0)
    >>> results = evaluate_classifier(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    88.8

    .. [#0] `Online bagging and boosting. Nikunj Oza, Stuart Russell. Artiﬁcial
             Intelligence and Statistics 2001.
             <https://proceedings.mlr.press/r3/oza01a.html>`_
    """

    def __init__(
        self,
        schema: Schema,
        base_learner: str | Classifier = "trees.HoeffdingTree",
        boosting_iterations: int = 10,
        use_pure_boost: bool = False,
        random_seed: int | None = None,
    ) -> None:
        """Incremental on-line boosting classifier of Oza and Russell.

        :param schema: Describes the stream's structure and types.
        :param base_learner: The base learner to be trained. Default trees.HoeffdingTree.
        :param boosting_iterations: The number of boosting iterations.
        :param use_pure_boost: Boost with weights only; no poisson..
        :param random_seed: Seed for all random choices. ``None`` is a fresh random seed each run.
        """

        options = {
            "baseLearner": base_learner,
            "ensembleSize": boosting_iterations,
            "pureBoost": use_pure_boost,
        }
        super().__init__(
            moa_learner=_MOA_OzaBoost,
            schema=schema,
            random_seed=random_seed,
            options=options,
        )
