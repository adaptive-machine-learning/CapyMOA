from __future__ import annotations
from typing import Union

from capymoa.base import (
    MOAClassifier,
)
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

from moa.classifiers.meta import OzaBoost as _MOA_OzaBoost


class OzaBoost(MOAClassifier):
    """Incremental on-line boosting classifier of Oza and Russell.

    For the boosting method, Oza and Russell note that the
    weighting procedure of AdaBoost actually divides the total example weight
    into two halves – half of the weight is assigned to the correctly classiﬁed
    examples, and the other half goes to the misclassiﬁed examples. They use the
    Poisson distribution for deciding the random probability that an example is
    used for training, only this time the parameter changes according to the
    boosting weight of the example as it is passed through each model in
    sequence.

    Reference:

    `Online bagging and boosting.
    Nikunj Oza, Stuart Russell.
    Artiﬁcial Intelligence and Statistics 2001.
    <https://proceedings.mlr.press/r3/oza01a.html>`_

    Example usages:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import OzaBoost
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = OzaBoost(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    88.8
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 0,
        base_learner = 'trees.HoeffdingTree',
        boosting_iterations: int = 10,
        use_pure_boost: bool = False,
    ):
        """Incremental on-line boosting classifier of Oza and Russell.

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param base_learner: The base learner to be trained. Default trees.HoeffdingTree.
        :param boosting_iterations: The number of boosting iterations.
        :param use_pure_boost: Boost with weights only; no poisson..
        """

        mapping = {
            "base_learner": "-l",
            "boosting_iterations": "-s",
            "use_pure_boost": "-p",
        }

        assert (type(base_learner) == str
                ), "Only MOA CLI strings are supported for OzaBoost base_learner, at the moment."

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super(OzaBoost, self).__init__(
            moa_learner=_MOA_OzaBoost,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )