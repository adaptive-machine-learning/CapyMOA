from __future__ import annotations

from capymoa.base import (
    MOAClassifier,
)
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

from moa.classifiers.meta import OnlineSmoothBoost as _MOA_OnlineSmoothBoost


class OnlineSmoothBoost(MOAClassifier):
    """Online Smooth Boost.

    Online Smooth Boost [#0]_ is a ensemble classifier. Incremental on-line boosting
    with Theoretical Justifications of Shang-Tse Chen.

    >>> from capymoa.classifier import OnlineSmoothBoost
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = OnlineSmoothBoost(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    87.8

    .. [#0] `An Online Boosting Algorithm with Theoretical Justifications. Shang-Tse
             Chen, Hsuan-Tien Lin, Chi-Jen Lu. ICML, 2012.
             <https://icml.cc/2012/papers/538.pdf>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 0,
        base_learner="trees.HoeffdingTree",
        boosting_iterations: int = 100,
        gamma=0.1,
    ):
        """OnlineSmoothBoost Classifier

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param base_learner: The base learner to be trained. Default trees.HoeffdingTree.
        :param boosting_iterations: The number of boosting iterations (ensemble size).
        :param gamma: The value of the gamma parameter.
        """

        mapping = {
            "base_learner": "-l",
            "boosting_iterations": "-s",
            "gamma": "-g",
        }

        assert isinstance(base_learner, str), (
            "Only MOA CLI strings are supported for OnlineSmoothBoost base_learner, at the moment."
        )

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super(OnlineSmoothBoost, self).__init__(
            moa_learner=_MOA_OnlineSmoothBoost,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )
