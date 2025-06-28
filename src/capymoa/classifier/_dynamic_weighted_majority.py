from capymoa.base import MOAClassifier
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
from moa.classifiers.meta import DynamicWeightedMajority as _MOA_DWM


class DynamicWeightedMajority(MOAClassifier):
    """Dynamic Weighted Majority.

    Dynamic Weighted Majority [#0]_ is a meta-strategy.

    >>> from capymoa.classifier import DynamicWeightedMajority
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = DynamicWeightedMajority(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    85.7

    .. [#0] `J. Zico Kolter and Marcus A. Maloof. Dynamic weighted majority: An ensemble
             method for drifting concepts. The Journal of Machine Learning Research,
             8:2755-2790, December 2007. ISSN 1532-4435.
             <http://dl.acm.org/citation.cfm?id=1314498.1390333>`_
    """

    def __init__(
        self,
        schema: Schema,
        random_seed: int = 1,
        base_learner="bayes.NaiveBayes",
        period: int = 50,
        beta: float = 0.5,
        theta: float = 0.01,
        max_experts: int = 10000,  # overwrite Integer.MAX_VALUE in Java with 10000
    ):
        """Construct Dynamic Weighted Majority.

        :param schema: Describes the data stream.
        :param random_seed: random seed for reproducibility.
        :param base_learner: the base learner to be used, defaults to "NaiveBayes"
        :param period: period between expert removal, creation, and weight update.
        :param beta: factor to punish mistakes of experts.
        :param theta: minimum fraction of weight per model.
        :param max_experts: maximum number of allowed experts.
        """
        mapping = {
            "base_learner": "-l",
            "period": "-p",
            "beta": "-b",
            "theta": "-t",
            "max_experts": "-e",
        }

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super(DynamicWeightedMajority, self).__init__(
            schema=schema,
            random_seed=random_seed,
            CLI=config_str,
            moa_learner=_MOA_DWM,
        )
