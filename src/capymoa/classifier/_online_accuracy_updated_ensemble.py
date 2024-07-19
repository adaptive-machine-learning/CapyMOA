from capymoa.base import MOAClassifier
from moa.classifiers.meta import OnlineAccuracyUpdatedEnsemble as _MOA_OnlineAccuracyUpdatedEnsemble
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals


class OnlineAccuracyUpdatedEnsemble(MOAClassifier):
    """OnlineAccuracyUpdatedEnsemble

    The online version of the Accuracy Updated Ensemble as proposed by
    Brzezinski and Stefanowski in "Combining block-based and online methods 
    in learning ensembles from concept drifting data streams", Information Sciences, 2014.

    Reference:
    
    `Combining block-based and online methods in learning ensembles from concept drifting data streams.
    Daruisz Brzezinski, Jerzy Stefanowski
    IS, 2014.
    <https://doi.org/10.1016/j.ins.2013.12.011>`_

    Example usages:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import OnlineAccuracyUpdatedEnsemble
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = OnlineAccuracyUpdatedEnsemble(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    87.5
    """


    def __init__(
        self,
        schema: Schema,
        random_seed: int = 1,
        learner_option= 'trees.HoeffdingTree -e 2000000 -g 100 -c 0.01', 
        member_count_option: int = 5,
        window_size_option: float = 50.0,
        max_byte_size_option: int = 33554432,
        verbose_option: bool = False,
        linear_option: bool = False,
    ):
        

        """ Online Accuracy Updated Ensemble Classifier

        :param schema: The schema of the stream.
        :param learner_option: Classifier to train.
        :param member_count_option: The maximum number of classifiers in an ensemble.
        :param window_size_option: The window size used for classifier creation and evaluation.
        :param max_byte_size_option: Maximum memory consumed by ensemble.
        :param verbose_option: When checked the algorithm outputs additional information about component classifier weights.
        :param linear_option: When checked the algorithm uses a linear weighting function.
        """


        mapping = {
            "learner_option": "-l",
            "member_count_option": "-n",
            "window_size_option": "-w",
            "max_byte_size_option": "-m",
            "verbose_option": "-v",
            "linear_option": "-f",
        }


        assert (type(learner_option) == str
            ), "Only MOA CLI strings are supported for SGBT base_learner, at the moment."


        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        self.moa_learner = _MOA_OnlineAccuracyUpdatedEnsemble()
        super(OnlineAccuracyUpdatedEnsemble, self).__init__(
            schema=schema,
            random_seed=random_seed,
            CLI=config_str,
            moa_learner=self.moa_learner,
        )