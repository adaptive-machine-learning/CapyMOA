from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
from moa.classifiers.meta.imbalanced import CSMOTE as _MOA_CSMOTE


class CSMOTE(MOAClassifier):
    """CSMOTE

    This strategy saves all the minority samples in a window managed by ADWIN. Meanwhile, a model is trained with
    the input data. When the minority sample ratio falls below a certain threshold, an online version of SMOTE is
    applied. A random minority sample is chosen from the window, and a new synthetic sample is generated until the
    minority sample ratio is greater than or equal to the threshold. The model is then trained with the newly
    generated samples.

    Reference:
    `Alessio Bernardo, Heitor Murilo Gomes, Jacob Montiel, Bernhard Pfahringer, Albert Bifet, Emanuele Della Valle.
    C-SMOTE: Continuous Synthetic Minority Oversampling for Evolving Data Streams.
    In BigData, IEEE, 2020. <https://ieeexplore.ieee.org/document/9377768>`_


    Example usages:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import CSMOTE
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = CSMOTE(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    83.1
    """

    def __init__(
        self,
        schema: Schema = None,
        random_seed: int = 0,
        base_learner = 'trees.HoeffdingTree',
        neighbors: int = 10,
        threshold: float = 0.5,
        min_size_allowed: int = 100,
        disable_drift_detection: bool = False
    ):
        """Incremental on-line boosting classifier of Oza and Russell.

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param base_learner: The base learner to be trained. Default AdaptiveRandomForestClassifier.
        :param neighbors: Number of neighbors for SMOTE.
        :param threshold: Minority class samples threshold.
        :param min_size_allowed: Minimum number of samples in the minority class for applying SMOTE.
        :param disable_drift_detection: If set, disables ADWIN drift detector
        """

        mapping = {
            "base_learner": "-l",
            "neighbors": "-k",
            "threshold": "-t",
            "min_size_allowed": "-m",
            "disable_drift_detection": "-d",
        }

        assert (type(base_learner) == str
                ), "Only MOA CLI strings are supported for CSMOTE base_learner, at the moment."

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super(CSMOTE, self).__init__(
            moa_learner=_MOA_CSMOTE,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )
