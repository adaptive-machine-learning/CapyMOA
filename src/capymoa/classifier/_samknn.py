from capymoa.base import MOAClassifier
from moa.classifiers.lazy import SAMkNN as _MOA_SAMkNN
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals


class SAMkNN(MOAClassifier):
    """Self Adjusted Memory k Nearest Neighbor (SAMkNN) Classifier

    Reference:

    "KNN Classifier with Self Adjusting Memory for Heterogeneous Concept Drift"
    Viktor Losing, Barbara Hammer and Heiko Wersing
    http://ieeexplore.ieee.org/document/7837853
    PDF can be found at https://pub.uni-bielefeld.de/download/2907622/2907623
    BibTex:
    "@INPROCEEDINGS{7837853,
    author={V. Losing and B. Hammer and H. Wersing},
    booktitle={2016 IEEE 16th International Conference on Data Mining (ICDM)},
    title={KNN Classifier with Self Adjusting Memory for Heterogeneous Concept Drift},
    year={2016},
    ages={291-300},
    keywords={data mining;optimisation;pattern classification;Big Data;Internet of Things;KNN classifier;SAM-kNN robustness;data mining;k nearest neighbor algorithm;metaparameter optimization;nonstationary data streams;performance evaluation;self adjusting memory model;Adaptation models;Benchmark testing;Biological system modeling;Data mining;Heuristic algorithms;Prediction algorithms;Predictive models;Data streams;concept drift;data mining;kNN},
    doi={10.1109/ICDM.2016.0040},
    month={Dec}
    }"

    Example usages:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import SAMkNN
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = SAMkNN(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    78.60000000000001
    """

    def __init__(
        self,
        schema: Schema,
        random_seed: int = 1,
        k: int = 5,
        limit: int = 5000,
        min_stm_size: int = 50,
        relative_ltm_size: float = 0.4,
        recalculate_stm_error: bool = False,
    ):
        """Self Adjusted Memory k Nearest Neighbor (SAMkNN) Classifier

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param k: The number of nearest neighbors.
        :param limit: The maximum number of instances to store.
        :param min_stm_size: The minimum number of instances in the STM.
        :param relative_ltm_size: The allowed LTM size relative to the total limit.
        :param recalculate_stm_error: Recalculates the error rate of the STM for size adaption (Costly operation).
            Otherwise, an approximation is used.
        """

        mapping = {
            "k": "-k",
            "limit": "-w",
            "min_stm_size": "-m",
            "relative_ltm_size": "-p",
            "recalculate_stm_error": "-r",
        }

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        self.moa_learner = _MOA_SAMkNN()
        super(SAMkNN, self).__init__(
            schema=schema,
            random_seed=random_seed,
            CLI=config_str,
            moa_learner=self.moa_learner,
        )
