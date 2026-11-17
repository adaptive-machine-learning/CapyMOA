from moa.classifiers.lazy import SAMkNN as _MOA_SAMkNN

from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.base import MOAClassifier
from capymoa.stream import Schema


class SAMkNN(MOAClassifier):
    """Self Adjusted Memory k Nearest Neighbor.

    Self Adjusted Memory k Nearest Neighbor (SAMkNN) [#0]_ is a lazy classifier.

    >>> from capymoa.classifier import SAMkNN
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = SAMkNN(stream.get_schema())
    >>> results = evaluate_classifier(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    78.6

    .. [#0] `Losing, V., Hammer, B., & Wersing, H. (2016, December). KNN classifier with
             self adjusting memory for heterogeneous concept drift. In 2016 IEEE 16th
             international conference on data mining (ICDM) (pp. 291-300). IEEE.
             <https://pub.uni-bielefeld.de/download/2907622/2907623>`_
    """

    def __init__(
        self,
        schema: Schema,
        k: int = 5,
        limit: int = 5000,
        min_stm_size: int = 50,
        relative_ltm_size: float = 0.4,
        recalculate_stm_error: bool = False,
        random_seed: int | None = None,
    ) -> None:
        """Self Adjusted Memory k Nearest Neighbor (SAMkNN) Classifier

        :param schema: Describes the stream's structure and types.
        :param k: The number of nearest neighbors.
        :param limit: The maximum number of instances to store.
        :param min_stm_size: The minimum number of instances in the STM.
        :param relative_ltm_size: The allowed LTM size relative to the total limit.
        :param recalculate_stm_error: Recalculates the error rate of the STM for size adaption (Costly operation).
            Otherwise, an approximation is used.
        :param random_seed: Seed for all random choices. ``None`` is a fresh random seed each run.
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
        super().__init__(
            schema=schema,
            random_seed=random_seed,
            CLI=config_str,
            moa_learner=self.moa_learner,
        )
