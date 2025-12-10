from __future__ import annotations

from capymoa.base import MOAClassifierSSL
from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.stream import Schema
import moa.classifiers.semisupervised as moa_ssl


class SLEADE(MOAClassifierSSL):
    """Semi-supervised SLEADE ensemble.

    SLEADE method handles partially labelled data and unsupervised drift detection.

    >>> from capymoa.ssl import SLEADE
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> clf = SLEADE(stream.get_schema())
    >>> results = prequential_evaluation(stream, clf, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    90.7
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 0,
        base_ensemble: str = "StreamingRandomPatches",
        confidence_strategy: str = "ArgMax",
        enable_random_threshold: bool = False,
        auto_weight_shrinkage: str = "LabeledNoWarmupDivTotal",
        ssl_strategy: str = "PseudoLabelCheckConfidence",
        ssl_min_confidence: float = 0.0,
        weight_function: str = "ConfidenceWeightShrinkage",
        pairing_function: str = "MajorityTrainsMinority",
        ssl_weight_shrinkage: float = 100.0,
        use_unsupervised_drift_detection: bool = False,
        student_learner_for_unsupervised_drift_detection: str = (
            "trees.HoeffdingTree -g 50 -c 0.01"
        ),
        drift_detection_method: str = "ADWINChangeDetector -a 1.0E-5",
        unsupervised_detection_weight_window: int = 20,
        labeled_window_limit: int = 100,
    ):
        """Construct the SLEADE semi-supervised ensemble.

        :param schema: Stream schema.
        :param random_seed: Random seed.
        :param base_ensemble: Base ensemble learner (e.g., StreamingRandomPatches).
        :param confidence_strategy: Confidence strategy ('Sum' or 'ArgMax').
        :param enable_random_threshold: Use random min-confidence threshold.
        :param auto_weight_shrinkage: Strategy for automatic weight shrinkage.
        :param ssl_strategy: Semi-supervised learning strategy.
        :param ssl_min_confidence: Minimum confidence to accept pseudo-label.
        :param weight_function: Function for weighting pseudo-labelled instances.
        :param pairing_function: Learner pairing function.
        :param ssl_weight_shrinkage: Pseudo-label weight shrinkage value.
        :param use_unsupervised_drift_detection: Whether to enable unsupervised drift detection.
        :param student_learner_for_unsupervised_drift_detection: Student model for drift detection.
        :param drift_detection_method: Drift detection algorithm and parameters.
        :param unsupervised_detection_weight_window: Window size for unsupervised drift detection weighting.
        :param labeled_window_limit: Maximum number of labelled instances in buffer.
        """

        mapping = {
            "base_ensemble": "-l",
            "confidence_strategy": "-b",
            "enable_random_threshold": "-q",
            "auto_weight_shrinkage": "-e",
            "ssl_strategy": "-p",
            "ssl_min_confidence": "-m",
            "weight_function": "-w",
            "pairing_function": "-t",
            "ssl_weight_shrinkage": "-n",
            "use_unsupervised_drift_detection": "-s",
            "student_learner_for_unsupervised_drift_detection": "-g",
            "drift_detection_method": "-x",
            "unsupervised_detection_weight_window": "-z",
            "labeled_window_limit": "-j",
        }

        # Build CLI string (handles nested learners if applicable)
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super(SLEADE, self).__init__(
            moa_learner=moa_ssl.SLEADE,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )
