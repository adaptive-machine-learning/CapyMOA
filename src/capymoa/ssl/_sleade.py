from __future__ import annotations

from typing import Literal, Optional

from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.core.moa._cli import cli_str_classifier, cli_str_drift_detector
from capymoa.base import Classifier, MOAClassifierSSL
from capymoa.classifier import HoeffdingTree
from capymoa.drift.base_detector import MOADriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.stream import Schema
import moa.classifiers.semisupervised as moa_ssl


class SLEADE(MOAClassifierSSL):
    """Semi-supervised SLEADE ensemble.

    SLEADE handles partially labelled data by having ensemble members teach one
    another: a member is trained on a pseudo-label when the rest of the ensemble
    predicts it with more confidence than that member would itself. Unsupervised
    drift detection lets the ensemble react to change without waiting for labels.

    The defaults are the configuration used in the paper, so ``SLEADE(schema)``
    reproduces the published method. In particular pseudo-labels are only accepted
    above ``ssl_min_confidence``, and the base ensemble runs with its own drift
    detection and background learner disabled, because SLEADE supplies its own
    unsupervised drift detection instead.

    Reference:

    Gomes, H. M., Read, J., Grzenda, M., Pfahringer, B., & Bifet, A. (2025).
    *SLEADE: Disagreement-Based Semi-Supervised Learning for Sparsely Labeled
    Evolving Data Streams.* IEEE Transactions on Knowledge and Data Engineering.

    >>> from capymoa.ssl import SLEADE
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> clf = SLEADE(stream.get_schema())
    >>> results = prequential_evaluation(stream, clf, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    90.2
    """

    def __init__(
        self,
        schema: Schema,
        random_seed: int = 0,
        ensemble_size: int = 10,
        confidence_strategy: Literal["Sum", "ArgMax"] = "ArgMax",
        enable_random_threshold: bool = False,
        auto_weight_shrinkage: Literal[
            "Constant", "LabeledDivTotal", "LabeledNoWarmupDivTotal"
        ] = "LabeledNoWarmupDivTotal",
        ssl_strategy: Literal[
            "PseudoLabelAll", "PseudoLabelCheckConfidence"
        ] = "PseudoLabelCheckConfidence",
        ssl_min_confidence: float = 0.9,
        weight_function: Literal[
            "Constant1",
            "Confidence",
            "ConfidenceWeightShrinkage",
            "UnsupervisedDetectionWeightShrinkage",
        ] = "ConfidenceWeightShrinkage",
        pairing_function: Literal[
            "MinKappa", "Random", "MajorityTrainsMinority"
        ] = "MajorityTrainsMinority",
        ssl_weight_shrinkage: float = 100.0,
        use_unsupervised_drift_detection: bool = True,
        student_learner: Optional[Classifier] = None,
        drift_detection_method: Optional[MOADriftDetector] = None,
        unsupervised_detection_weight_window: int = 20,
        labeled_window_limit: int = 100,
    ):
        """Construct the SLEADE semi-supervised ensemble.

        :param schema: Stream schema.
        :param random_seed: Random seed.
        :param ensemble_size: Number of learners in the base ensemble. SLEADE is
            built on :class:`~capymoa.classifier.StreamingRandomPatches`, which is
            configured internally with its own drift detection and background
            learner disabled so that SLEADE's unsupervised drift detection is what
            responds to change.
        :param confidence_strategy: How a prediction's confidence is derived from
            the votes: ``"Sum"`` uses the votes directly, ``"ArgMax"`` assigns 1 to
            the argmax of each vote array.
        :param enable_random_threshold: Draw the minimum confidence at random
            instead of using a fixed threshold. When set, ``ssl_min_confidence``
            is **ignored**.
        :param auto_weight_shrinkage: Strategy for setting the weight shrinkage
            automatically.
        :param ssl_strategy: Whether to pseudo-label everything, or only instances
            passing the confidence check.
        :param ssl_min_confidence: Minimum confidence for a pseudo-label to be used
            for training. Ignored when ``enable_random_threshold`` is set.
        :param weight_function: How pseudo-labelled instances are weighted.
            ``"UnsupervisedDetectionWeightShrinkage"`` only makes sense together
            with ``use_unsupervised_drift_detection``.
        :param pairing_function: How learners are paired for teaching.
        :param ssl_weight_shrinkage: Pseudo-labelled instances are weighted by
            ``instance weight * 1/ws``.
        :param use_unsupervised_drift_detection: Whether to use the unsupervised
            drift detection and recovery strategy.
        :param student_learner: Model trained to mimic the ensemble's predictions.
            Because it learns from those predictions rather than from labels, its
            error can be tracked without any labelled data, and a change in that
            error is what signals drift. Only used when
            ``use_unsupervised_drift_detection`` is set. Defaults to
            ``HoeffdingTree(grace_period=50, confidence=0.01)``.
        :param drift_detection_method: Change detector applied to the student's
            error. Defaults to ``ADWIN(delta=1e-5)``.
        :param unsupervised_detection_weight_window: Length of the sigmoid used to
            weight pseudo-labelled instances after an unsupervised detection.
        :param labeled_window_limit: Maximum number of labelled instances kept in
            the sliding window used to quick-start learners.
        """
        if student_learner is None:
            student_learner = HoeffdingTree(
                schema=schema, grace_period=50, confidence=0.01
            )
        if drift_detection_method is None:
            drift_detection_method = ADWIN(delta=1e-5)

        mapping = {
            "confidence_strategy": "-b",
            "enable_random_threshold": "-q",
            "auto_weight_shrinkage": "-e",
            "ssl_strategy": "-p",
            "ssl_min_confidence": "-m",
            "weight_function": "-w",
            "pairing_function": "-t",
            "ssl_weight_shrinkage": "-n",
            "use_unsupervised_drift_detection": "-s",
            "unsupervised_detection_weight_window": "-z",
            "labeled_window_limit": "-j",
        }
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        # The base ensemble is not a hyperparameter: SLEADE is only implemented
        # against StreamingRandomPatches, and `-u -q` are part of the method rather
        # than a tuning choice. `cli_str_*` already parenthesise their output, so
        # these are appended directly rather than through the mapping above.
        config_str += f"-l (StreamingRandomPatches -s {ensemble_size} -u -q) "
        config_str += f"-g {cli_str_classifier(student_learner)} "
        config_str += f"-x {cli_str_drift_detector(drift_detection_method)} "

        super(SLEADE, self).__init__(
            moa_learner=moa_ssl.SLEADE,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )
