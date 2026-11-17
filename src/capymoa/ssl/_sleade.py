from __future__ import annotations

from typing import Literal

import moa.classifiers.semisupervised as moa_ssl

from capymoa.base import Classifier, MOAClassifierSSL
from capymoa.classifier import HoeffdingTree
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.stream import Schema


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
    >>> from capymoa.classifier import evaluate_classifier
    >>>
    >>> stream = ElectricityTiny()
    >>> clf = SLEADE(stream.get_schema())
    >>> results = evaluate_classifier(stream, clf, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    90.2
    """

    def __init__(
        self,
        schema: Schema,
        random_seed: int | None = None,
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
        student_learner: Classifier | None = None,
        drift_detection_method: BaseDriftDetector | None = None,
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
            ``HoeffdingTree(grace_period=50, confidence=0.01)``. Any CapyMOA
            classifier, Python or MOA.
        :param drift_detection_method: Change detector applied to the student's
            error. Defaults to ``ADWIN(delta=1e-5)``. Any CapyMOA drift detector,
            Python or MOA.
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

        options = {
            # The base ensemble is not a hyperparameter: SLEADE is only
            # implemented against StreamingRandomPatches, and `-u -q` are part
            # of the method rather than a tuning choice.
            "baseEnsemble": f"StreamingRandomPatches -s {ensemble_size} -u -q",
            "confidenceStrategy": confidence_strategy,
            "enableRandomThreshold": enable_random_threshold,
            "autoWeightShrinkage": auto_weight_shrinkage,
            "SSLStrategy": ssl_strategy,
            "SSL_minConfidence": ssl_min_confidence,
            "weightFunction": weight_function,
            "pairingFunction": pairing_function,
            "SSL_weightShrinkageOption": ssl_weight_shrinkage,
            "useUnsupervisedDriftDetection": use_unsupervised_drift_detection,
            "studentLearnerForUnsupervisedDriftDetection": student_learner,
            "driftDetectionMethod": drift_detection_method,
            "unsupervisedDetectionWeightWindow": unsupervised_detection_weight_window,
            "labeledWindowLimit": labeled_window_limit,
        }
        super().__init__(
            moa_learner=moa_ssl.SLEADE,
            schema=schema,
            random_seed=random_seed,
            options=options,
        )
