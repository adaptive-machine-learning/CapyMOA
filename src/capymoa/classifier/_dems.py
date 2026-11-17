from __future__ import annotations

from moa.classifiers.meta import DynamicEnsembleMemberSelection as _MOA_DEMS

from capymoa.base import MOAClassifier
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.stream import Schema


class DynamicEnsembleMemberSelection(MOAClassifier):
    """Dynamic Ensemble Member Selection (DEMS).

    Dynamic Ensemble Member Selection (DEMS) [#0]_ dynamically selects a subset of ensemble members based on their estimated performance and tree-level information.
    Only SRP and ARF are included here because of the performance significance.

    >>> from capymoa.classifier import DynamicEnsembleMemberSelection
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = DynamicEnsembleMemberSelection(stream.get_schema(), random_seed=0)
    >>> results = evaluate_classifier(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    90.6

    .. [#0] `Dynamic Ensemble Member Selection for Data Stream Classification.
             Yibin Sun, Bernhard Pfahringer, Heitor Murilo Gomes, Albert Bifet.
             ACM Conference on Information and Knowledge Management (CIKM), 2025.
             <https://doi.org/10.1145/3746252.3761072>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        ensemble_class: str = "StreamingRandomPatches",  # StreamingRandomPatches / SRP or AdaptiveRandomForest / ARF
        base_learner: str = "trees.HoeffdingTree -g 50 -c 0.01",
        tree_learner: str = "ARFHoeffdingTree -e 2000000 -g 50 -c 0.01",
        ensemble_size: int = 100,
        max_features=0.6,
        training_method: str = "RandomPatches",
        lambda_param: float = 6.0,
        number_of_jobs: int = 1,
        drift_detection_method: str
        | BaseDriftDetector = "ADWINChangeDetector -a 1.0E-5",
        warning_detection_method: str
        | BaseDriftDetector = "ADWINChangeDetector -a 1.0E-4",
        disable_weighted_vote: bool = False,
        disable_drift_detection: bool = False,
        disable_background_learner: bool = False,
        k_value: int = 5,
        disable_self_optimising: bool = False,
        random_seed: int | None = None,
    ):
        """Construct a DynamicEnsembleMemberSelection classifier.

        :param schema: Describes the stream's structure and types.
        :param ensemble_class: Which ensemble to use (``"StreamingRandomPatches"`` or
            ``"AdaptiveRandomForest"``).
        :param base_learner: The base learner to use (used by SRP).
        :param tree_learner: The tree learner to use (only used by ARF, cannot be
            changed).
        :param ensemble_size: The number of members in the ensemble.
        :param max_features: The subspace size for each ensemble member.
            If a float between 0.0 and 1.0, it is the fraction of features (0.6 is 60%).
            If an integer, it is the exact number of features.
            If ``"sqrt"``, it is the square root of the number of features plus 1.
            If ``None``, 60% of the features are used.
        :param training_method: ``"RandomSubspaces"``, ``"Resampling"`` or
            ``"RandomPatches"`` (SRP).
        :param lambda_param: The lambda parameter that controls the Poisson
            distribution for the online bagging simulation.
        :param number_of_jobs: The number of parallel jobs for ARF (-1 is as many as
            possible).
        :param drift_detection_method: The method used for drift detection.
        :param warning_detection_method: The method used for warning detection.
        :param disable_weighted_vote: Whether to disable accuracy-weighted voting.
        :param disable_drift_detection: Whether to disable drift detection (and
            background learners).
        :param disable_background_learner: Whether to disable background learning.
        :param k_value: The fixed K for DEMS when self-optimising is disabled.
        :param disable_self_optimising: Whether to use the fixed ``k_value`` instead
            of self-optimising.
        :param random_seed: Seed for all random choices. ``None`` is a fresh random seed each run.
        """
        ensemble_class_map = {
            "StreamingRandomPatches": "StreamingRandomPatches",
            "SRP": "StreamingRandomPatches",
            "AdaptiveRandomForest": "AdaptiveRandomForest",
            "ARF": "AdaptiveRandomForest",
        }
        assert ensemble_class in ensemble_class_map, (
            f"{ensemble_class} is not a valid ensemble_class. Choose from {list(ensemble_class_map.keys())}"
        )
        ensemble_class_str = ensemble_class_map[ensemble_class]

        # --- Training method (-t) ---
        training_method_map = {
            "RandomSubspaces": "RandomSubspaces",
            "Resampling": "Resampling (bagging)",
            "RandomPatches": "Random Patches",
        }
        assert training_method in training_method_map, (
            f"{training_method} is not a valid training method. "
            f"Choose from {list(training_method_map.keys())}"
        )
        training_method_str = training_method_map[training_method]

        # --- Subspace configuration (-o, -m) ---
        # We mimic SRP wrapper semantics:
        #   feature_mode: one of the textual choices expected by MOA's subspaceModeOption
        #   max_features_per_ensemble_item: integer "m" value
        if isinstance(max_features, float) and 0.0 <= max_features <= 1.0:
            # Percentage mode
            feature_mode = "Percentage (M * (m / 100))"
            max_features_per_ensemble_item = int(max_features * 100)
        elif isinstance(max_features, int):
            # Exact integer
            feature_mode = "Specified m (integer value)"
            max_features_per_ensemble_item = max_features
        elif max_features in ["sqrt"]:
            feature_mode = "sqrt(M)+1"
            max_features_per_ensemble_item = -1  # MOA interprets this with the mode
        elif max_features is None:
            # Default: 60% of features
            feature_mode = "Percentage (M * (m / 100))"
            max_features_per_ensemble_item = 60
        else:
            raise ValueError(
                "Invalid value for max_features. Valid options:\n"
                "  * float between 0.0 and 1.0 representing a percentage,\n"
                "  * an integer specifying exact number, or\n"
                "  * 'sqrt' for square root of total features."
            )

        # Simple sanity check for k_value relative to ensemble_size
        if k_value < 1:
            raise ValueError("k_value must be >= 1")
        k_value = min(k_value, ensemble_size)

        options = {
            "ensembleClassOption": ensemble_class_str,
            "baseLearner": base_learner,
            "ensembleSize": ensemble_size,
            "subspaceMode": feature_mode,
            "subspaceSize": max_features_per_ensemble_item,
            "trainingMethod": training_method_str,
            "lambda": lambda_param,
            "numberOfJobs": number_of_jobs,
            "driftDetectionMethod": drift_detection_method,
            "warningDetectionMethod": warning_detection_method,
            "disableWeightedVote": disable_weighted_vote,
            "disableDriftDetection": disable_drift_detection,
            "disableBackgroundLearner": disable_background_learner,
            "kValues": k_value,
            "disableSelfOptimizing": disable_self_optimising,
            "treeLearner": tree_learner,
        }
        super().__init__(
            moa_learner=_MOA_DEMS(),
            schema=schema,
            random_seed=random_seed,
            options=options,
        )
