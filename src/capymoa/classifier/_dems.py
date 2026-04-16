from __future__ import annotations

from capymoa.base import MOAClassifier
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

from moa.classifiers.meta import DynamicEnsembleMemberSelection as _MOA_DEMS


class DynamicEnsembleMemberSelection(MOAClassifier):
    """Dynamic Ensemble Member Selection (DEMS).

    Python wrapper for the MOA implementation:

        moa.classifiers.meta.DynamicEnsembleMemberSelection

    DEMS dynamically selects a subset of ensemble members based on their estimated performance and tree-level information.
    Only SRP and ARF are included here because of the performance significance.

    >>> from capymoa.classifier import DynamicEnsembleMemberSelection
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = DynamicEnsembleMemberSelection(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results.accuracy():.1f}")
    90.6

    Parameters mirror the Java options:

    - ensemble_class: which ensemble to use ("StreamingRandomPatches" or "AdaptiveRandomForest").
    - base_learner: base classifier (used by SRP).
    - tree_learner: ARF tree learner (only used by ARF, cannot be changed).
    - ensemble_size: number of ensemble members.
    - max_features: subspace size configuration, similar to SRP:
        * float in [0, 1]: percentage of features (e.g. 0.6 = 60%).
        * int: exact number of features.
        * "sqrt": use sqrt(M)+1.
        * None: default (60%).
    - training_method: "RandomSubspaces", "Resampling", or "RandomPatches" (SRP).
    - lambda_param: Poisson lambda for bagging.
    - number_of_jobs: number of parallel jobs for ARF (-1 = as many as possible).
    - drift_detection_method: MOA CLI string for drift detector.
    - warning_detection_method: MOA CLI string for warning detector.
    - disable_weighted_vote: if True, disables accuracy-weighted voting.
    - disable_drift_detection: if True, turns off drift detectors (and bkg learners).
    - disable_background_learner: if True, turns off background learners.
    - k_value: fixed K for DEMS when self-optimising is disabled.
    - disable_self_optimising: if True, use the fixed k_value instead of self-optimising.

    Reference:
      Yibin Sun, Bernhard Pfahringer, Heitor Murilo Gomes, Albert Bifet.
      "Dynamic Ensemble Member Selection for Data Stream Classification."
      CIKM 2025.
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 0,
        ensemble_class: str = "StreamingRandomPatches",    # StreamingRandomPatches / SRP or AdaptiveRandomForest / ARF
        base_learner: str = "trees.HoeffdingTree -g 50 -c 0.01",
        tree_learner: str = "ARFHoeffdingTree -e 2000000 -g 50 -c 0.01",
        ensemble_size: int = 100,
        max_features=0.6,
        training_method: str = "RandomPatches",
        lambda_param: float = 6.0,
        number_of_jobs: int = 1,
        drift_detection_method: str = "ADWINChangeDetector -a 1.0E-5",
        warning_detection_method: str = "ADWINChangeDetector -a 1.0E-4",
        disable_weighted_vote: bool = False,
        disable_drift_detection: bool = False,
        disable_background_learner: bool = False,
        k_value: int = 5,
        disable_self_optimising: bool = False,
    ):
        # Map Python-friendly params to MOA CLI strings

        # --- Ensemble choice (-e) ---
        ensemble_class_map = {
            "StreamingRandomPatches": "StreamingRandomPatches",
            "SRP": "StreamingRandomPatches",
            "AdaptiveRandomForest": "AdaptiveRandomForest",
            "ARF": "AdaptiveRandomForest",
        }
        assert (
            ensemble_class in ensemble_class_map
        ), f"{ensemble_class} is not a valid ensemble_class. Choose from {list(ensemble_class_map.keys())}"
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
        if k_value > ensemble_size:
            # We don't hard-fail, but you may want to be stricter:
            k_value = ensemble_size

        # Mapping from local variable names to MOA CLI flags
        mapping = {
            "ensemble_class_str": "-e",
            "base_learner": "-l",
            "ensemble_size": "-s",
            "feature_mode": "-o",
            "max_features_per_ensemble_item": "-m",
            "training_method_str": "-t",
            "lambda_param": "-a",
            "number_of_jobs": "-j",
            "drift_detection_method": "-x",
            "warning_detection_method": "-p",
            "disable_weighted_vote": "-w",
            "disable_drift_detection": "-u",
            "disable_background_learner": "-q",
            "k_value": "-k",
            "disable_self_optimising": "-f",
            "tree_learner": "-1",
        }

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        moa_learner = _MOA_DEMS()

        super(DynamicEnsembleMemberSelection, self).__init__(
            moa_learner=moa_learner,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )