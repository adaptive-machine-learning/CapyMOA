from __future__ import annotations

from capymoa.base import (
    MOAClassifier,
)
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

from moa.classifiers.meta import StreamingRandomPatches as _MOA_SRP
from moa.classifiers.meta.minibatch import StreamingRandomPatchesMB as _MOA_SRP_MB
import os


class StreamingRandomPatches(MOAClassifier):
    """Streaming Random Patches.

    Streaming Random Patches (SRP) [#0]_ is a ensemble classifier. It uses a hoeffding
    tree by default, but it can be used with any other base model (differently from
    random forest variations). This algorithm can be used to simulate bagging or random
    subspaces, see parameter training_method. The default algorithm uses both bagging
    and random subspaces, namely Random Patches.

    >>> from capymoa.classifier import StreamingRandomPatches
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = StreamingRandomPatches(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    89.7

    .. [#0] `Streaming Random Patches for Evolving Data Stream Classification. Heitor
             Murilo Gomes, Jesse Read, Albert Bifet. IEEE International Conference on
             Data Mining (ICDM), 2019. <https://doi.org/10.1109/ICDM.2019.00034>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 0,
        base_learner="trees.HoeffdingTree -g 50 -c 0.01",
        ensemble_size=100,
        max_features=0.6,
        training_method: str = "RandomPatches",
        lambda_param: float = 6.0,
        minibatch_size=None,
        number_of_jobs=None,
        drift_detection_method="ADWINChangeDetector -a 1.0E-5",
        warning_detection_method="ADWINChangeDetector -a 1.0E-4",
        disable_weighted_vote: bool = False,
        disable_drift_detection: bool = False,
        disable_background_learner: bool = False,
    ):
        """Streaming Random Patches (SRP) Classifier

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param base_learner: The base learner to be trained. Default trees.HoeffdingTree -g 50 -c 0.01.
        :param ensemble_size: The number of trees in the ensemble.
        :param max_features: The subspace size for each ensemble member.
            If provided as a float between 0.0 and 1.0, it represents the percentage of features to consider.
            If provided as an integer, it specifies the exact number of features to consider.
            If provided as the string "sqrt", it indicates that the square root of the total number of features.
            If not provided, the default value is 60%.
        :param training_method: The training method to use: RandomSubspaces, Resampling or RandomPatches.
            RandomSubspaces: Random Subspaces.
            Resampling: Resampling (bagging).
            RandomPatches: Random Patches.
        :param lambda_param: The lambda parameter that controls the Poisson distribution for
            the online bagging simulation.
        :param minibatch_size: The number of instances that a learner must accumulate before training.
        :param number_of_jobs: The number of parallel jobs to run during the execution of the algorithm.
            By default, the algorithm executes tasks sequentially (i.e., with `number_of_jobs=1`).
            Increasing the `number_of_jobs` can lead to faster execution on multi-core systems.
            However, setting it to a high value may consume more system resources and memory.
            This implementation is designed to be embarrassingly parallel, meaning that the algorithm's computations
            can be efficiently distributed across multiple processing units without sacrificing predictive
            performance. It's recommended to experiment with different values to find the optimal setting based on
            the available hardware resources and the nature of the workload.
        :param drift_detection_method: The method used for drift detection.
        :param warning_detection_method: The method used for warning detection.
        :param disable_weighted_vote: Whether to disable weighted voting.
        :param disable_drift_detection: Whether to disable drift detection.
        :param disable_background_learner: Whether to disable background learning.
        """
        moa_learner = None

        mapping = {
            "base_learner": "-l",
            "ensemble_size": "-s",
            "feature_mode": "-o",
            "max_features_per_ensemble_item": "-m",
            "training_method_str": "-t",
            "lambda_param": "-a",
            "drift_detection_method": "-x",
            "warning_detection_method": "-p",
            "disable_weighted_vote": "-w",
            "disable_drift_detection": "-u",
            "disable_background_learner": "-q",
        }

        training_method_map = {
            "RandomSubspaces": "Random Subspaces",
            "Resampling": "Resampling (bagging)",
            "RandomPatches": "Random Patches",
        }
        assert training_method in training_method_map, (
            f"{training_method} is not a valid training method."
        )
        training_method_str = training_method_map[training_method]

        assert isinstance(base_learner, str), (
            "Only MOA CLI strings are supported for SRP base_learner, at the moment."
        )

        # max_features = max_features
        if isinstance(max_features, float) and 0.0 <= max_features <= 1.0:
            feature_mode = "Percentage (M * (m / 100))"
            max_features_per_ensemble_item = int(max_features * 100)
        elif isinstance(max_features, int):
            feature_mode = "Specified m (integer value)"
            max_features_per_ensemble_item = max_features
        elif max_features in ["sqrt"]:
            feature_mode = "sqrt(M)+1"
            max_features_per_ensemble_item = -1  # or leave it unchanged
        elif max_features is None:
            feature_mode = "Percentage (M * (m / 100))"
            max_features_per_ensemble_item = 60
        else:
            # Raise an exception with information about valid options for max_features
            raise ValueError(
                "Invalid value for max_features. Valid options: \n"
                "float between 0.0 and 1.0 representing a percentage,\n"
                "an integer specifying exact number, or\n"
                "'sqrt' for square root of total features."
            )
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        if (number_of_jobs is None or number_of_jobs == 0 or number_of_jobs == 1) and (
            minibatch_size is None or minibatch_size <= 0 or minibatch_size == 1
        ):
            number_of_jobs = 1
            minibatch_size = 1
            moa_learner = _MOA_SRP()

        else:
            if number_of_jobs == 0 or number_of_jobs is None:
                self.number_of_jobs = 1
            elif number_of_jobs < 0:
                self.number_of_jobs = os.cpu_count()
            else:
                self.number_of_jobs = int(min(number_of_jobs, os.cpu_count()))
            if minibatch_size <= 1:
                # if the user sets the number of jobs and the minibatch_size less than 1 it is considered that the user wants a parallel execution of a single instance at a time
                self.minibatch_size = 1
            elif minibatch_size is None:
                # if the user sets only the number_of_jobs, we assume he wants the parallel minibatch version and initialize minibatch_size to the default 25
                self.minibatch_size = 25
            else:
                # if the user sets both parameters to values greater than 1, we initialize the minibatch_size to the user's choice
                self.minibatch_size = int(minibatch_size)
            moa_learner = _MOA_SRP_MB()
            config_str += f"-b {self.minibatch_size} "
            config_str += f"-c {self.number_of_jobs} "

        super(StreamingRandomPatches, self).__init__(
            moa_learner=moa_learner,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )
