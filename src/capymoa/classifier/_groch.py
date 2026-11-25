# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Literal, Sequence

from moa.classifiers.meta import GroCH as _MOA_GroCH

from capymoa.base import MOAClassifier
from capymoa.drift.base_detector import MOADriftDetector
from capymoa.stream import Schema

from capymoa.core.moa._cli import cli_str_classifier, cli_str_drift_detector


_DISTANCE_METRICS = ("Euclidean", "Mahalanobis")
_MATRIX_MODES = ("n Matrices", "1 matrix only")
_REPLACEMENT_POLICIES = ("FIFO", "Least Used Model")
_RECURRING_PRIORITIES = ("Old", "New")


def _format_learner_cli(base_learner: MOAClassifier | str) -> str:
    if isinstance(base_learner, str):
        s = base_learner.strip()
        return s if (s.startswith("(") and s.endswith(")")) else f"({s})"
    cli = cli_str_classifier(base_learner)
    cli = cli.strip()
    return cli if (cli.startswith("(") and cli.endswith(")")) else f"({cli})"


def _format_drift_cli(detector: MOADriftDetector | str) -> str:
    if isinstance(detector, str):
        s = detector.strip()
        return s if (s.startswith("(") and s.endswith(")")) else f"({s})"
    cli = cli_str_drift_detector(detector)
    cli = cli.strip()
    return cli if (cli.startswith("(") and cli.endswith(")")) else f"({cli})"


class GroCH(MOAClassifier):
    """Growing Concept History (GroCH) meta-classifier for recurring concepts.

    Growing Concept History (GroCH) [#0]_ maintains a topology-based Concept
    History using Growing Neural Gas (GNG) summaries of past concepts. When
    concept drift is detected, GroCH compares the current concept's topology
    against stored groups using Mahalanobis or Euclidean distance and
    reactivates the best matching historical classifier when a recurring
    concept is identified.

    >>> from capymoa.classifier import GroCH
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = GroCH(
    ...     schema=stream.get_schema(),
    ...     base_learner="trees.HoeffdingTree",
    ...     distance_metric="Euclidean",
    ...     console_log_level=0,
    ... )
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results.accuracy():.1f}")
    84.1

    .. [#0] `Suárez-Cetrulo, A. L., Cervantes, A., & Quintana, D. (2026).
             A growing concept history of recurring classifiers for high-frequency
             data streams. Neurocomputing, 684, 133506.
             <https://doi.org/10.1016/j.neucom.2026.133506>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        CLI: str | None = None,
        random_seed: int = 1,
        base_learner: MOAClassifier | str | None = None,
        drift_detection_method: MOADriftDetector | str | None = None,
        topology_learner: str | None = None,
        distance_metric: Literal["Euclidean", "Mahalanobis"] = "Mahalanobis",
        number_of_matrices: Literal["n Matrices", "1 matrix only"] = "1 matrix only",
        dist_threshold: float = 1.0,
        max_group_size: int = 1,
        group_replacement_policy: Literal["FIFO", "Least Used Model"] = "FIFO",
        priority_of_recurring_classifiers: Literal["Old", "New"] = "New",
        min_insertion_threshold: int = 0,
        min_w_size_for_drift: int = 20,
        min_topology_size_for_drift: int = 1,
        default_window: int = 50,
        min_window_size: int = 50,
        window_resize_policy: int = -1,
        warning_window_size_threshold: int = 500,
        topology_attribute_subset: Sequence[int] | str | None = None,
        initialize_concept_history: bool = False,
        arffs_to_initialize_ch: Sequence[str] | str | None = None,
        save_classifiers_on_false_alarm: bool = False,
        disable_drift_detection: bool = False,
        disable_recurring_drift_detection: bool = False,
        disable_resizing_of_all_windows: bool = False,
        always_insert_classifiers: bool = False,
        update_group_topologies: bool = False,
        disable_topology_learner: bool = False,
        train_online_flag: bool = False,
        multi_pass_topology_training: bool = False,
        multi_cluster: bool = False,
        insert_examples_only: bool = False,
        end_of_warning_after_warning_zone: bool = False,
        always_reset_warning_at_early_drift: bool = False,
        train_after_drift_evaluation: bool = False,
        test_with_bkg_on_warn: bool = False,
        insert_on_early_drift: bool = False,
        weighted_tests_on_warn: bool = False,
        beta: float = 0.5,
        train_active_on_warn: bool = False,
        force_early_drifts: bool = False,
        active_better_when_bkg_nan: bool = False,
        events_log_file: str = "",
        disable_events_log_file: bool = False,
        events_log_file_level: int = 1,
        console_log_level: int = 0,
    ):
        """Construct a GroCH meta-classifier.

        :param schema: The schema of the stream.
        :param CLI: Optional raw MOA CLI string. When provided, all other
            algorithm hyperparameters are ignored and ``CLI`` is passed
            directly to ``moa.classifiers.meta.GroCH``.
        :param random_seed: Seed for the random number generator.
        :param base_learner: Base classifier (CapyMOA ``MOAClassifier`` or MOA
            CLI string, e.g. ``"trees.HoeffdingTree"``). Defaults to
            ``"bayes.NaiveBayes"`` (MOA default).
        :param drift_detection_method: Drift detector (CapyMOA
            ``MOADriftDetector`` or MOA CLI string, e.g. ``"RDDM -o 2.0"``).
            Defaults to ``"DDM"``.
        :param topology_learner: MOA CLI string for the GNG topology clusterer
            (e.g. ``"GNG -l 5 -m 200 -c 2147483647 -b"``).
        :param distance_metric: Concept similarity metric: ``"Euclidean"`` or
            ``"Mahalanobis"``.
        :param number_of_matrices: Mahalanobis covariance modality:
            ``"n Matrices"`` or ``"1 matrix only"``.
        :param dist_threshold: Maximum distance between topologies to match a
            Concept History group (``-t``).
        :param max_group_size: Maximum number of classifiers stored per group
            in the Concept History (``-M``).
        :param group_replacement_policy: Replacement policy when a group is
            full: ``"FIFO"`` or ``"Least Used Model"`` (``-P``).
        :param priority_of_recurring_classifiers: Tie-breaking priority:
            ``"Old"`` or ``"New"`` (``-G``).
        :param min_insertion_threshold: Minimum examples an active classifier
            must see before insertion into the Concept History (``-I``).
        :param min_w_size_for_drift: Minimum warning window size before
            allowing a drift (``-w``).
        :param min_topology_size_for_drift: Minimum number of GNG prototypes
            before allowing a drift (``-a``).
        :param default_window: Default size of dynamic sliding windows (``-D``).
        :param min_window_size: Minimum size of dynamic sliding windows (``-z``).
        :param window_resize_policy: Dynamic window resize policy (``-W``).
        :param warning_window_size_threshold: Threshold for warning window size
            that disables a warning (``-i``).
        :param topology_attribute_subset: Zero-indexed feature indices used for
            topology similarity (sequence of ints or comma-separated string,
            ``-k``).
        :param initialize_concept_history: Whether to pre-initialize the
            Concept History from ARFF files (``-n``).
        :param arffs_to_initialize_ch: Sequence of ARFF file paths or
            semicolon-separated string for Concept History initialization
            (``-o``).
        :param save_classifiers_on_false_alarm: Save classifiers on false
            alarms (``-m``).
        :param disable_drift_detection: Disable drift detection (``-u``).
        :param disable_recurring_drift_detection: Disable recurring concept
            retrieval (``-r``).
        :param disable_resizing_of_all_windows: Keep internal evaluator windows
            static (``-b``).
        :param always_insert_classifiers: Always insert active classifier into
            its group on drift (``-A``).
        :param update_group_topologies: Update stored group topologies on
            insertion (``-j``).
        :param disable_topology_learner: Disable topology learner (``-q``).
        :param train_online_flag: Train topology online rather than only on
            drift (``-H``).
        :param multi_pass_topology_training: Enable multi-pass topology
            training (``-y``).
        :param multi_cluster: Consider all groups within ``dist_threshold``
            during retrieval (``-Y``).
        :param insert_examples_only: Keep a single classifier per group and
            feed instances to it (``-Q``).
        :param end_of_warning_after_warning_zone: End warning window when the
            detector exits its warning zone (``-E``).
        :param always_reset_warning_at_early_drift: Reset to a fresh classifier
            on early drifts (``-Z``).
        :param train_after_drift_evaluation: Train active learner after drift
            evaluation (``-B``).
        :param test_with_bkg_on_warn: Predict with background learner during
            warnings (``-K``).
        :param insert_on_early_drift: Insert active classifier into Concept
            History on early drift (``-J``).
        :param weighted_tests_on_warn: Combine active and background
            predictions during warnings (``-L``).
        :param beta: Penalty factor for weighted tests during warning (``-C``).
        :param train_active_on_warn: Continue training active learner during
            warnings (``-T``).
        :param force_early_drifts: Force background drifts on early drifts
            (``-F``).
        :param active_better_when_bkg_nan: Prefer active learner when
            background error is NaN (``-U``).
        :param events_log_file: Optional file path to export semicolon-separated
            warning/drift event logs (``-e``). Empty disables file logging.
        :param disable_events_log_file: Explicitly disable event file logging
            (``-g``).
        :param events_log_file_level: Event log verbosity: 0=drifts,
            1=drifts+warnings, 2=all instances (``-h``).
        :param console_log_level: Console observability level (0=silent,
            1=drifts, 2=groups, 3=deep structures, ``-V``).
        """
        if CLI is None:
            if distance_metric not in _DISTANCE_METRICS:
                raise ValueError(
                    f"Invalid distance_metric={distance_metric!r}. "
                    f"Choose from {_DISTANCE_METRICS}."
                )
            if number_of_matrices not in _MATRIX_MODES:
                raise ValueError(
                    f"Invalid number_of_matrices={number_of_matrices!r}. "
                    f"Choose from {_MATRIX_MODES}."
                )
            if group_replacement_policy not in _REPLACEMENT_POLICIES:
                raise ValueError(
                    f"Invalid group_replacement_policy={group_replacement_policy!r}. "
                    f"Choose from {_REPLACEMENT_POLICIES}."
                )
            if priority_of_recurring_classifiers not in _RECURRING_PRIORITIES:
                raise ValueError(
                    f"Invalid priority_of_recurring_classifiers="
                    f"{priority_of_recurring_classifiers!r}. "
                    f"Choose from {_RECURRING_PRIORITIES}."
                )

            parts: list[str] = []
            if base_learner is not None:
                parts.append(f"-l {_format_learner_cli(base_learner)}")
            if drift_detection_method is not None:
                parts.append(f"-d {_format_drift_cli(drift_detection_method)}")
            if topology_learner is not None:
                topo = topology_learner.strip()
                if not (topo.startswith("(") and topo.endswith(")")):
                    topo = f"({topo})"
                parts.append(f"-c {topo}")

            parts.append(f"-s {distance_metric}")
            parts.append(f"-v '{number_of_matrices}'")
            parts.append(f"-t {dist_threshold}")
            parts.append(f"-M {max_group_size}")
            parts.append(f"-P '{group_replacement_policy}'")
            parts.append(f"-G {priority_of_recurring_classifiers}")
            parts.append(f"-I {min_insertion_threshold}")
            parts.append(f"-w {min_w_size_for_drift}")
            parts.append(f"-a {min_topology_size_for_drift}")
            parts.append(f"-D {default_window}")
            parts.append(f"-z {min_window_size}")
            parts.append(f"-W {window_resize_policy}")
            parts.append(f"-i {warning_window_size_threshold}")
            parts.append(f"-C {beta}")
            parts.append(f"-h {events_log_file_level}")
            parts.append(f"-V {console_log_level}")

            if topology_attribute_subset is not None:
                if isinstance(topology_attribute_subset, str):
                    subset_str = topology_attribute_subset
                else:
                    subset_str = ",".join(str(idx) for idx in topology_attribute_subset)
                parts.append(f"-k {subset_str}")

            if arffs_to_initialize_ch is not None:
                if isinstance(arffs_to_initialize_ch, str):
                    arffs_str = arffs_to_initialize_ch
                else:
                    arffs_str = ";".join(str(p) for p in arffs_to_initialize_ch)
                parts.append(f"-o {arffs_str}")

            if events_log_file:
                parts.append(f"-e {events_log_file}")

            flag_map = [
                (initialize_concept_history, "-n"),
                (save_classifiers_on_false_alarm, "-m"),
                (disable_drift_detection, "-u"),
                (disable_recurring_drift_detection, "-r"),
                (disable_resizing_of_all_windows, "-b"),
                (always_insert_classifiers, "-A"),
                (update_group_topologies, "-j"),
                (disable_topology_learner, "-q"),
                (train_online_flag, "-H"),
                (multi_pass_topology_training, "-y"),
                (multi_cluster, "-Y"),
                (insert_examples_only, "-Q"),
                (end_of_warning_after_warning_zone, "-E"),
                (always_reset_warning_at_early_drift, "-Z"),
                (train_after_drift_evaluation, "-B"),
                (test_with_bkg_on_warn, "-K"),
                (insert_on_early_drift, "-J"),
                (weighted_tests_on_warn, "-L"),
                (train_active_on_warn, "-T"),
                (force_early_drifts, "-F"),
                (active_better_when_bkg_nan, "-U"),
                (disable_events_log_file, "-g"),
            ]
            for enabled, flag in flag_map:
                if enabled:
                    parts.append(flag)

            CLI = " ".join(parts)

        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=_MOA_GroCH,
        )
