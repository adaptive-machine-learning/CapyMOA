# SPDX-License-Identifier: BSD-3-Clause
import warnings
import numpy as np
from collections import deque
from typing import Literal, Optional, List
from openmoa.stream import Stream
from openmoa.stream import Schema
from openmoa.instance import LabeledInstance, RegressionInstance


def _calc_eds_boundaries(total_instances: int, n_segments: int, overlap_ratio: float) -> List[int]:
    """Compute the 2n-1 stage boundaries for an EDS / EvolvableStream timeline.

    Shared by :class:`OpenFeatureStream` (EDS pattern) and
    :class:`EvolvableStream` so the formula lives in exactly one place.

    Stages alternate between *stable* (single partition active) and *overlap*
    (two adjacent partitions active).  A stable period has length ``L`` and an
    overlap period has length ``L * overlap_ratio``, giving::

        total = n * L + (n-1) * overlap_ratio * L
              = L * (n + overlap_ratio * (n-1))

    :param total_instances: Total stream length.
    :param n_segments: Number of feature partitions (>= 2).
    :param overlap_ratio: Ratio of overlap period to stable period length.
    :returns: List of *exclusive* right-boundary indices for each of the
        ``2n - 1`` stages.  The last entry is always ``total_instances``.
    """
    denom = n_segments + overlap_ratio * (n_segments - 1)
    L = total_instances / denom if denom > 0 else 0.0

    boundaries: List[int] = []
    pos = 0.0
    for i in range(2 * n_segments - 1):
        pos += L if i % 2 == 0 else L * overlap_ratio
        boundaries.append(int(pos))

    boundaries[-1] = total_instances
    return boundaries


class OpenFeatureStream(Stream):
    """Wraps a fixed-feature stream into an evolving-feature stream.

    Simulates various feature-evolution scenarios (concept drift in feature
    space).  Attaches ``feature_indices`` (global feature IDs) to every
    generated instance so downstream algorithms can correctly align features
    regardless of the physical array position – resolving the *index-shift
    problem* that arises when dimensions change over time.

    Supported evolution patterns
    ----------------------------
    ``'pyramid'``
        Dimension increases linearly to *d_max* then decreases back to
        *d_min*.
    ``'incremental'``
        Dimension increases monotonically from *d_min* to *d_max*.
    ``'decremental'``
        Dimension decreases monotonically from *d_max* to *d_min*.
    ``'tds'``
        Trapezoidal Data Stream.  Features have distinct "birth times"
        spread across 10 evenly-spaced stages (the number of stages is an
        implementation constant matching the original TDS definition).
        ``tds_mode='random'`` assigns birth stages randomly;
        ``tds_mode='ordered'`` assigns them by feature index.
    ``'cds'``
        Capricious Data Stream.  Each feature is independently present with
        probability ``1 - missing_ratio`` at every time step (Bernoulli
        trial).
    ``'eds'``
        Evolvable Data Stream.  Feature space evolves in *n_segments*
        sequential partitions with overlapping transition periods of
        relative length *overlap_ratio*.

    .. note::
        ``missing_ratio`` is only used by the ``'cds'`` pattern.  It is
        silently ignored for all other patterns.
    """

    def __init__(
        self,
        base_stream: Stream,
        d_min: int = 2,
        d_max: Optional[int] = None,
        evolution_pattern: Literal["pyramid", "incremental", "decremental", "tds", "cds", "eds"] = "pyramid",
        total_instances: int = 10000,
        feature_selection: Literal["prefix", "suffix", "random"] = "prefix",
        missing_ratio: float = 0.0,
        random_seed: int = 42,
        # TDS-specific
        tds_mode: Literal["random", "ordered"] = "random",
        # EDS-specific
        n_segments: int = 2,
        overlap_ratio: float = 1.0,
    ):
        self.base_stream = base_stream
        self.d_min = d_min

        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d

        if self.d_max > original_d:
            raise ValueError(
                f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})"
            )
        if self.d_min < 1:
            raise ValueError("d_min must be >= 1")
        if self.d_min > self.d_max:
            raise ValueError(f"d_min ({self.d_min}) must be <= d_max ({self.d_max})")

        self.evolution_pattern = evolution_pattern
        self.total_instances = total_instances
        self.feature_selection = feature_selection
        self.missing_ratio = missing_ratio
        self.random_seed = random_seed
        self.tds_mode = tds_mode
        self.n_segments = n_segments
        self.overlap_ratio = overlap_ratio

        # Instance-level RNG (never pollutes global numpy state)
        self._rng = np.random.RandomState(random_seed)
        self._current_t = 0
        self._schema = base_stream.get_schema()

        # Pre-compute deterministic schedules
        if evolution_pattern in ("pyramid", "incremental", "decremental"):
            self._dimension_schedule = self._generate_dimension_schedule()
            # Indices are computed lazily in _get_active_indices to avoid
            # allocating O(N) arrays upfront.  Only the 'random' selection
            # mode needs per-timestep seeding; prefix/suffix are O(1).

        elif evolution_pattern == "tds":
            self._feature_offsets = self._generate_tds_offsets()

        elif evolution_pattern == "eds":
            if self.n_segments < 2:
                raise ValueError("n_segments must be >= 2 for EDS pattern")
            self._eds_partitions = self._generate_eds_partitions()
            self._eds_boundaries = _calc_eds_boundaries(
                total_instances, n_segments, overlap_ratio
            )

        # 'cds' is fully stochastic and computed on-the-fly via per-t seeding.

    # ------------------------------------------------------------------
    # Schedule / offset helpers
    # ------------------------------------------------------------------

    def _generate_dimension_schedule(self) -> np.ndarray:
        """Return integer dimension for each time step (pyramid / incr / decr)."""
        if self.evolution_pattern == "pyramid":
            half = self.total_instances // 2
            up   = np.linspace(self.d_min, self.d_max, half)
            down = np.linspace(self.d_max, self.d_min, self.total_instances - half)
            dims = np.concatenate([up, down])
        elif self.evolution_pattern == "incremental":
            dims = np.linspace(self.d_min, self.d_max, self.total_instances)
        else:  # decremental
            dims = np.linspace(self.d_max, self.d_min, self.total_instances)
        return dims.astype(int)

    def _generate_tds_offsets(self) -> np.ndarray:
        """Assign a birth time (time-step index) to each feature for TDS.

        The timeline is divided into 10 evenly-spaced stages – this constant
        comes from the original TDS paper definition and is intentional.
        """
        # 10 birth stages as defined by TDS
        n_stages = 10
        time_step = self.total_instances // n_stages
        offsets = np.zeros(self.d_max, dtype=int)

        if self.tds_mode == "random":
            # Randomly distribute features across the 10 birth stages
            perm = self._rng.permutation(self.d_max)
            for i, feat_idx in enumerate(perm):
                offsets[feat_idx] = (i % n_stages) * time_step
        else:  # ordered
            # Feature i belongs to stage floor(i * n_stages / d_max)
            for i in range(self.d_max):
                bucket = (i * n_stages) // self.d_max
                offsets[i] = bucket * time_step

        return offsets

    def _generate_eds_partitions(self) -> List[np.ndarray]:
        """Partition d_max features sequentially into n_segments groups."""
        parts = np.array_split(np.arange(self.d_max), self.n_segments)
        return [np.sort(p) for p in parts]

    # ------------------------------------------------------------------
    # Core index engine
    # ------------------------------------------------------------------

    def _get_active_indices(self, t: int) -> np.ndarray:
        """Return the set of active global feature IDs at time step *t*."""

        # --- Deterministic patterns (lazy, no cache) ---
        if self.evolution_pattern in ("pyramid", "incremental", "decremental"):
            d = int(self._dimension_schedule[t]) if t < len(self._dimension_schedule) else self.d_min
            if self.feature_selection == "prefix":
                return np.arange(d)
            elif self.feature_selection == "suffix":
                return np.arange(self.d_max - d, self.d_max)
            else:  # random – reproducible via per-t seed offset
                rng_t = np.random.RandomState(self.random_seed + t)
                idx = rng_t.choice(self.d_max, d, replace=False)
                idx.sort()
                return idx

        # --- TDS ---
        elif self.evolution_pattern == "tds":
            return np.where(self._feature_offsets <= t)[0]

        # --- CDS ---
        elif self.evolution_pattern == "cds":
            rng_t = np.random.RandomState(self.random_seed + t)
            mask = rng_t.rand(self.d_max) > self.missing_ratio
            indices = np.where(mask)[0]
            # Guarantee at least one feature; prefer d_min but fall back to 1
            if len(indices) == 0:
                indices = rng_t.choice(self.d_max, max(1, self.d_min), replace=False)
                indices.sort()
            return indices

        # --- EDS ---
        elif self.evolution_pattern == "eds":
            stage_idx = len(self._eds_boundaries) - 1
            for i, boundary in enumerate(self._eds_boundaries):
                if t < boundary:
                    stage_idx = i
                    break

            if stage_idx % 2 == 0:
                # Stable stage: single partition
                p_idx = stage_idx // 2
                return self._eds_partitions[p_idx]
            else:
                # Overlap stage: union of two adjacent partitions
                prev = (stage_idx - 1) // 2
                return np.concatenate([
                    self._eds_partitions[prev],
                    self._eds_partitions[prev + 1],
                ])

        return np.arange(self.d_max)

    # ------------------------------------------------------------------
    # Stream interface
    # ------------------------------------------------------------------

    def next_instance(self):
        if not self.has_more_instances():
            return None

        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None

        active_indices = self._get_active_indices(self._current_t)

        x_full = np.asarray(base_instance.x, dtype=float)
        valid_indices = active_indices[active_indices < len(x_full)]
        if len(valid_indices) == 0:
            valid_indices = np.array([0])

        x_subset = x_full[valid_indices]

        if self._schema.is_classification():
            new_instance = LabeledInstance.from_array(
                self._schema, x_subset, base_instance.y_index
            )
        else:
            new_instance = RegressionInstance.from_array(
                self._schema, x_subset, base_instance.y_value
            )

        # Attach global IDs so downstream algorithms can align features
        new_instance.feature_indices = valid_indices

        self._current_t += 1
        return new_instance

    def has_more_instances(self) -> bool:
        return self._current_t < self.total_instances and self.base_stream.has_more_instances()

    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
        # Reset RNG so repeated runs produce identical sequences
        self._rng = np.random.RandomState(self.random_seed)

    def get_schema(self) -> Schema:
        """Return the schema of the original (global) feature space."""
        return self._schema

    def get_moa_stream(self):
        return None


# Readable alias used in papers and tutorials
EvolvingFeatureStream = OpenFeatureStream


# ---------------------------------------------------------------------------


class TrapezoidalStream(Stream):
    """Fixed-dimension stream that marks inactive features with ``np.nan``.

    Unlike :class:`OpenFeatureStream` (which shrinks the physical vector),
    this wrapper keeps the vector length at ``d_max`` and fills inactive
    positions with ``np.nan``.  Useful for algorithms such as OVFM that
    consume a fixed-size schema but handle missing values natively.

    Evolution modes
    ---------------
    ``'random'``
        Features appear one-by-one in random order until all ``d_max``
        features are active (linear growth: *d_min* → *d_max*).
    ``'ordered'``
        Same as ``'random'`` but features activate in index order (0, 1, …).
    ``'pyramid'``
        Features activate sequentially up to *d_max* then deactivate
        sequentially (triangular trend: *d_min* → *d_max* → *d_min*).
    """

    def __init__(
        self,
        base_stream: Stream,
        d_min: int = 2,
        d_max: Optional[int] = None,
        evolution_mode: Literal["random", "ordered", "pyramid"] = "random",
        total_instances: int = 10000,
        random_seed: int = 42,
    ):
        self.base_stream = base_stream
        self.d_min = d_min

        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d

        if self.d_max > original_d:
            raise ValueError(
                f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})"
            )

        self.evolution_mode = evolution_mode
        self.total_instances = total_instances
        self.random_seed = random_seed

        self._current_t = 0
        self._rng = np.random.RandomState(random_seed)
        self._schema = base_stream.get_schema()

        self._feature_ranking = self._generate_feature_ranking()
        self._dimension_schedule = self._generate_dimension_schedule()

    def _generate_feature_ranking(self) -> np.ndarray:
        """Return feature indices in birth-priority order (lowest index = born first)."""
        all_indices = np.arange(self.d_max)
        if self.evolution_mode == "random":
            self._rng.shuffle(all_indices)
            return all_indices
        elif self.evolution_mode in ("ordered", "pyramid"):
            return all_indices
        else:
            raise ValueError(f"Unknown evolution_mode: {self.evolution_mode!r}")

    def _generate_dimension_schedule(self) -> np.ndarray:
        if self.evolution_mode in ("random", "ordered"):
            dims = np.linspace(self.d_min, self.d_max, self.total_instances)
        else:  # pyramid
            half = self.total_instances // 2
            up   = np.linspace(self.d_min, self.d_max, half)
            down = np.linspace(self.d_max, self.d_min, self.total_instances - half)
            dims = np.concatenate([up, down])
        return dims.astype(int)

    def next_instance(self):
        if not self.has_more_instances():
            return None

        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None

        t_idx = min(self._current_t, self.total_instances - 1)
        num_active = self._dimension_schedule[t_idx]
        active_indices = self._feature_ranking[:num_active]

        x_full = np.full(self.d_max, np.nan)
        x_base = np.asarray(base_instance.x, dtype=float)[:self.d_max]
        x_full[active_indices] = x_base[active_indices]

        if self._schema.is_classification():
            instance = LabeledInstance.from_array(
                self._schema, x_full, base_instance.y_index
            )
        else:
            instance = RegressionInstance.from_array(
                self._schema, x_full, base_instance.y_value
            )

        self._current_t += 1
        return instance

    def has_more_instances(self) -> bool:
        return self._current_t < self.total_instances and self.base_stream.has_more_instances()

    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
        # Re-seed so the random birth order is identical on every run
        self._rng = np.random.RandomState(self.random_seed)
        self._feature_ranking = self._generate_feature_ranking()

    def get_schema(self) -> Schema:
        return self._schema

    def get_moa_stream(self):
        return None


# ---------------------------------------------------------------------------


class CapriciousStream(Stream):
    """Simulates a Capricious Data Stream (CDS) with NaN-masked missing features.

    Maintains a **fixed** feature dimension of ``d_max``.  At each time step
    each feature is independently present (Bernoulli trial with probability
    ``1 - missing_ratio``); absent features become ``np.nan``.

    At least ``min_features`` features are guaranteed to be observed per
    instance.
    """

    def __init__(
        self,
        base_stream: Stream,
        d_max: Optional[int] = None,
        missing_ratio: float = 0.5,
        total_instances: int = 10000,
        min_features: int = 1,
        random_seed: int = 42,
    ):
        self.base_stream = base_stream

        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d

        self.missing_ratio = missing_ratio
        self.total_instances = total_instances
        self.min_features = min_features
        self.random_seed = random_seed

        self._current_t = 0
        self._schema = base_stream.get_schema()

    def _get_feature_mask(self, t: int) -> np.ndarray:
        """Return boolean mask (True = observed) for time step *t*."""
        rng_t = np.random.RandomState(self.random_seed + t)
        mask = rng_t.rand(self.d_max) > self.missing_ratio

        if np.sum(mask) < self.min_features:
            forced = rng_t.choice(self.d_max, self.min_features, replace=False)
            mask = np.zeros(self.d_max, dtype=bool)
            mask[forced] = True

        return mask

    def next_instance(self):
        if not self.has_more_instances():
            return None

        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None

        mask = self._get_feature_mask(self._current_t)

        x_base = np.asarray(base_instance.x, dtype=float)
        if len(x_base) > self.d_max:
            x_base = x_base[:self.d_max]

        x_masked = x_base.copy()
        x_masked[~mask] = np.nan

        if self._schema.is_classification():
            instance = LabeledInstance.from_array(
                self._schema, x_masked, base_instance.y_index
            )
        else:
            instance = RegressionInstance.from_array(
                self._schema, x_masked, base_instance.y_value
            )

        self._current_t += 1
        return instance

    def has_more_instances(self) -> bool:
        return self._current_t < self.total_instances and self.base_stream.has_more_instances()

    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
        # CapriciousStream uses deterministic per-t seeding (seed + t), so
        # resetting _current_t is sufficient for reproducibility.

    def get_schema(self) -> Schema:
        return self._schema

    def get_moa_stream(self):
        return None


# ---------------------------------------------------------------------------


class EvolvableStream(Stream):
    """N-phase Evolvable Data Stream (EDS) with fixed dimension and NaN masking.

    Designed for algorithms (e.g. OVFM) that expect a fixed global feature
    space but handle missing values.  The feature space evolves through
    ``n_segments`` sequential partitions with overlapping transition periods.

    Stage schedule (2n − 1 stages)
    --------------------------------
    - Even stages  → stable: only partition *k* is active.
    - Odd stages   → overlap: partitions *k* and *k+1* are both active.
    """

    def __init__(
        self,
        base_stream: Stream,
        d_max: Optional[int] = None,
        n_segments: int = 2,
        overlap_ratio: float = 1.0,
        total_instances: int = 10000,
        random_seed: int = 42,
    ):
        self.base_stream = base_stream

        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d

        if self.d_max > original_d:
            raise ValueError(
                f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})"
            )
        if n_segments < 2:
            raise ValueError("n_segments must be >= 2")

        self.n_segments = n_segments
        self.overlap_ratio = overlap_ratio
        self.total_instances = total_instances
        self.random_seed = random_seed

        self._current_t = 0
        self._schema = base_stream.get_schema()

        self._partitions = self._generate_partitions()
        self._stage_boundaries = _calc_eds_boundaries(total_instances, n_segments, overlap_ratio)

    def _generate_partitions(self) -> List[np.ndarray]:
        parts = np.array_split(np.arange(self.d_max), self.n_segments)
        return [np.sort(p) for p in parts]

    def _get_active_mask(self, t: int) -> np.ndarray:
        """Return boolean mask (True = active) for the full d_max vector at time *t*."""
        stage_idx = len(self._stage_boundaries) - 1
        for i, boundary in enumerate(self._stage_boundaries):
            if t < boundary:
                stage_idx = i
                break

        mask = np.zeros(self.d_max, dtype=bool)

        if stage_idx % 2 == 0:
            p_idx = stage_idx // 2
            if p_idx < len(self._partitions):
                mask[self._partitions[p_idx]] = True
        else:
            prev = (stage_idx - 1) // 2
            if prev + 1 < len(self._partitions):
                mask[self._partitions[prev]] = True
                mask[self._partitions[prev + 1]] = True
            else:
                mask[self._partitions[prev]] = True

        return mask

    def next_instance(self):
        if not self.has_more_instances():
            return None

        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None

        mask = self._get_active_mask(self._current_t)

        x_full = np.full(self.d_max, np.nan)
        x_base = np.asarray(base_instance.x, dtype=float)
        limit = min(len(x_base), self.d_max)

        active_in_range = mask[:limit]
        x_full[:limit][active_in_range] = x_base[:limit][active_in_range]

        if self._schema.is_classification():
            instance = LabeledInstance.from_array(
                self._schema, x_full, base_instance.y_index
            )
        else:
            instance = RegressionInstance.from_array(
                self._schema, x_full, base_instance.y_value
            )

        self._current_t += 1
        return instance

    def has_more_instances(self) -> bool:
        return self._current_t < self.total_instances and self.base_stream.has_more_instances()

    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
        # No RNG state to reset (partitions are deterministic).

    def get_schema(self) -> Schema:
        return self._schema

    def get_moa_stream(self):
        return None


# ---------------------------------------------------------------------------


class ShuffledStream(Stream):
    """Buffers the entire base stream and serves instances in shuffled order.

    Essential for static datasets (e.g. Magic04, Spambase) that may be sorted
    by label, which would distort prequential evaluation.

    .. warning::
        Loads the full dataset into memory.  Safe for UCI-scale datasets
        (MBs).  Do **not** use for multi-GB streams.
    """

    def __init__(self, base_stream: Stream, random_seed: int = 42):
        self.base_stream = base_stream
        self._schema = base_stream.get_schema()
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)

        self._instances = []
        while base_stream.has_more_instances():
            try:
                inst = base_stream.next_instance()
                if inst is not None:
                    self._instances.append(inst)
            except StopIteration:
                break
            except Exception as exc:
                warnings.warn(
                    f"ShuffledStream: stopped early after {len(self._instances)} instances "
                    f"due to exception: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

        self.n_instances = len(self._instances)

        self._indices = np.arange(self.n_instances)
        self._rng.shuffle(self._indices)
        self._ptr = 0

    def has_more_instances(self) -> bool:
        return self._ptr < self.n_instances

    def next_instance(self):
        if not self.has_more_instances():
            return None
        instance = self._instances[self._indices[self._ptr]]
        self._ptr += 1
        return instance

    def restart(self):
        """Reset pointer; shuffle order is preserved across restarts for reproducibility."""
        self._ptr = 0

    def get_schema(self) -> Schema:
        return self._schema

    def get_num_instances(self) -> int:
        """Total number of buffered instances."""
        return self.n_instances

    def get_moa_stream(self):
        return None
