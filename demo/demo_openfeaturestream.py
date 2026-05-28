# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal, List

# ==========================================
# 1. 模拟 OpenMOA 基础环境 (无需安装库)
# ==========================================
class Schema:
    def __init__(self, d): self.d = d
    def get_num_attributes(self): return self.d
    def is_classification(self): return True

class Instance:
    def __init__(self, x, y):
        self.x = x
        self.y_index = y
        self.y_value = y
        self.feature_indices = None 

class LabeledInstance(Instance):
    @classmethod
    def from_array(cls, schema, x, y): return cls(x, y)

class RegressionInstance(Instance):
    @classmethod
    def from_array(cls, schema, x, y): return cls(x, y)

class Stream:
    def next_instance(self): pass
    def has_more_instances(self): pass
    def restart(self): pass
    def get_schema(self): pass

# ==========================================
# 2. 你的 OpenFeatureStream 类 (最新优化版)
# ==========================================
class OpenFeatureStream(Stream):
    """
    Wraps a fixed-feature data stream into an evolving feature stream.
    (Optimized version with TDS ordered/random modes and refined EDS)
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
        tds_mode: Literal["random", "ordered"] = "random",
        n_segments: int = 2,
        overlap_ratio: float = 1.0,
    ):
        self.base_stream = base_stream
        self.d_min = d_min
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        self.evolution_pattern = evolution_pattern
        self.total_instances = total_instances
        self.feature_selection = feature_selection
        self.missing_ratio = missing_ratio
        self.random_seed = random_seed
        self.tds_mode = tds_mode
        self.n_segments = n_segments
        self.overlap_ratio = overlap_ratio

        self._rng = np.random.RandomState(random_seed)
        self._current_t = 0
        self._schema = base_stream.get_schema()

        if evolution_pattern in ["pyramid", "incremental", "decremental"]:
            self._dimension_schedule = self._generate_dimension_schedule()
            self._feature_indices_cache = self._generate_feature_indices()
        elif evolution_pattern == "tds":
            self._feature_offsets = self._generate_tds_offsets()
        elif evolution_pattern == "eds":
            self._eds_partitions = self._generate_eds_partitions()
            self._eds_boundaries = self._calculate_eds_boundaries()

    def _generate_dimension_schedule(self) -> np.ndarray:
        dims = np.zeros(self.total_instances, dtype=int)
        if self.evolution_pattern == "pyramid":
            half = self.total_instances // 2
            dims[:half] = np.linspace(self.d_min, self.d_max, half)
            dims[half:] = np.linspace(self.d_max, self.d_min, self.total_instances - half)
        elif self.evolution_pattern == "incremental":
            dims = np.linspace(self.d_min, self.d_max, self.total_instances)
        elif self.evolution_pattern == "decremental":
            dims = np.linspace(self.d_max, self.d_min, self.total_instances)
        return dims.astype(int)

    def _generate_feature_indices(self) -> list:
        indices_list = []
        for t in range(self.total_instances):
            d_current = self._dimension_schedule[t]
            if self.feature_selection == "prefix":
                indices = np.arange(d_current)
            elif self.feature_selection == "suffix":
                indices = np.arange(self.d_max - d_current, self.d_max)
            elif self.feature_selection == "random":
                rng_t = np.random.RandomState(self.random_seed + t)
                indices = rng_t.choice(self.d_max, d_current, replace=False)
                indices.sort()
            else:
                indices = np.arange(d_current)
            indices_list.append(indices)
        return indices_list
    
    def _generate_tds_offsets(self) -> np.ndarray:
        offsets = np.zeros(self.d_max, dtype=int)
        time_step = self.total_instances // 10
        if self.tds_mode == "random":
            indices = self._rng.permutation(self.d_max)
            for i in range(self.d_max):
                offsets[indices[i]] = (i % 10) * time_step
        elif self.tds_mode == "ordered":
            for i in range(self.d_max):
                bucket = (i * 10) // self.d_max
                offsets[i] = bucket * time_step
        return offsets

    def _generate_eds_partitions(self) -> List[np.ndarray]:
        all_indices = np.arange(self.d_max)
        partitions = np.array_split(all_indices, self.n_segments)
        return [np.sort(p) for p in partitions]

    def _calculate_eds_boundaries(self) -> List[int]:
        n = self.n_segments
        r = self.overlap_ratio
        L = self.total_instances / (n + r * (n - 1))
        boundaries = []
        current_pos = 0.0
        for i in range(2 * n - 1):
            if i % 2 == 0: current_pos += L
            else: current_pos += L * r
            boundaries.append(int(current_pos))
        boundaries[-1] = self.total_instances
        return boundaries

    def _get_active_indices(self, t: int) -> np.ndarray:
        if self.evolution_pattern in ["pyramid", "incremental", "decremental"]:
            if t < len(self._feature_indices_cache): return self._feature_indices_cache[t]
            return np.arange(self.d_min)
        elif self.evolution_pattern == "tds":
            return np.where(self._feature_offsets <= t)[0]
        elif self.evolution_pattern == "cds":
            rng_t = np.random.RandomState(self.random_seed + t)
            mask = rng_t.rand(self.d_max) > self.missing_ratio
            indices = np.where(mask)[0]
            if len(indices) == 0: indices = np.array([0]) 
            return indices
        elif self.evolution_pattern == "eds":
            stage_idx = 0
            for i, boundary in enumerate(self._eds_boundaries):
                if t < boundary:
                    stage_idx = i
                    break
            else: stage_idx = len(self._eds_boundaries) - 1
            if stage_idx % 2 == 0:
                return self._eds_partitions[stage_idx // 2]
            else:
                prev = (stage_idx - 1) // 2
                indices = np.concatenate([self._eds_partitions[prev], self._eds_partitions[prev + 1]])
                indices.sort()
                return indices
        return np.arange(self.d_max)

    def next_instance(self):
        if not self.has_more_instances(): return None
        base_instance = self.base_stream.next_instance()
        active_indices = self._get_active_indices(self._current_t)
        x_full = np.array(base_instance.x)
        valid_indices = active_indices[active_indices < len(x_full)]
        if len(valid_indices) == 0: valid_indices = np.array([0])
        x_subset = x_full[valid_indices]
        if self._schema.is_classification():
            new_instance = LabeledInstance.from_array(self._schema, x_subset, base_instance.y_index)
        else:
            new_instance = RegressionInstance.from_array(self._schema, x_subset, base_instance.y_value)
        new_instance.feature_indices = valid_indices
        self._current_t += 1
        return new_instance

    def has_more_instances(self) -> bool: return self._current_t < self.total_instances
    def restart(self): self.base_stream.restart(); self._current_t = 0
    def get_schema(self): return self._schema
    def get_moa_stream(self): return None

# ==========================================
# 3. 模拟与绘图逻辑
# ==========================================
class MockBaseStream(Stream):
    def __init__(self, d=20, total=1000):
        self.d = d; self.total = total; self.count = 0; self.schema = Schema(d)
    def next_instance(self):
        if self.count >= self.total: return None
        self.count += 1
        return Instance(np.ones(self.d), 0)
    def has_more_instances(self): return self.count < self.total
    def get_schema(self): return self.schema

def run_scenario(stream_params, total=100, d_max=20):
    base = MockBaseStream(d=d_max, total=total)
    ofs = OpenFeatureStream(base_stream=base, d_max=d_max, total_instances=total, **stream_params)
    mask = np.zeros((d_max, total), dtype=bool)
    t = 0
    while ofs.has_more_instances():
        inst = ofs.next_instance()
        mask[inst.feature_indices, t] = True
        t += 1
    return mask

def plot_full_demo():
    D_MAX = 20
    N = 100
    
    # 构造所有场景
    scenarios = [
        # --- Incremental ---
        ("Inc (Prefix)", {"evolution_pattern": "incremental", "d_min": 4, "feature_selection": "prefix"}),
        ("Inc (Suffix)", {"evolution_pattern": "incremental", "d_min": 4, "feature_selection": "suffix"}),
        ("Inc (Random)", {"evolution_pattern": "incremental", "d_min": 4, "feature_selection": "random"}),
        
        # --- Decremental ---
        ("Dec (Prefix)", {"evolution_pattern": "decremental", "d_min": 4, "feature_selection": "prefix"}),
        ("Dec (Suffix)", {"evolution_pattern": "decremental", "d_min": 4, "feature_selection": "suffix"}),
        ("Dec (Random)", {"evolution_pattern": "decremental", "d_min": 4, "feature_selection": "random"}),

        # --- Pyramid ---
        ("Pyr (Prefix)", {"evolution_pattern": "pyramid", "d_min": 4, "feature_selection": "prefix"}),
        ("Pyr (Suffix)", {"evolution_pattern": "pyramid", "d_min": 4, "feature_selection": "suffix"}),
        ("Pyr (Random)", {"evolution_pattern": "pyramid", "d_min": 4, "feature_selection": "random"}),

        # --- TDS (Trapezoidal) ---
        ("TDS (Ordered)", {"evolution_pattern": "tds", "tds_mode": "ordered"}),
        ("TDS (Random)",  {"evolution_pattern": "tds", "tds_mode": "random"}),
        
        # --- CDS (Capricious) ---
        ("CDS (Missing 20%)", {"evolution_pattern": "cds", "missing_ratio": 0.2}),

        # --- EDS (Evolvable) ---
        ("EDS (2 Segs, No Overlap)", {"evolution_pattern": "eds", "n_segments": 2, "overlap_ratio": 0.0}),
        ("EDS (2 Segs, Full Overlap)", {"evolution_pattern": "eds", "n_segments": 2, "overlap_ratio": 1.0}),
        ("EDS (3 Segs, Half Overlap)", {"evolution_pattern": "eds", "n_segments": 3, "overlap_ratio": 0.5}),
        
        # Placeholder
        ("OpenFeatureStream Demo", None) 
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (title, params) in enumerate(scenarios):
        ax = axes[i]
        
        if params is None:
            # 最后一个格子写标题
            ax.text(0.5, 0.5, "Demo Finished\nRun by OpenFeatureStream", 
                    ha='center', va='center', fontsize=12, color='gray')
            ax.axis('off')
            continue

        print(f"Simulating: {title}")
        mask = run_scenario(params, total=N, d_max=D_MAX)
        
        # 绘图
        ax.imshow(mask, aspect='auto', cmap='Blues', interpolation='nearest', origin='lower')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.3)
        
        # 坐标轴标签
        if i % 4 == 0: ax.set_ylabel("Feature ID")
        if i >= 12: ax.set_xlabel("Time Step")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_full_demo()