# SPDX-License-Identifier: BSD-3-Clause
"""openmoa/classifier/_orf3v_classifier.py"""
from __future__ import annotations
from typing import Optional
import numpy as np

from openmoa.base import Classifier
from openmoa.stream import Schema
from openmoa.instance import Instance


class ORF3VClassifier(Classifier):
    """Online Random Feature Forests for Varying Feature Spaces (ORF3V).
    
    Builds independent "feature forests" for each observed feature, where each
    forest contains decision stumps. Handles dynamic feature spaces where features
    can appear and disappear over time using Hoeffding bound-based pruning.
    
    Reference:
    
    Schreckenberger, C., He, Y., Lüdtke, S., Bartelt, C., & Stuckenschmidt, H. (2023).
    Online Random Feature Forests for Learning in Varying Feature Spaces.
    AAAI Conference on Artificial Intelligence.
    
    Example:
    
    >>> from openmoa.datasets import Electricity
    >>> from openmoa.classifier import ORF3VClassifier
    >>> from openmoa.stream import EvolvingFeatureStream
    >>> from openmoa.evaluation import prequential_evaluation
    >>> 
    >>> base_stream = Electricity()
    >>> evolving_stream = EvolvingFeatureStream(
    ...     base_stream, evolution_pattern="vfs", missing_ratio=0.75
    ... )
    >>> learner = ORF3VClassifier(
    ...     schema=evolving_stream.get_schema(),
    ...     n_stumps=10,
    ...     alpha=0.1
    ... )
    >>> results = prequential_evaluation(evolving_stream, learner, max_instances=10000)
    """

    def __init__(
        self,
        schema: Schema,
        n_stumps: int = 10,
        alpha: float = 0.1,
        grace_period: int = 100,
        replacement_interval: int = 100,
        replacement_strategy: str = "oldest",
        window_size: int = 200,
        delta: float = 0.001,
        compression: float = 1000,
        d_max: int = 1000,
        enable_pruning: bool = False,
        random_seed: int = 1,
    ):
        """Initialize ORF3V Classifier.
        
        :param schema: Stream schema
        :param n_stumps: Number of decision stumps per feature forest
        :param alpha: Learning rate for weight updates
        :param grace_period: Number of instances before initializing forests
        :param replacement_interval: How often to replace stumps
        :param replacement_strategy: 'oldest' or 'random'
        :param window_size: Sliding window size for pruning
        :param delta: Hoeffding bound parameter
        :param compression: t-digest compression parameter
        :param d_max: Maximum expected features
        :param enable_pruning: Whether to enable Hoeffding bound pruning
        :param random_seed: Random seed
        """
        super().__init__(schema=schema, random_seed=random_seed)
        
        self.n_stumps = n_stumps
        self.alpha = alpha
        self.grace_period = grace_period
        self.replacement_interval = replacement_interval
        self.replacement_strategy = replacement_strategy
        self.window_size = window_size
        self.delta = delta
        self.compression = compression
        self.d_max = d_max
        self.enable_pruning = enable_pruning
        
        # Instance-level RNG only – no global seed pollution
        self.rng = np.random.RandomState(random_seed)
        
        # Feature forests: {feature_id: FeatureForest}
        self.feature_forests = {}
        
        # Feature weights
        self.weights = {}
        
        # Feature statistics: {feature_id: FeatureStats}
        self.feature_stats = {}
        
        # First occurrence of each feature
        self.first_occurrence = {}
        
        # Time counter
        self.t = 0
        
        # Hoeffding bound threshold
        self.hb = self._calc_hoeffding_bound(delta, window_size)
        
        # Number of classes
        self.n_classes = schema.get_num_classes()


    def __str__(self):
        return (f"ORF3VClassifier(n_stumps={self.n_stumps}, alpha={self.alpha}, "
                f"replacement_interval={self.replacement_interval}, "
                f"strategy={self.replacement_strategy})")

    def train(self, instance: Instance):
        """Train on a single instance."""
        self.t += 1
        
        x = np.asarray(instance.x)
        y = instance.y_index
        
        # --- [修改开始] ---
        # 尝试获取 Stream 传递过来的真实特征 ID
        # 如果没有这个属性 (普通流)，则默认使用 0, 1, 2...
        indices = getattr(instance, 'feature_indices', range(len(x)))
        
        # 使用 zip 同时遍历 (真实ID, 数值)
        # 原代码: for i, val in enumerate(x):
        for i, val in zip(indices, x):
            # 确保 i 是 int 类型 (numpy array 可能是 int64)
            feature_id = int(i) 
            
            # Update feature statistics
            if feature_id not in self.feature_stats:
                self.feature_stats[feature_id] = FeatureStats(self.n_classes, self.window_size)
                self.first_occurrence[feature_id] = self.t
                self.weights[feature_id] = 1.0
            
            self.feature_stats[feature_id].update(val, y, rng=self.rng)
        
        # Pruning check
        if self.enable_pruning and self.t > self.window_size:
            if self.t % self.replacement_interval == 0:
                self._check_pruning()
        
        # Generate feature forests after grace period
        if self.t == self.grace_period:
            self._initialize_forests()
        
        # Replace stumps periodically
        if self.t > self.grace_period and self.t % self.replacement_interval == 0:
            self._replace_stumps()
            self._generate_forests_for_new_features()
        
        # Update weights
        self._update_weights(instance)

    def predict(self, instance: Instance) -> int:
            x = np.asarray(instance.x)
            if len(self.feature_forests) == 0:
                return 0
                
            class_scores = np.zeros(self.n_classes)
            
            # --- [修改开始] ---
            indices = getattr(instance, 'feature_indices', range(len(x)))
            
            for i, feature_val in zip(indices, x):
                feature_id = int(i)
                # 只有当这个特征ID有对应的森林时才预测
                if feature_id in self.feature_forests and feature_id in self.weights:
                    forest = self.feature_forests[feature_id]
                    probs = forest.predict(feature_val)
                    for c in range(self.n_classes):
                        class_scores[c] += self.weights[feature_id] * probs.get(c, 0)
            # --- [修改结束] ---
                        
            return int(np.argmax(class_scores))

    def predict_proba(self, instance: Instance) -> np.ndarray:
            """Predict class probabilities."""
            x = np.asarray(instance.x)
            
            # 如果还没建立任何森林，返回均匀分布
            if len(self.feature_forests) == 0:
                return np.ones(self.n_classes) / self.n_classes
            
            class_scores = np.zeros(self.n_classes)
            
            # --- [修改开始] ---
            # 1. 获取 Instance 携带的真实特征 ID (如果没有则回退到 range)
            indices = getattr(instance, 'feature_indices', range(len(x)))
            
            # 2. 使用 zip 同时遍历 (真实ID, 特征值)
            # 不要使用 enumerate(x)
            for i, feature_val in zip(indices, x):
                feature_id = int(i)
                
                # 3. 只有当这个特征 ID 有对应的森林且有权重时才参与投票
                if feature_id in self.feature_forests and feature_id in self.weights:
                    forest = self.feature_forests[feature_id]
                    probs = forest.predict(feature_val)
                    
                    # 累加加权概率
                    for c in range(self.n_classes):
                        class_scores[c] += self.weights[feature_id] * probs.get(c, 0)
            # --- [修改结束] ---
            
            # 归一化 (Normalize)
            total = class_scores.sum()
            if total > 0:
                return class_scores / total
                
            # 如果总分为0 (比如所有特征都缺失或没有匹配的森林)，返回均匀分布
            return np.ones(self.n_classes) / self.n_classes

    def _initialize_forests(self):
        """Initialize feature forests after grace period."""
        for feature_id in self.feature_stats.keys():
            if feature_id not in self.feature_forests:
                stumps = self._generate_stumps_for_feature(feature_id, self.n_stumps * 2)
                if stumps:
                    # Select best stumps
                    stumps = sorted(stumps, key=lambda s: s['quality'], reverse=True)[:self.n_stumps]
                    weights = np.ones(self.n_stumps)
                    self.feature_forests[feature_id] = FeatureForest(stumps, weights)

    def _generate_stumps_for_feature(self, feature_id: int, n: int) -> list:
        """Generate decision stumps for a feature."""
        if feature_id not in self.feature_stats:
            return []
        
        stats = self.feature_stats[feature_id]
        stumps = []
        
        for _ in range(n):
            # Random split threshold
            split_val = stats.min_val + self.rng.rand() * (stats.max_val - stats.min_val)
            
            # Calculate approximate Gini impurity
            quality, class_dist_below, class_dist_above = self._calc_gini_gain(
                stats, split_val
            )
            
            stumps.append({
                'split_value': split_val,
                'class_dist_below': class_dist_below,
                'class_dist_above': class_dist_above,
                'quality': quality
            })
        
        return stumps

    def _calc_gini_gain(self, stats: 'FeatureStats', split_val: float):
        """Calculate approximate Gini gain using sample buffer."""
        count_below = np.zeros(self.n_classes)
        count_above = np.zeros(self.n_classes)
        
        for c in range(self.n_classes):
            if c in stats.class_counts and stats.class_counts[c] > 0:
                total_c = stats.class_counts[c]
                
                # 使用样本计算 CDF
                cdf_below = stats.get_cdf(c, split_val)
                cdf_above = 1 - cdf_below
                
                count_below[c] = cdf_below * total_c
                count_above[c] = cdf_above * total_c
        
        total_below = count_below.sum()
        total_above = count_above.sum()
        total = total_below + total_above
        
        if total == 0:
            return 0, {}, {}
        
        # Normalize to probabilities
        prob_below = count_below / total_below if total_below > 0 else np.zeros(self.n_classes)
        prob_above = count_above / total_above if total_above > 0 else np.zeros(self.n_classes)
        
        # Gini impurity
        gini_below = 1 - np.sum(prob_below ** 2)
        gini_above = 1 - np.sum(prob_above ** 2)
        
        # Weighted Gini gain
        gini_gain = 1 - (total_below / total * gini_below + total_above / total * gini_above)
        
        # Convert to dictionaries
        class_dist_below = {c: prob_below[c] for c in range(self.n_classes) if prob_below[c] > 0}
        class_dist_above = {c: prob_above[c] for c in range(self.n_classes) if prob_above[c] > 0}
        
        return gini_gain, class_dist_below, class_dist_above

    def _update_weights(self, instance: Instance):
        """Update feature-forest weights based on per-feature prediction accuracy.

        Uses global feature IDs (from ``feature_indices`` when available) so
        that weights are updated for the correct forest even when the physical
        vector position differs from the global ID.
        """
        x      = np.array(instance.x)
        y_true = instance.y_index
        # Mirror the same ID-resolution logic used in train() and predict()
        indices = getattr(instance, "feature_indices", range(len(x)))

        for feature_id, feature_val in zip(indices, x):
            feature_id = int(feature_id)
            if feature_id in self.feature_forests:
                forest = self.feature_forests[feature_id]
                probs  = forest.predict(feature_val)
                y_pred = max(probs, key=probs.get) if probs else 0

                correct = 1 if y_pred == y_true else 0
                self.weights[feature_id] = (
                    (2 * self.alpha * correct + self.weights[feature_id])
                    / (1 + self.alpha)
                )

    def _replace_stumps(self):
        """Replace stumps according to strategy."""
        if self.replacement_strategy == "oldest":
            self._replace_stumps_oldest()
        elif self.replacement_strategy == "random":
            self._replace_stumps_random()

    def _replace_stumps_oldest(self):
        """Replace the oldest (lowest weight) stump in each forest."""
        age_loss = 0.0001 / self.replacement_interval
        
        for feature_id, forest in self.feature_forests.items():
            # Age all stumps
            forest.weights *= (1 - age_loss)
            
            # Find oldest stump
            oldest_idx = np.argmin(forest.weights)
            
            # Generate new stumps
            new_stumps = self._generate_stumps_for_feature(feature_id, self.n_stumps)
            if new_stumps:
                best_stump = max(new_stumps, key=lambda s: s['quality'])
                forest.stumps[oldest_idx] = best_stump
                forest.weights[oldest_idx] = 1.0

    def _replace_stumps_random(self):
        """Randomly replace some stumps."""
        replacement_chance = 0.2
        
        for feature_id, forest in self.feature_forests.items():
            for i in range(len(forest.stumps)):
                if self.rng.rand() < replacement_chance:
                    new_stumps = self._generate_stumps_for_feature(feature_id, 1)
                    if new_stumps:
                        forest.stumps[i] = new_stumps[0]

    def _generate_forests_for_new_features(self):
        """Generate forests for newly observed features."""
        for feature_id in self.feature_stats.keys():
            if feature_id not in self.feature_forests:
                stumps = self._generate_stumps_for_feature(feature_id, self.n_stumps * 2)
                if stumps:
                    stumps = sorted(stumps, key=lambda s: s['quality'], reverse=True)[:self.n_stumps]
                    weights = np.ones(self.n_stumps)
                    self.feature_forests[feature_id] = FeatureForest(stumps, weights)

    def _check_pruning(self):
        """Check if any feature should be pruned using Hoeffding bound."""
        for feature_id, stats in list(self.feature_stats.items()):
            if stats.instances_seen > self.window_size:
                total_mean = 1 - stats.instances_seen / (self.t - self.first_occurrence[feature_id])
                window_mean = stats.sliding_window.get_mean()
                
                if total_mean - window_mean > self.hb:
                    # Prune this feature
                    self._prune_feature(feature_id)

    def _prune_feature(self, feature_id: int):
        """Remove a feature's forest and statistics."""
        if feature_id in self.feature_forests:
            del self.feature_forests[feature_id]
        if feature_id in self.weights:
            del self.weights[feature_id]
        if feature_id in self.feature_stats:
            del self.feature_stats[feature_id]
        if feature_id in self.first_occurrence:
            del self.first_occurrence[feature_id]

    @staticmethod
    def _calc_hoeffding_bound(delta: float, n: int) -> float:
        """Calculate Hoeffding bound."""
        return np.sqrt(np.log(1 / delta) / (2 * n))


class FeatureStats:
    """Track statistics for a single feature using sample buffer."""
    
    def __init__(self, n_classes: int, window_size: int, max_samples: int = 1000):
        self.n_classes = n_classes
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.max_samples = max_samples
        
        # 存储每个类别的最近样本值
        self.class_samples = {}
        self.class_counts = {}
        self.instances_seen = 0
        self.sliding_window = SlidingWindow(window_size)
    
    def update(self, value: float, class_label: int, rng: np.random.RandomState = None):
        """Update statistics with a new observation.

        Uses reservoir sampling (uniform random replacement) to keep a bounded
        sample buffer.  Pass *rng* to avoid touching global numpy state.
        """
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

        if class_label not in self.class_samples:
            self.class_samples[class_label] = []
            self.class_counts[class_label]  = 0

        buf = self.class_samples[class_label]
        if len(buf) >= self.max_samples:
            # Reservoir sampling: replace a random existing entry
            _rng = rng if rng is not None else np.random.RandomState()
            buf[_rng.randint(0, self.max_samples)] = value
        else:
            buf.append(value)

        self.class_counts[class_label] += 1
        self.instances_seen += 1
        self.sliding_window.add(False)

    def get_cdf(self, class_label: int, split_val: float) -> float:
        """Empirical CDF at *split_val* for *class_label* (vectorised)."""
        if class_label not in self.class_samples or not self.class_samples[class_label]:
            return 0.5
        arr = np.asarray(self.class_samples[class_label])
        return float(np.sum(arr < split_val)) / len(arr)


class SlidingWindow:
    """Sliding window for tracking feature availability."""
    
    def __init__(self, size: int):
        self.size = size
        self.window = np.zeros(size, dtype=bool)
        self.pos = 0
        self.count = 0
    
    def add(self, missing: bool):
        """Add observation (True if missing, False if present)."""
        self.window[self.pos] = missing
        self.pos = (self.pos + 1) % self.size
        self.count = min(self.count + 1, self.size)
    
    def get_mean(self) -> float:
        """Get mean of window (proportion missing)."""
        if self.count == 0:
            return 0
        return np.mean(self.window[:self.count])


class FeatureForest:
    """Ensemble of decision stumps for one feature."""
    
    def __init__(self, stumps: list, weights: np.ndarray):
        self.stumps = stumps
        self.weights = weights
    
    def predict(self, feature_value: float) -> dict:
        """Predict class distribution."""
        class_scores = {}
        total_weight = 0
        
        for stump, weight in zip(self.stumps, self.weights):
            if feature_value < stump['split_value']:
                dist = stump['class_dist_below']
            else:
                dist = stump['class_dist_above']
            
            for class_label, prob in dist.items():
                if class_label not in class_scores:
                    class_scores[class_label] = 0
                class_scores[class_label] += weight * prob
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for c in class_scores:
                class_scores[c] /= total_weight
        
        return class_scores