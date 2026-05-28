# SPDX-License-Identifier: BSD-3-Clause
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
_src_path = os.path.join(_project_root, 'src')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

from capymoa.stream import Schema
from capymoa.instance import LabeledInstance, Instance
from openmoa.classifier import PLASTIC  # type: ignore[import]

class PlasticModel:
    def __init__(self, n_features: int, n_classes: int, seed: int = 42):
        schema = Schema.from_custom(
            feature_names=[f"f{i}" for i in range(n_features)],
            values_for_class_label=[str(i) for i in range(n_classes)],
            dataset_name="experiment",
        )
        self.schema = schema
        self.n_classes = n_classes
        self._seed = seed
        self.model = PLASTIC(
            schema=schema,
            grace_period=400,        # 默认200，降低分裂频率
            split_confidence=0.05,   # 默认1e-7，大幅放宽分裂置信度
            max_depth=6,             # 默认20，限制树深度
            leaf_prediction="NB",   # 默认NBA，退回普通NB
        )

    def train_on_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        for i in range(len(X)):
            instance = LabeledInstance.from_array(
                self.schema,
                X[i].astype(np.float64),
                int(y[i]),
            )
            self.model.train(instance)

    def predict_on_batch(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for i in range(len(X)):
            instance = Instance.from_array(
                self.schema,
                X[i].astype(np.float64),
            )
            pred = self.model.predict(instance)
            predictions.append(int(pred) if pred is not None else 0)
        return np.array(predictions, dtype=np.int64)

class ExperimentRunner:
    def __init__(self, model_class, model_kwargs=None, sample_limit=18000, seed=42):
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.sample_limit = sample_limit
        self.seed = seed
        np.random.seed(seed)

    def run_on_dataset(self, train_stream, test_stream, dataset_name):
        first_X, first_y = train_stream[0]
        n_features = first_X.shape[1]
        all_labels = np.concatenate([y for _, y in train_stream])
        n_classes = len(np.unique(all_labels))

        model = self.model_class(
            n_features=n_features,
            n_classes=n_classes,
            seed=self.seed,
            **self.model_kwargs
        )

        batch_accuracies = []
        total_samples = 0
        n_batches = len(train_stream)

        for batch_idx, ((train_X, train_y), (test_X, test_y)) in \
                enumerate(zip(train_stream, test_stream)):

            model.train_on_batch(train_X, train_y)
            predictions = model.predict_on_batch(test_X)
            batch_acc = accuracy_score(test_y, predictions)
            batch_accuracies.append(float(batch_acc))
            total_samples += len(test_X)

        overall_accuracy = float(np.mean(batch_accuracies))

        return {
            'dataset_name': dataset_name,
            'batch_accuracies': batch_accuracies,
            'overall_accuracy': overall_accuracy,
            'n_batches': n_batches,
            'n_samples': total_samples,
        }

    def run_experiments(self, selected_datasets=None):
        available_datasets = {
            'simple_mixed': self._generate_simple_mixed,
            'medium_mixed': self._generate_medium_mixed,
            'complex_mixed': self._generate_complex_mixed,
            'sea_concepts': self._generate_sea_concepts,
            'hyperplane_drift': self._generate_hyperplane_drift,
            'weather_forecast': self._generate_weather_forecast,
            'electricity_market': self._generate_electricity_market,
            'airlines_dataset': self._generate_airlines_dataset,
            'credit_fraud_dataset': self._generate_credit_fraud_dataset,
            'simple_longtail': self._generate_simple_longtail,
            'mixed_longtail': self._generate_mixed_longtail,
            'dynamic_longtail': self._generate_dynamic_longtail,
        }

        datasets_to_run = (
            available_datasets if selected_datasets is None
            else {k: v for k, v in available_datasets.items()
                  if k in selected_datasets}
        )

        all_results = {}

        for dataset_name, gen_func in datasets_to_run.items():
            try:
                train_stream, test_stream = gen_func()
                result = self.run_on_dataset(train_stream, test_stream, dataset_name)
                all_results[dataset_name] = result
            except Exception as e:
                import traceback
                traceback.print_exc()

        if all_results:
            self._save_results(all_results)

        return all_results

    def _save_results(self, results, output_dir='plastic_results'):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self.model_class.__name__.lower()

        batch_rows = []
        for dataset_name, r in results.items():
            for batch_idx, acc in enumerate(r['batch_accuracies']):
                batch_rows.append({
                    'dataset': dataset_name,
                    'batch': batch_idx + 1,
                    'accuracy': acc,
                })
        batch_file = os.path.join(output_dir, f'{model_name}_batch_results_{ts}.csv')
        pd.DataFrame(batch_rows).to_csv(batch_file, index=False)

        summary_rows = []
        for dataset_name, r in results.items():
            accs = r['batch_accuracies']
            summary_rows.append({
                'dataset': dataset_name,
                'overall_accuracy': r['overall_accuracy'],
                'std_accuracy': float(np.std(accs)),
                'min_accuracy': float(min(accs)),
                'max_accuracy': float(max(accs)),
                'n_batches': r['n_batches'],
                'n_samples': r['n_samples'],
            })
        summary_file = os.path.join(output_dir, f'{model_name}_summary_{ts}.csv')
        pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    def _convert_to_streaming_format(self, X, y, batch_size=1000):
        n_batches = len(X) // batch_size
        train_stream, test_stream = [], []

        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(X))
            batch_X = X[start:end]
            batch_y = y[start:end]
            split = int(0.8 * len(batch_X))
            train_stream.append((batch_X[:split], batch_y[:split]))
            test_stream.append((batch_X[split:], batch_y[split:]))

        return train_stream, test_stream

    def _create_mixed_features(self, X, binary_cols=None, categorical_cols=None):
        X = X.copy()
        if binary_cols:
            for col in binary_cols:
                X[:, col] = (X[:, col] > np.median(X[:, col])).astype(int)
        if categorical_cols:
            for col in categorical_cols:
                levels = 3 + (col % 3)
                quantiles = np.linspace(0, 1, levels + 1)
                thresholds = np.quantile(X[:, col], quantiles[1:-1])
                X[:, col] = np.clip(np.searchsorted(thresholds, X[:, col]),
                                    0, levels - 1)
        return X

    def _generate_simple_mixed(self):
        X, y = make_classification(
            n_samples=15000, n_features=8, n_classes=2, n_informative=5,
            n_redundant=2, n_clusters_per_class=2, class_sep=0.6, random_state=42
        )
        X = self._create_mixed_features(X, binary_cols=[0, 1], categorical_cols=[2, 3])
        return self._convert_to_streaming_format(X, y, batch_size=750)

    def _generate_medium_mixed(self):
        X, y = make_classification(
            n_samples=18000, n_features=20, n_classes=3, n_informative=12,
            n_redundant=4, n_clusters_per_class=3, class_sep=0.45, random_state=123
        )
        X = self._create_mixed_features(
            X, binary_cols=[0, 1, 2], categorical_cols=[3, 4, 5, 6]
        )
        return self._convert_to_streaming_format(X, y, batch_size=900)

    def _generate_complex_mixed(self):
        X, y = make_classification(
            n_samples=20000, n_features=28, n_classes=4, n_informative=14,
            n_redundant=6, n_clusters_per_class=4, class_sep=0.3, random_state=456
        )
        X = self._create_mixed_features(
            X, binary_cols=list(range(5)), categorical_cols=list(range(5, 12))
        )
        return self._convert_to_streaming_format(X, y, batch_size=500)

    def _generate_sea_concepts(self):
        try:
            from river.datasets import synth
            dataset = synth.SEA(variant=3, noise=0.88, seed=42)
            X_all, y_all = [], []
            for x, y in dataset.take(16000):
                X_all.append([x[k] for k in sorted(x.keys())])
                y_all.append(int(y))
            X = self._create_mixed_features(
                np.array(X_all), binary_cols=[], categorical_cols=[]
            )
            return self._convert_to_streaming_format(X, np.array(y_all), batch_size=400)
        except ImportError:
            return self._generate_sea_fallback()

    def _generate_sea_fallback(self):
        n = 16000
        X = np.random.uniform(0, 10, (n, 4))
        y = np.array([
            1 if (X[i, 0] + X[i, 1]) > 6.0 else 0 if i < n // 2
            else 1 if (X[i, 0] + X[i, 2]) > 8.0 else 0
            for i in range(n)
        ])
        noise = np.random.choice(n, int(0.32 * n), replace=False)
        y[noise] = 1 - y[noise]
        return self._convert_to_streaming_format(X, y, batch_size=800)

    def _generate_hyperplane_drift(self):
        try:
            from river.datasets import synth
            dataset = synth.Hyperplane(n_features=10, n_drift_features=4, seed=123)
            X_all, y_all = [], []
            for x, y in dataset.take(18000):
                X_all.append([x[k] for k in sorted(x.keys())])
                y_all.append(int(y))
            return self._convert_to_streaming_format(
                np.array(X_all), np.array(y_all), batch_size=400
            )
        except ImportError:
            return self._generate_hyperplane_fallback()

    def _generate_hyperplane_fallback(self):
        n = 18000
        X = np.random.uniform(-1, 1, (n, 10))
        w0 = np.array([1, 0.5, -0.8, 0.3, -0.6, 0.4, -0.2, 0.7, -0.9, 0.1])
        w1 = np.array([-0.5, 1, 0.3, -0.8, 0.2, -0.7, 0.9, -0.4, 0.6, -0.3])
        y = np.array([
            1 if np.dot(X[i], w0 * (1 - i / n) + w1 * (i / n)) > 0 else 0
            for i in range(n)
        ])
        noise = np.random.choice(n, int(0.35 * n), replace=False)
        y[noise] = 1 - y[noise]
        return self._convert_to_streaming_format(X, y, batch_size=900)

    def _generate_weather_forecast(self):
        n = min(self.sample_limit, 20000)
        t = np.linspace(0, 4 * np.pi, n)
        rng = np.random.default_rng(42)
        mintemp = 10 + 15 * np.sin(t) + rng.normal(0, 3, n)
        maxtemp = mintemp + 10 + rng.normal(0, 2, n)
        hum9am = np.clip(70 - 20 * np.sin(t) + rng.normal(0, 10, n), 0, 100)
        hum3pm = np.clip(hum9am - 5 + rng.normal(0, 8, n), 0, 100)
        prs9am = 1013 + 10 * np.sin(t * 0.7) + rng.normal(0, 5, n)
        prs3pm = prs9am + rng.normal(0, 3, n)
        wind9am = np.abs(rng.normal(15, 8, n))
        wind3pm = np.abs(rng.normal(18, 10, n))
        cloud9am = np.clip(rng.beta(2, 5, n) * 9, 0, 9)
        cloud3pm = np.clip(cloud9am + rng.normal(0, 1, n), 0, 9)
        sunshine = np.clip(12 - cloud9am + rng.normal(0, 2, n), 0, 14)
        evap = np.clip(rng.gamma(2, 2, n), 0, 20)
        rainfall = np.clip(rng.exponential(2, n), 0, 50)
        raintoday = rng.binomial(1, 0.3 + 0.2 * np.sin(t), n)
        X = np.column_stack([
            mintemp, maxtemp, hum9am, hum3pm,
            prs9am, prs3pm, wind9am, wind3pm,
            cloud9am, cloud3pm, sunshine, evap, rainfall, raintoday,
            (maxtemp - mintemp)
        ])
        shift = t / (4 * np.pi)
        prob = (0.25
                + (hum3pm - 70) / 100 * (0.2 + 0.15 * shift)
                + (1013 - prs3pm) / 60 * (0.15 + 0.1 * np.sin(shift * 3 * np.pi))
                + raintoday * (0.08 + 0.2 * shift))
        prob += rng.normal(0, 0.18, n)   # 额外噪声
        prob = np.clip(prob, 0.1, 0.9)   # 扩宽边界，让边界样本更难分
        y = rng.binomial(1, prob).astype(np.int32)
        return self._convert_to_streaming_format(X.astype(np.float32), y, batch_size=600)

    def _generate_electricity_market(self):
        n = min(self.sample_limit, 20000)
        t = np.linspace(0, 8 * np.pi, n)
        rng = np.random.default_rng(42)
        day = (np.arange(n) % 7 > 4).astype(int)
        period = np.clip(((np.sin(t * 24) + 0.5) * 12 + 12).astype(int), 1, 24)
        nswprice = 20 + 15 * np.sin(t / 4) + rng.normal(0, 5, n)
        nswdemand = (6000 + 1000 * np.sin(t / 2)
                     + (nswprice - 20) * 50 - 500 * day
                     + rng.normal(0, 200, n))
        vicprice = 25 + 10 * np.sin(t / 3 + np.pi / 4) + rng.normal(0, 3, n)
        vicdemand = (5500 + 800 * np.sin(t / 2.5)
                     + (vicprice - 25) * 40 - 400 * day
                     + rng.normal(0, 180, n))
        transfer = (nswdemand - vicdemand) * 0.1 + rng.normal(0, 50, n)
        X = np.column_stack([day, period, nswprice, nswdemand,
                             vicprice, vicdemand, transfer,
                             nswprice - vicprice])
        shift = t / (8 * np.pi)
        prob = (0.5
                + (nswprice > 25) * (0.2 + 0.1 * shift)
                + (nswdemand > 6500) * (0.15 + 0.1 * np.sin(shift * 2 * np.pi)))
        prob += rng.normal(0, 0.18, n)   # 额外噪声
        prob = np.clip(prob, 0.1, 0.9)
        y = rng.binomial(1, prob).astype(np.int32)
        return self._convert_to_streaming_format(X.astype(np.float32), y, batch_size=400)

    def _generate_airlines_dataset(self):
        n = 20000
        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.integers(1, 8, n),
            rng.integers(1, 13, n),
            rng.normal(12, 4, n),
            rng.exponential(1000, n),
            rng.integers(1, 6, n),
            rng.gamma(2, 5, n),
            rng.beta(2, 2, n),
            rng.uniform(0.3, 1.0, n),
        ]).astype(np.float32)
        X = MinMaxScaler().fit_transform(X)
        X = self._create_mixed_features(X, binary_cols=[0, 1], categorical_cols=[2, 3])
        drift = np.arange(n) / n
        prob = np.where(
            drift < 0.5,
            0.3 * (1 - X[:, 6]) + 0.2 * (X[:, 3] > 0.7) + 0.1 * (X[:, 0] > 0.6),
            0.4 * (X[:, 4] > 0.6) + 0.3 * X[:, 7] + 0.2 * (X[:, 5] > 0.5)
        )
        prob += rng.normal(0, 0.15, n)   # 额外噪声
        prob = np.clip(prob, 0.05, 0.95)
        y = (rng.uniform(0, 1, n) < prob).astype(np.int32)
        return self._convert_to_streaming_format(X, y, batch_size=500)

    def _generate_credit_fraud_dataset(self):
        n = 20000
        d = 15
        rng = np.random.default_rng(42)
        n_normal = int(n * 0.99)
        n_fraud = n - n_normal
        X_normal = rng.multivariate_normal(np.zeros(d), np.eye(d) * 1.5, n_normal)
        X_fraud = rng.multivariate_normal(np.ones(d) * 1.0, np.eye(d) * 1.5, n_fraud)
        X = np.vstack([X_normal, X_fraud]).astype(np.float32)
        y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)]).astype(np.int32)
        idx = rng.permutation(n)
        X, y = X[idx], y[idx]
        X = MinMaxScaler().fit_transform(X)
        X = self._create_mixed_features(
            X, binary_cols=[0, 1], categorical_cols=[2, 3, 4]
        )
        return self._convert_to_streaming_format(X, y, batch_size=400)

    def _generate_simple_longtail(self):
        n = min(self.sample_limit, 24000)
        d = 20
        rng = np.random.default_rng(self.seed)
        X = np.zeros((n, d))
        for i in range(d):
            mask = rng.random(n) < (0.9 if i < 12 else 0.7)
            if i < 8:
                X[mask, i] = rng.uniform(0.85, 1.00, mask.sum())
                X[~mask, i] = rng.uniform(0.00, 0.85, (~mask).sum())
            elif i < 12:
                X[mask, i] = rng.uniform(0.00, 0.15, mask.sum())
                X[~mask, i] = rng.uniform(0.15, 1.00, (~mask).sum())
            else:
                X[mask, i] = rng.uniform(0.6, 0.8, mask.sum())
                X[~mask, i] = rng.uniform(0.0, 1.0, (~mask).sum())
        y = self._generate_longtail_target(X, 'simple', rng=rng)
        return self._convert_to_streaming_format(X, y, batch_size=600)

    def _generate_mixed_longtail(self):
        n = min(self.sample_limit, 18000)
        d = 18
        rng = np.random.default_rng(self.seed)
        X = np.zeros((n, d))
        for i in range(d):
            if i < 6:
                mask = rng.random(n) < 0.95
                X[mask, i] = rng.uniform(0.9, 1.0, mask.sum())
                X[~mask, i] = rng.uniform(0.0, 0.9, (~mask).sum())
            elif i < 12:
                mask = rng.random(n) < 0.95
                X[mask, i] = rng.uniform(0.0, 0.1, mask.sum())
                X[~mask, i] = rng.uniform(0.1, 1.0, (~mask).sum())
            elif i < 15:
                X[:, i] = rng.beta(0.3, 3, n)
            else:
                raw = rng.pareto(0.5, n) + 1
                X[:, i] = (raw - 1) / (raw.max() - 1)
        y = self._generate_longtail_target(X, 'mixed', rng=rng)
        return self._convert_to_streaming_format(X, y, batch_size=750)

    def _generate_dynamic_longtail(self):
        n = self.sample_limit
        d = 20
        rng = np.random.default_rng(self.seed)
        t = np.linspace(0, 4 * np.pi, n)
        X = np.zeros((n, d))
        for feat in range(d):
            tf = (np.sin(t + feat * np.pi / 4) + 1) / 2
            if feat < 10:
                center = 0.1 + 0.8 * tf
                concentrated = rng.random(n) < 0.85
                X[concentrated, feat] = np.clip(
                    rng.normal(center[concentrated], 0.05), 0, 1
                )
                X[~concentrated, feat] = rng.uniform(0, 1, (~concentrated).sum())
            else:
                right_tail = tf < 0.5
                high_prob = rng.random(n) < 0.9
                X[right_tail & high_prob, feat] = rng.uniform(0.8, 1.0, (right_tail & high_prob).sum())
                X[right_tail & ~high_prob, feat] = rng.uniform(0.0, 0.8, (right_tail & ~high_prob).sum())
                X[~right_tail & high_prob, feat] = rng.uniform(0.0, 0.2, (~right_tail & high_prob).sum())
                X[~right_tail & ~high_prob, feat] = rng.uniform(0.2, 1.0, (~right_tail & ~high_prob).sum())
        y = self._generate_longtail_target(X, 'dynamic', time_progression=t, rng=rng)
        return self._convert_to_streaming_format(X, y, batch_size=600)

    def _generate_longtail_target(self, X, target_type, time_progression=None, rng=None):
        if rng is None:
            rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        prob = np.full(n, 0.5)
        if target_type == 'simple':
            for j in range(8):
                prob += 0.15 * (X[:, j] < 0.3)
            for j in range(8, 12):
                prob += 0.12 * (X[:, j] > 0.7)
            prob += 0.2 * ((X[:, 0] > 0.9) & (X[:, 8] < 0.1))
        elif target_type == 'mixed':
            eh = (X[:, :6] > 0.95).sum(axis=1)
            el = (X[:, 6:12] < 0.05).sum(axis=1)
            prob += 0.20 * (eh >= 2)
            prob += 0.15 * (el >= 2)
            prob += 0.25 * ((eh >= 1) & (el >= 1))
            prob += 0.10 * ((X[:, 12:15] > 0.8).sum(axis=1) >= 2)
        elif target_type == 'dynamic':
            tf = ((np.sin(time_progression) + 1) / 2
                  if time_progression is not None
                  else np.full(n, 0.5))
            early = tf < 0.3
            late = tf > 0.7
            middle = ~early & ~late
            prob[early] += 0.10 * (X[early, :5] > 0.95).sum(axis=1)
            prob[late] += 0.10 * (X[late, 5:10] < 0.05).sum(axis=1)
            interaction = ((X[:, :5] > 0.9).sum(axis=1)
                           + (X[:, 5:10] < 0.1).sum(axis=1))
            prob[middle] += 0.15 * (interaction[middle] >= 3)
        prob += rng.normal(0, 0.2, n)
        prob = np.clip(prob, 0.1, 0.9)
        return rng.binomial(1, prob).astype(np.int32)

def main():
    run_mode = 1 
    runner = ExperimentRunner(
        model_class=PlasticModel,
        model_kwargs={},
        sample_limit=18000,
        seed=42
    )
    if run_mode == 1:
        results = runner.run_experiments()
    elif run_mode == 2:
        results = runner.run_experiments(selected_datasets=[
            'simple_mixed',
            'weather_forecast',
            'simple_longtail',
        ])
    elif run_mode == 3:
        results = runner.run_experiments(selected_datasets=['simple_mixed'])

    print(f"\n{'='*60}")
    print(f"{'Dataset':<25} {'Accuracy':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*60}")
    for name, r in results.items():
        accs = r['batch_accuracies']
        print(f"{name:<25} {r['overall_accuracy']:>10.4f} "
              f"{np.std(accs):>8.4f} {min(accs):>8.4f} {max(accs):>8.4f}")
    print(f"{'='*60}")
    print(f"Mean overall accuracy: {np.mean([r['overall_accuracy'] for r in results.values()]):.4f}")
    return results

if __name__ == "__main__":
    main()