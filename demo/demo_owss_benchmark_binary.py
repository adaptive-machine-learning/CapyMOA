# SPDX-License-Identifier: BSD-3-Clause
"""
demo_owss_benchmark_binary.py
-----------------------------
Benchmark Runner - OWSS Pure Classification Edition (Binary)
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
sys.path.insert(0, os.path.abspath('./src'))

try:
    from openmoa.classifier._owss_classifier import OWSSClassifier
    from openmoa.stream.stream_wrapper import TrapezoidalStream, CapriciousStream, EvolvableStream, ShuffledStream
    import openmoa.datasets
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# === Configuration ===
N_REPEATS = 10
OUTPUT_DIR = "./experiments_owss_binary"
BURN_IN_SIZE = 500

# OWSS GNN OOM risk on RCV1
OOM_RISK_DATASETS = ["RCV1"]
TIME_SERIES_DATASETS = ["RCV1", "Electricity", "Covertype", "Bike", "Sensor"]

def get_datasets():
    target_list = [
        ("Australian", "Australian"), 
        ("Ionosphere", "Ionosphere"),
        ("German", "German"),
        ("SVMGuide3", "SVMGuide3"),
        ("Spambase", "Spambase"), 
        ("Magic04", "Magic04"),
        ("Musk", "Musk"),
        ("InternetAds", "InternetAds"),
        ("Adult", "Adult"),
        ("w8a", "W8a"),
        ("RCV1", "RCV1") 
    ]
    available = []
    for d_name, c_name in target_list:
        if hasattr(openmoa.datasets, c_name):
            available.append((d_name, getattr(openmoa.datasets, c_name)))
    return available

def get_stream_length(base_stream, default=10000):
    if hasattr(base_stream, "get_num_instances"): return base_stream.get_num_instances()
    if hasattr(base_stream, "n_instances"): return base_stream.n_instances
    return default

class AutoTuner:
    """Auto-Tuner for OWSS (Learning Rate only)."""
    def __init__(self, max_burn_in=500, safe_ratio=0.2):
        self.max_burn_in = max_burn_in
        self.safe_ratio = safe_ratio
        self.lrs = [0.001, 0.01, 0.05] # Standard LR range for Adam

    def tune(self, stream_builder_func, schema, base_seed, n_total):
        best_acc = -1.0
        best_lr = 0.01 
        
        actual_burn_in = min(self.max_burn_in, int(n_total * self.safe_ratio))
        actual_burn_in = max(50, actual_burn_in)

        for lr in self.lrs:
            stream = stream_builder_func()
            try:
                learner = OWSSClassifier(
                    schema=schema, 
                    learning_rate=lr,
                    window_size=100,
                    hidden_dim=32, 
                    random_seed=base_seed
                )
                
                correct = 0
                count = 0
                while stream.has_more_instances() and count < actual_burn_in:
                    inst = stream.next_instance()
                    if learner.predict(inst) == inst.y_index:
                        correct += 1
                    learner.train(inst)
                    count += 1
                
                acc = correct / count if count > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
            except Exception:
                continue
        return best_lr

def run_single_seed_trace(dataset_cls, wrapper_mode, seed, d_name):
    def get_base_stream():
        b = dataset_cls()
        if d_name not in TIME_SERIES_DATASETS and d_name not in OOM_RISK_DATASETS:
            return ShuffledStream(b, random_seed=seed)
        return b

    temp_stream = get_base_stream()
    n_total = get_stream_length(temp_stream)
    
    def create_wrapped_stream():
        base = get_base_stream()
        if wrapper_mode == "TDS":
            return TrapezoidalStream(base, evolution_mode="ordered", total_instances=n_total, random_seed=seed)
        elif wrapper_mode == "CDS":
            return CapriciousStream(base, missing_ratio=0.4, total_instances=n_total, random_seed=seed)
        elif wrapper_mode == "EDS":
            return EvolvableStream(base, n_segments=3, overlap_ratio=0.2, total_instances=n_total, random_seed=seed)
    
    schema = temp_stream.get_schema()
    tuner = AutoTuner(max_burn_in=BURN_IN_SIZE, safe_ratio=0.2)
    best_lr = tuner.tune(create_wrapped_stream, schema, seed, n_total)
    
    stream = create_wrapped_stream()
    learner = OWSSClassifier(
        schema=schema,
        learning_rate=best_lr,
        window_size=100,
        hidden_dim=64,
        random_seed=seed
    )
    
    log_interval = max(10, n_total // 100)
    window_size = max(50, min(1000, int(n_total * 0.2)))
    
    total_seen = 0
    total_correct = 0
    window = deque(maxlen=window_size)
    window_correct = 0
    trace_data = []
    step = 0
    
    while stream.has_more_instances():
        instance = stream.next_instance()
        if instance is None: break
        step += 1
        
        pred = learner.predict(instance)
        is_correct = (pred == instance.y_index)
        
        total_seen += 1
        if is_correct: total_correct += 1
        
        if len(window) >= window_size:
            left = window.popleft()
            if left: window_correct -= 1
        window.append(is_correct)
        if is_correct: window_correct += 1
        
        learner.train(instance)
        
        if step % log_interval == 0 or not stream.has_more_instances():
            trace_data.append({
                "Step": step,
                "PreqAcc": window_correct/len(window),
                "CumAcc": total_correct/total_seen,
                "BestLR": best_lr,
                "WindowSize": window_size
            })
            
    return pd.DataFrame(trace_data), best_lr

def plot_and_save_trace(agg_df, mode, d_name, output_dir):
    plt.figure(figsize=(10, 6))
    w_size = int(agg_df['WindowSize_mean'].iloc[0])
    avg_lr = agg_df['BestLR_mean'].iloc[0]
    
    plt.plot(agg_df['Step'], agg_df['PreqAcc_mean'], label=f'Preq Acc (w={w_size})', color='tab:blue', linewidth=2)
    plt.plot(agg_df['Step'], agg_df['CumAcc_mean'], label='Cum Acc', color='tab:orange', linewidth=2, linestyle='--')
    
    plt.fill_between(agg_df['Step'], 
                     np.maximum(0, agg_df['PreqAcc_mean'] - agg_df['PreqAcc_std']), 
                     np.minimum(1, agg_df['PreqAcc_mean'] + agg_df['PreqAcc_std']), 
                     color='tab:blue', alpha=0.15)

    plt.title(f"OWSS: {d_name} - {mode} (Avg LR={avg_lr:.4f})", fontsize=14)
    plt.xlabel("Instances Processed")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, f"plot_{mode}_{d_name}.png"), dpi=150)
    plt.close()

def main():
    datasets = get_datasets()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    modes = ["TDS", "CDS", "EDS"]
    
    print(f"🚀 Starting OWSS Benchmark (Binary - Pure)")
    
    for mode in modes:
        print(f"\n🔹 === Mode: {mode} ===")
        for d_name, d_cls in datasets:
            if d_name in OOM_RISK_DATASETS:
                print(f"{d_name:<15} | SKIPPED (Graph OOM Risk)")
                continue

            print(f"{d_name:<15} | Seeds: ", end="", flush=True)
            
            all_runs_data = []
            lrs = []
            
            start = time.time()
            for seed in range(1, N_REPEATS + 1):
                try:
                    df, lr = run_single_seed_trace(d_cls, mode, seed, d_name)
                    all_runs_data.append(df)
                    lrs.append(lr)
                    print(f"{seed}", end="", flush=True)
                except Exception as e:
                    print(f"![{e}]", end="")
            
            if not all_runs_data:
                print(" Failed.")
                continue
            
            combined = pd.concat(all_runs_data, ignore_index=True)
            agg = combined.groupby('Step').agg(['mean', 'std']).reset_index()
            agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
            
            elapsed = time.time() - start
            mode_lr = max(set(lrs), key=lrs.count) if lrs else 0
            print(f" Done ({elapsed:.1f}s) -> Best LR: {mode_lr}")
            
            agg.to_csv(os.path.join(OUTPUT_DIR, f"trace_{mode}_{d_name}.csv"), index=False)
            plot_and_save_trace(agg, mode, d_name, OUTPUT_DIR)

    print("\n✅ OWSS Benchmark Completed.")

if __name__ == "__main__":
    main()