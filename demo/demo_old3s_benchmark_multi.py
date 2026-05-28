# SPDX-License-Identifier: BSD-3-Clause
"""
demo_old3s_benchmark_multi.py
----------------------------------
The ULTIMATE OpenMOA Benchmark Runner - OLD3S Multi-class Edition (Fixed)
-------------------------------------------------------------------------
Target: Multi-class Classification (OLD3S)
Protocol: 10 Repeats, Auto-Tuning, Shuffling, Adaptive Logging.
Special: 
1. ONLY runs EDS mode (S1 -> S2 transition).
2. Native Multi-class support.
3. [FIX] Adapted to new OLD3S (Lifelong) API: No manual switch points needed.
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
import torch

sns.set_theme(style="whitegrid")
sys.path.insert(0, os.path.abspath('./src'))

try:
    from openmoa.classifier._old3s_classifier import OLD3SClassifier
    from openmoa.stream.stream_wrapper import OpenFeatureStream, ShuffledStream
    import openmoa.datasets
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# === Configuration ===
N_REPEATS = 10
OUTPUT_DIR = "./experiments_old3s_multiclass"
BURN_IN_SIZE = 500

# Deep Learning on Covertype (581k samples) is slow on CPU. 
# Skip by default. Remove from list to run.
SLOW_DATASETS = ["Covertype", "Covtype"]

# Datasets that MUST retain temporal order
TIME_SERIES_DATASETS = ["Covertype", "Covtype", "Electricity"]

def get_datasets():
    target_list = [
        ("DryBean", "DryBean"),
        ("Optdigits", "Optdigits"),
        ("Frogs", "Frogs"),
        ("Wine", "Wine"),
        ("Splice", "Splice"),
        ("Covertype", "Covtype") 
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
    """Adaptive Auto-Tuner for OLD3S Learning Rate."""
    def __init__(self, max_burn_in=500, safe_ratio=0.2):
        self.max_burn_in = max_burn_in
        self.safe_ratio = safe_ratio
        # Tune Learning Rate
        self.lrs = [0.0001, 0.001, 0.01]

    def tune(self, stream_builder_func, schema, base_seed, n_total):
        best_acc = -1.0
        best_lr = 0.001 
        
        actual_burn_in = min(self.max_burn_in, int(n_total * self.safe_ratio))
        actual_burn_in = max(50, actual_burn_in)

        for lr in self.lrs:
            stream = stream_builder_func()
            try:
                # [FIX] New API: Just learning rate and dimensions
                learner = OLD3SClassifier(
                    schema=schema,
                    learning_rate=lr,
                    hidden_dim=64, # Smaller for speed in tuning
                    latent_dim=16,
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
    if wrapper_mode != "EDS":
        raise ValueError("OLD3S only supports EDS mode")

    # 1. Base Stream Setup
    def get_base_stream():
        b = dataset_cls()
        if d_name not in TIME_SERIES_DATASETS:
            return ShuffledStream(b, random_seed=seed)
        return b

    temp_stream = get_base_stream()
    n_total = get_stream_length(temp_stream)
    
    # 2. Calculate EDS Parameters (S1 -> S2)
    # Still need to setup the STREAM with S1/S2 logic, but MODEL figures it out.
    n_segments = 2
    overlap_ratio = 0.2
    
    # Note: OLD3S Auto-detects, so we don't need to pass s1_idx/s2_idx to it anymore.
    # But we still construct the stream with them.

    def create_wrapped_stream():
        base = get_base_stream()
        return OpenFeatureStream(base, evolution_pattern="eds", 
                                 n_segments=n_segments, overlap_ratio=overlap_ratio, 
                                 total_instances=n_total, random_seed=seed)

    schema = temp_stream.get_schema()
    
    # 3. Auto-Tuning
    tuner = AutoTuner(max_burn_in=BURN_IN_SIZE, safe_ratio=0.2)
    # Tuner signature simplified (removed manual indices args)
    best_lr = tuner.tune(create_wrapped_stream, schema, seed, n_total)
    
    # 4. Final Run
    stream = create_wrapped_stream()
    
    # [FIX] New API Initialization
    learner = OLD3SClassifier(
        schema=schema,
        learning_rate=best_lr,
        hidden_dim=128,
        latent_dim=32,
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

    plt.title(f"OLD3S: {d_name} - {mode} (Avg LR={avg_lr:.4f})", fontsize=14)
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
    
    modes = ["EDS"]
    
    print(f"🚀 Starting OLD3S Multi-class Benchmark")
    
    for mode in modes:
        print(f"\n🔹 === Mode: {mode} ===")
        for d_name, d_cls in datasets:
            if d_name in SLOW_DATASETS:
                print(f"{d_name:<15} | SKIPPED (Slow NN on CPU)")
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
            avg_lr = np.mean(lrs) if lrs else 0
            print(f" Done ({elapsed:.1f}s) -> Best LR: {avg_lr:.4f}")
            
            agg.to_csv(os.path.join(OUTPUT_DIR, f"trace_{mode}_{d_name}.csv"), index=False)
            plot_and_save_trace(agg, mode, d_name, OUTPUT_DIR)

    print("\n✅ OLD3S Multi-class Benchmark Completed.")

if __name__ == "__main__":
    main()