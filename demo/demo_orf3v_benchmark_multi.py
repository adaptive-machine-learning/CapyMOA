# SPDX-License-Identifier: BSD-3-Clause
"""
demo_orf3v_benchmark_multiclass.py
----------------------------------
The ULTIMATE OpenMOA Benchmark Runner - ORF3V Multi-class Edition
-----------------------------------------------------------------
Target: Multi-class Classification (ORF3V)
Protocol Alignment: Matches Binary Edition exactly.
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
    from openmoa.classifier._orf3v_classifier import ORF3VClassifier
    from openmoa.stream.stream_wrapper import OpenFeatureStream, ShuffledStream
    import openmoa.datasets
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# === Configuration ===
N_REPEATS = 10
OUTPUT_DIR = "./experiments_orf3v_multiclass"
BURN_IN_SIZE = 500

TIME_SERIES_DATASETS = ["Covertype", "Covtype", "Electricity"]

# Skipping very high dimensional datasets if they appear here
SLOW_DATASETS = [] 

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
    def __init__(self, max_burn_in=500, safe_ratio=0.2):
        self.max_burn_in = max_burn_in
        self.safe_ratio = safe_ratio
        # Tuning configs
        self.configs = [
            {'alpha': 0.1, 'stumps': 5},
            {'alpha': 0.5, 'stumps': 5},
            {'alpha': 0.1, 'stumps': 10},
        ]

    def tune(self, stream_builder_func, schema, base_seed, n_total):
        best_acc = -1.0
        best_cfg = self.configs[0]
        
        actual_burn_in = min(self.max_burn_in, int(n_total * self.safe_ratio))
        actual_burn_in = max(10, actual_burn_in)

        for cfg in self.configs:
            stream = stream_builder_func()
            
            learner = ORF3VClassifier(
                schema=schema, 
                alpha=cfg['alpha'], 
                n_stumps=cfg['stumps'],
                grace_period=50,
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
                best_cfg = cfg
                
        return best_cfg

def run_single_seed_trace(dataset_cls, wrapper_mode, seed, d_name):
    
    def get_base_stream():
        b = dataset_cls()
        if d_name not in TIME_SERIES_DATASETS:
            return ShuffledStream(b, random_seed=seed)
        return b

    temp_stream = get_base_stream()
    n_total = get_stream_length(temp_stream)
    
    def create_wrapped_stream():
        base = get_base_stream()
        if wrapper_mode == "TDS":
            return OpenFeatureStream(base, evolution_pattern="tds", tds_mode="ordered", 
                                     d_min=2, total_instances=n_total, random_seed=seed)
        elif wrapper_mode == "CDS":
            return OpenFeatureStream(base, evolution_pattern="cds", missing_ratio=0.4, 
                                     total_instances=n_total, random_seed=seed)
        elif wrapper_mode == "EDS":
            return OpenFeatureStream(base, evolution_pattern="eds", n_segments=3, 
                                     overlap_ratio=0.2, total_instances=n_total, random_seed=seed)
    
    schema = temp_stream.get_schema()
    tuner = AutoTuner(max_burn_in=BURN_IN_SIZE, safe_ratio=0.2)
    best_cfg = tuner.tune(create_wrapped_stream, schema, seed, n_total)
    
    stream = create_wrapped_stream()
    learner = ORF3VClassifier(
        schema=schema,
        alpha=best_cfg['alpha'],
        n_stumps=best_cfg['stumps'],
        grace_period=50,
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
            left_out = window.popleft()
            if left_out: window_correct -= 1
        window.append(is_correct)
        if is_correct: window_correct += 1
        
        learner.train(instance)
        
        if step % log_interval == 0 or not stream.has_more_instances():
            curr_preq = window_correct / len(window) if window else 0.0
            curr_cum = total_correct / total_seen if total_seen > 0 else 0.0
            
            trace_data.append({
                "Step": step,
                "PreqAcc": curr_preq,
                "CumAcc": curr_cum,
                "BestAlpha": best_cfg['alpha'],
                "WindowSize": window_size
            })
            
    return pd.DataFrame(trace_data), best_cfg['alpha']

def plot_and_save_trace(agg_df, mode, d_name, output_dir):
    plt.figure(figsize=(10, 6))
    w_size = int(agg_df['WindowSize_mean'].iloc[0])
    avg_alpha = agg_df['BestAlpha_mean'].iloc[0]
    
    plt.plot(agg_df['Step'], agg_df['PreqAcc_mean'], label=f'Preq Acc (w={w_size})', color='tab:blue', linewidth=2)
    plt.plot(agg_df['Step'], agg_df['CumAcc_mean'], label='Cum Acc', color='tab:orange', linewidth=2, linestyle='--')
    
    plt.fill_between(agg_df['Step'], 
                     np.maximum(0, agg_df['PreqAcc_mean'] - agg_df['PreqAcc_std']), 
                     np.minimum(1, agg_df['PreqAcc_mean'] + agg_df['PreqAcc_std']), 
                     color='tab:blue', alpha=0.15)

    plt.title(f"ORF3V: {d_name} - {mode} (Avg $\\alpha$={avg_alpha:.2f})", fontsize=14)
    plt.xlabel("Instances Processed")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_filename = os.path.join(output_dir, f"plot_{mode}_{d_name}.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()

def main():
    datasets = get_datasets()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    modes = ["TDS", "CDS", "EDS"]
    
    print(f"🚀 Starting ORF3V Multi-class Benchmark")
    
    for mode in modes:
        print(f"\n🔹 === Mode: {mode} ===")
        for d_name, d_cls in datasets:
            if d_name in SLOW_DATASETS:
                print(f"{d_name:<15} | SKIPPED (Too slow)")
                continue
                
            print(f"{d_name:<15} | Seeds: ", end="", flush=True)
            
            all_runs_data = []
            alphas_selected = []
            
            start = time.time()
            for seed in range(1, N_REPEATS + 1):
                try:
                    df, alpha = run_single_seed_trace(d_cls, mode, seed, d_name)
                    all_runs_data.append(df)
                    alphas_selected.append(alpha)
                    print(f"{seed}", end="", flush=True)
                except Exception as e:
                    print(f"![{e}]", end="")
            
            if not all_runs_data:
                print(" Failed.")
                continue
            
            combined = pd.concat(all_runs_data, ignore_index=True)
            agg = combined.groupby('Step').agg({
                'PreqAcc': ['mean', 'std'],
                'CumAcc': ['mean', 'std'],
                'BestAlpha': ['mean'],
                'WindowSize': ['mean']
            }).reset_index()
            agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
            
            elapsed = time.time() - start
            
            if alphas_selected:
                mode_alpha = max(set(alphas_selected), key=alphas_selected.count) 
            else:
                mode_alpha = 0.0
            print(f" Done ({elapsed:.1f}s) -> Best α: {mode_alpha}")
            
            csv_filename = os.path.join(OUTPUT_DIR, f"trace_{mode}_{d_name}.csv")
            agg.to_csv(csv_filename, index=False)
            plot_and_save_trace(agg, mode, d_name, OUTPUT_DIR)

    print("\n✅ ORF3V Multi-class Benchmark Completed.")

if __name__ == "__main__":
    main()