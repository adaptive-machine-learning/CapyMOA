# SPDX-License-Identifier: BSD-3-Clause
"""
demo_fesl_benchmark_binary.py
-----------------------------
The ULTIMATE OpenMOA Benchmark Runner - FESL Edition
----------------------------------------------------
Target: Binary Classification (FESL)
Protocol Alignment (Strict Match to FOBOS/FTRL):
1. [Output] 33 Plots + 33 CSVs (Preq & Cum Accuracy ONLY).
2. [Exp] 10 Repeats (Mean +/- Std).
3. [Data] Smart Shuffling for static datasets.
4. [Tuning] Adaptive Auto-Tuning for Alpha (Learning Rate).
5. [Safety] RCV1 is automatically skipped for FESL to prevent OOM (Matrix Inversion).
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from collections import deque
import matplotlib
matplotlib.use('Agg') # Safe for server/background running
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
sys.path.insert(0, os.path.abspath('./src'))

try:
    # [CHANGE] Import FESL
    from openmoa.classifier._fesl_classifier import FESLClassifier
    from openmoa.stream.stream_wrapper import OpenFeatureStream, ShuffledStream
    import openmoa.datasets
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# === Configuration ===
N_REPEATS = 10           # 10 Runs for robustness
OUTPUT_DIR = "./experiments_fesl_binary"
BURN_IN_SIZE = 500       # Max samples for tuning

# Datasets that MUST retain temporal order
TIME_SERIES_DATASETS = ["RCV1", "Electricity", "Covertype", "Bike", "Sensor"]

# FESL Limitation: Cannot run on RCV1 due to O(d^2) memory for Mapping Matrix
# We blacklist it here to prevent the script from crashing.
OOM_RISK_DATASETS = ["RCV1"] 

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
        else:
            print(f"⚠️  Dataset '{d_name}' not found, skipping.")
    return available

def get_stream_length(base_stream, default=10000):
    if hasattr(base_stream, "get_num_instances"):
        return base_stream.get_num_instances()
    if hasattr(base_stream, "n_instances"):
        return base_stream.n_instances
    return default

class AutoTuner:
    """Adaptive Auto-Tuner (Matches FOBOS logic)."""
    def __init__(self, max_burn_in=500, safe_ratio=0.2):
        self.max_burn_in = max_burn_in
        self.safe_ratio = safe_ratio
        # Search space for FESL alpha (Learning Rate)
        self.alphas = [0.01, 0.1, 1.0, 5.0, 10.0]

    def tune(self, stream_builder_func, schema, base_seed, n_total):
        best_acc = -1.0
        best_alpha = 0.1 
        
        # Adaptive Burn-in
        actual_burn_in = min(self.max_burn_in, int(n_total * self.safe_ratio))
        actual_burn_in = max(10, actual_burn_in)

        for alpha in self.alphas:
            stream = stream_builder_func()
            
            # FESL Configuration
            learner = FESLClassifier(
                schema=schema, 
                alpha=alpha, 
                lambda_=0.1,    # Regularization for Mapping
                window_size=100, # Mapping buffer
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
                best_alpha = alpha
                
        return best_alpha

def run_single_seed_trace(dataset_cls, wrapper_mode, seed, d_name):
    
    # 1. Base Stream Setup & Smart Shuffling
    def get_base_stream():
        b = dataset_cls()
        if d_name not in TIME_SERIES_DATASETS:
            return ShuffledStream(b, random_seed=seed)
        return b

    temp_stream = get_base_stream()
    n_total = get_stream_length(temp_stream)
    
    # 2. Wrapper Factory
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
    
    # 3. Auto-Tuning (Adaptive)
    schema = temp_stream.get_schema()
    tuner = AutoTuner(max_burn_in=BURN_IN_SIZE, safe_ratio=0.2)
    best_alpha = tuner.tune(create_wrapped_stream, schema, seed, n_total)
    
    # 4. Final Execution
    stream = create_wrapped_stream()
    learner = FESLClassifier(
        schema=schema,
        alpha=best_alpha,  # Tuned
        lambda_=0.1,       # Fixed regularization
        window_size=100,   # Fixed window
        random_seed=seed
    )
    
    # Adaptive Logging Parameters
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
                "BestAlpha": best_alpha, 
                "WindowSize": window_size
            })
            
    return pd.DataFrame(trace_data), best_alpha

def plot_and_save_trace(agg_df, mode, d_name, output_dir):
    plt.figure(figsize=(10, 6))
    
    w_size = int(agg_df['WindowSize_mean'].iloc[0])
    avg_alpha = agg_df['BestAlpha_mean'].iloc[0]
    
    plt.plot(agg_df['Step'], agg_df['PreqAcc_mean'], label=f'Prequential Acc (w={w_size})', color='tab:blue', linewidth=2)
    plt.plot(agg_df['Step'], agg_df['CumAcc_mean'], label='Cumulative Acc', color='tab:orange', linewidth=2, linestyle='--')
    
    # Standard Deviation Shading
    plt.fill_between(agg_df['Step'], 
                     np.maximum(0, agg_df['PreqAcc_mean'] - agg_df['PreqAcc_std']), 
                     np.minimum(1, agg_df['PreqAcc_mean'] + agg_df['PreqAcc_std']), 
                     color='tab:blue', alpha=0.15)

    plt.title(f"FESL: {d_name} - {mode} (Avg $\\alpha$={avg_alpha:.2f})", fontsize=14)
    plt.xlabel("Instances Processed", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_filename = os.path.join(output_dir, f"plot_{mode}_{d_name}.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()

def main():
    datasets = get_datasets()
    if not datasets:
        print("❌ No datasets found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    modes = ["TDS", "CDS", "EDS"]
    
    total_combinations = len(modes) * len(datasets)
    curr_comb = 0

    print(f"🚀 Starting FESL Benchmark (Strict Mode)")
    print(f"   Config: {N_REPEATS} Repeats | Auto-Tuning | Shuffling | OOM Safety")
    print(f"   Output: {OUTPUT_DIR}")
    print("=" * 90)

    for mode in modes:
        print(f"\n🔹 === Mode: {mode} ===")
        
        for d_name, d_cls in datasets:
            curr_comb += 1
            
            # [SAFETY CHECK] Skip RCV1 for FESL to prevent crash
            if d_name in OOM_RISK_DATASETS:
                print(f"[{curr_comb}/{total_combinations}] {d_name:<15} | SKIPPED (OOM Risk for FESL)")
                continue
                
            print(f"[{curr_comb}/{total_combinations}] {d_name:<15} | Seeds: ", end="", flush=True)
            
            all_runs_data = []
            alphas_selected = []
            
            start_time = time.time()
            
            for seed in range(1, N_REPEATS + 1):
                try:
                    run_df, chosen_alpha = run_single_seed_trace(d_cls, mode, seed, d_name)
                    all_runs_data.append(run_df)
                    alphas_selected.append(chosen_alpha)
                    print(f"{seed}", end="", flush=True)
                except Exception as e:
                    print(f"![Err: {e}]", end="")
            
            if not all_runs_data:
                print(" Failed.")
                continue

            combined_df = pd.concat(all_runs_data, ignore_index=True)
            agg_df = combined_df.groupby('Step').agg({
                'PreqAcc': ['mean', 'std'],
                'CumAcc': ['mean', 'std'],
                'BestAlpha': ['mean'],
                'WindowSize': ['mean']
            }).reset_index()
            agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
            
            elapsed = time.time() - start_time
            
            if alphas_selected:
                mode_alpha = max(set(alphas_selected), key=alphas_selected.count) 
            else:
                mode_alpha = 0.0
            
            print(f" Done ({elapsed:.1f}s) -> Best α: {mode_alpha}")

            csv_filename = os.path.join(OUTPUT_DIR, f"trace_{mode}_{d_name}.csv")
            agg_df.to_csv(csv_filename, index=False)
            plot_and_save_trace(agg_df, mode, d_name, OUTPUT_DIR)

    print("\n✅ FESL Benchmark Completed.")

if __name__ == "__main__":
    main()