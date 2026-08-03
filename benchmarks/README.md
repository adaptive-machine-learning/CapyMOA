# Benchmarking

This directory documents how to reproduce the CapyMOA versus River versus MOA benchmark on supported CapyMOA datasets.

The benchmark code for CapyMOA and River lives in [`benchmarking.py`](benchmarking.py), while the MOA CLI benchmark lives in [`benchmarking_moa.py`](benchmarking_moa.py). Shared plotting helpers live in [`plotting.py`](plotting.py).

## What This Benchmark Measures

The script allows a clear comparison of CapyMOA and others on the same streaming classification dataset using test-then-train evaluation.

The reported outputs include:

- accuracy
- wall-clock time
- CPU time

The script benchmarks these learners:

- `NaiveBayes`
- `HT`
- `EFDT`
- `KNN`
- `ARF5`
- `ARF10`
- `ARF30`
- `ARF100`
- `ARF100j4` for CapyMOA only

## Setup

Use a dedicated CapyMOA environment first. A conda environment is a reasonable option:

```bash
conda create -n capymoa python=3.11
conda activate capymoa
```

CapyMOA also requires Java. Check that it is available:

```bash
java -version
```

CapyMOA currently expects PyTorch in the environment as well. For a CPU-only setup:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then install CapyMOA from the repository root:

```bash
pip install -e .
```

Finally, install River separately for this benchmark:

```bash
pip install river
```

## Dataset

The benchmark uses built-in datasets from `capymoa.datasets` for the CapyMOA side and the matching CapyMOA CSV sources for River.

Currently supported dataset keys are:

- `rtg_2abrupt`
- `hyper100k`
- `rbfm_100k`
- `electricity`
- `covtfd`
- `covtype_norm`
- `sensor`

These are intentionally limited to CapyMOA datasets that fit the current classification benchmark and have CSV companions available for River. For completeness, see the [CapyMOA API docs](../docs/api/index.rst), including `capymoa.datasets`.

On the first run, the script downloads any missing dataset files into the repository-local `data/` directory automatically. There is no separate dataset preparation step.

## Run

From the repository root:

```bash
python benchmarks/benchmarking.py
```

By default this runs the benchmark and writes the CSV, raw CSV, machine metadata, and Markdown summaries. Plots are optional and are rendered only when `--render-plots` is provided.

To choose a dataset explicitly:

```bash
python benchmarks/benchmarking.py --dataset electricity
```

To run only one side of the comparison:

```bash
python benchmarks/benchmarking.py --library capymoa
python benchmarks/benchmarking.py --library river
```

To override the default experiment-ID-based filenames:

```bash
python benchmarks/benchmarking.py --output-prefix reference_100k
```

To choose the run scale without editing the script:

```bash
python benchmarks/benchmarking.py --max-instances 100000 --repetitions 5
python benchmarks/benchmarking.py --max-instances 100 --repetitions 1
```

To run only selected learners:

```bash
python benchmarks/benchmarking.py --algorithms HT,EFDT,ARF30
python benchmarks/benchmarking_moa.py --algorithms HT,ARF5
```

To render plots after the benchmark run:

```bash
python benchmarks/benchmarking.py --render-plots
python benchmarks/benchmarking.py --render-plots --dark-theme
```

To regenerate plots from an existing benchmark CSV without rerunning the benchmark:

```bash
python benchmarks/benchmarking.py --plot-only --output-prefix reference_100k
```

This looks for `benchmarks/results/<experiment_id>/reference_100k.csv`. If that file does not exist, the script raises an error instead of silently creating new outputs. `--plot-only` requires an explicit `--output-prefix`.

To render plots with a dark background:

```bash
python benchmarks/benchmarking.py --plot-only --output-prefix reference_100k --dark-theme
```

To provide a custom plot title prefix:

```bash
python benchmarks/benchmarking.py --plot-title "RTG_2abrupt Benchmark"
python benchmarks/benchmarking.py --plot-only --output-prefix reference_100k --plot-title "RTG_2abrupt Benchmark"
```

To disable pulse output entirely:

```bash
python benchmarks/benchmarking.py --no-pulse
```

The MOA script supports the same dataset selection, `--plot-only`, `--render-plots`, `--dark-theme`, `--plot-title`, `--algorithms`, and output-prefix conventions, but it does not currently expose the same pulse output or library-selection options as `benchmarking.py`.

## Outputs

Each run writes its artifacts under [`results/`](results/), inside a subdirectory named after the experiment ID:

- `<dataset_name>_<number_of_instances>/`

Within that experiment directory, the artifacts are:

- `<experiment_id>.csv`: aggregated benchmark results
- `<experiment_id>_raw.csv`: per-repetition results
- `<experiment_id>_machine.json`: lightweight machine details relevant to interpretation of the benchmark
- `<experiment_id>_experiment.md`: human-readable benchmark summary including the CLI, dataset metadata, algorithms, and libraries
- `<experiment_id>_configurations.md`: learner-by-learner benchmark configurations for CapyMOA and River
- `<experiment_id>_performance_plot_*.png`: benchmark plots, written only when `--render-plots` is used or via `--plot-only`
- `pulse_<experiment_id>.csv`: pulse output, written only by `benchmarking.py` unless `--no-pulse` is used
- `pulse/`: pulse summary CSVs and plots, written only when pulse output exists and plots are rendered

When `--output-prefix` is provided, the same artifact set is written using that prefix instead of the default experiment ID.

## Notes

- This benchmark is not currently packaged as an install extra such as `capymoa[benchmark]`.
- The benchmark depends on `river`, but River is intended to be installed separately. 
