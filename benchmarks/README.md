# Benchmarking

This directory documents how to reproduce the CapyMOA versus River benchmark on supported CapyMOA datasets.

The benchmark code lives in [`benchmarking.py`](benchmarking.py).

## What This Benchmark Measures

The script compares CapyMOA and River on the same streaming classification dataset using test-then-train evaluation.

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
- `ARF100j4` for CapyMOA only, and it is included in the plots even though River has no matching bar

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

To regenerate plots from an existing benchmark CSV without rerunning the benchmark:

```bash
python benchmarks/benchmarking.py --plot-only --output-prefix reference_100k
```

This looks for `benchmarks/results/<experiment_id>/reference_100k.csv`. If that file does not exist, the script raises an error instead of silently creating new outputs. `--plot-only` requires an explicit `--output-prefix`.

To render plots with a dark background:

```bash
python benchmarks/benchmarking.py --dark-theme
python benchmarks/benchmarking.py --plot-only --output-prefix reference_100k --dark-theme
```

To provide a custom plot title prefix:

```bash
python benchmarks/benchmarking.py --plot-title "RTG_2abrupt Benchmark"
python benchmarks/benchmarking.py --plot-only --output-prefix reference_100k --plot-title "RTG_2abrupt Benchmark"
```

## Outputs

Each run writes its artifacts under [`results/`](results/), inside a subdirectory named after the experiment ID:

- `<dataset_name>_<number_of_instances>/`

Within that experiment directory, the artifacts are:

- `<experiment_id>.csv`: aggregated benchmark results
- `<experiment_id>_raw.csv`: per-repetition results
- `<experiment_id>_performance_plot_*.png`: benchmark plots
- `<experiment_id>_machine.json`: lightweight machine details relevant to interpretation of the benchmark
- `<experiment_id>_experiment.md`: human-readable benchmark summary including the CLI, dataset metadata, algorithms, and libraries
- `<experiment_id>_configurations.md`: learner-by-learner benchmark configurations for CapyMOA and River

When `--output-prefix` is provided, the same artifact set is written using that prefix instead of the default experiment ID.

## Notes

- This benchmark is not currently packaged as an install extra such as `capymoa[benchmark]`.
- The benchmark depends on `river`, but River is intended to be installed separately rather than as a core CapyMOA dependency.
- If you have `PYTHONPATH` exported to another CapyMOA checkout, unset it before running the benchmark so Python imports this repository's editable install.
