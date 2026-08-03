# Benchmarking

This directory documents how to reproduce CapyMOA benchmarks and comparisons against other libraries, such as MOA, on supported CapyMOA datasets.

The benchmark code for CapyMOA lives in [`benchmarking.py`](benchmarking.py), while the MOA CLI benchmark lives in [`benchmarking_moa.py`](benchmarking_moa.py). Shared plotting helpers live in [`plotting.py`](plotting.py).

## What This Benchmark Measures

The scripts allow a clear comparison of CapyMOA and other libraries on the same streaming classification dataset using test-then-train evaluation.

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

Install CapyMOA first by following the main project installation guide:

- [Installation Guide](https://capymoa.org/setup)
- [Developer Setup](https://capymoa.org/setup/developer.html)

## Dataset

The benchmark uses built-in datasets from `capymoa.datasets`. When the optional River comparison is used, the script also relies on the matching CapyMOA CSV sources.

Currently supported dataset keys are:

- `rtg_2abrupt`
- `hyper100k`
- `rbfm_100k`
- `electricity`
- `covtfd`
- `covtype_norm`
- `sensor`

These are intentionally limited to CapyMOA datasets that fit the current classification benchmark. For completeness, see the [CapyMOA API docs](../docs/api/index.rst), including `capymoa.datasets`.

On the first run, the script downloads any missing dataset files into the repository-local `data/` directory automatically. There is no separate dataset preparation step.

## Run

From the repository root:

```bash
python benchmarks/benchmarking.py
```

By default this runs the benchmark and writes the CSV, raw CSV, machine metadata, and Markdown summaries. Plots are optional and are rendered only when `--render-plots` is provided.

For a straightforward CapyMOA-only run:

```bash
python benchmarks/benchmarking.py --dataset rtg_2abrupt --library capymoa
```

For a quicker CapyMOA-only smoke run with just a few learners:

```bash
python benchmarks/benchmarking.py --dataset rtg_2abrupt --library capymoa --algorithms HT,EFDT,ARF30
```

To choose a dataset explicitly:

```bash
python benchmarks/benchmarking.py --dataset electricity
```

To run only one side of the comparison:

```bash
python benchmarks/benchmarking.py --library capymoa
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

The MOA script supports the same dataset selection, `--plot-only`, `--render-plots`, `--dark-theme`, `--plot-title`, `--algorithms`, and output-prefix conventions, but it does not currently expose the same pulse output options as `benchmarking.py`.

## Outputs

Each run writes its artifacts under [`results/`](results/), inside a subdirectory named after the experiment ID:

- `<dataset_name>_<number_of_instances>/`

Within that experiment directory, the artifacts are:

- `<experiment_id>.csv`: aggregated benchmark results
- `<experiment_id>_raw.csv`: per-repetition results
- `<experiment_id>_machine.json`: lightweight machine details relevant to interpretation of the benchmark
- `<experiment_id>_experiment.md`: human-readable benchmark summary including the CLI, dataset metadata, algorithms, and libraries
- `<experiment_id>_configurations.md`: learner-by-learner benchmark configurations
- `<experiment_id>_performance_plot_*.png`: benchmark plots, written only when `--render-plots` is used or via `--plot-only`
- `pulse_<experiment_id>.csv`: pulse output, written only by `benchmarking.py` unless `--no-pulse` is used
- `pulse/`: pulse summary CSVs and plots, written only when pulse output exists and plots are rendered

When `--output-prefix` is provided, the same artifact set is written using that prefix instead of the default experiment ID.

## Notes

- This benchmark is not currently packaged as an install extra such as `capymoa[benchmark]`.
- River is optional for this benchmark and is intended to be installed separately if you want to compare against it.
