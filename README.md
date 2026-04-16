# [CapyMOA](https://capymoa.org)

![Banner Image](https://github.com/adaptive-machine-learning/CapyMOA/raw/main/docs/images/CapyMOA.jpeg)

[![PyPi Version](https://img.shields.io/pypi/v/capymoa)](https://pypi.org/project/capymoa/)
[![Docker Image Version (tag)](https://img.shields.io/docker/v/tachyonic/jupyter-capymoa/latest?logo=docker&label=Docker&color=blue)](https://hub.docker.com/r/tachyonic/jupyter-capymoa)
[![Join the Discord](https://img.shields.io/discord/1235780483845984367?label=Discord)](https://discord.gg/spd2gQJGAb)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://capymoa.org)
[![GitHub](https://img.shields.io/github/stars/adaptive-machine-learning/CapyMOA?style=social)](https://github.com/adaptive-machine-learning/CapyMOA)
[![Coverage Status](https://coveralls.io/repos/github/adaptive-machine-learning/CapyMOA/badge.svg)](https://coveralls.io/github/adaptive-machine-learning/CapyMOA)


**CapyMOA does efficient machine learning for data streams in Python.**
A data stream is a sequences of items ariving one-by-one that is too
large to efficiently process non-sequentially. CapyMOA is a toolbox of
methods and evaluators for: classification, regression, clustering,
anomaly detection, semi-supervised learning, online continual learning,
and drift detection for data streams.

For the default PyTorch CUDA GPU installation, run:

```
pip install capymoa
```

Refer to the [Setup](https://capymoa.org/setup) guide for other options,
including CPU-only and dev dependencies.

```python
from capymoa.datasets import Electricity
from capymoa.classifier import HoeffdingTree
from capymoa.evaluation import prequential_evaluation

# 1. Load a streaming dataset
stream = Electricity()

# 2. Create a machine learning model
model = HoeffdingTree(stream.get_schema())

# 3. Run with test-then-train evaluation
results = prequential_evaluation(stream, model)

# 3. Success!
print(f"Accuracy: {results.accuracy():.2f}%")
```

Next, we recomend the [Tutorials](https://capymoa.org/tutorials).

> **⚠️ WARNING**
>
> CapyMOA is still in the early stages of development. The API is subject to
> change until version 1.0.0. If you encounter any issues, please report
> them in [GitHub Issues](https://github.com/adaptive-machine-learning/CapyMOA/issues)
> or talk to us on [Discord](https://discord.gg/spd2gQJGAb).

---

![Benchmark Image](https://github.com/adaptive-machine-learning/CapyMOA/raw/main/docs/images/arf100_cpu_time.png)
Benchmark comparing CapyMOA against other data stream libraries. The benchmark
was performed using an ensemble of 100 ARF learners trained on
`capymoa.datasets.RTG_2abrupt` dataset containing 100,000 samples and 30
features.  You can find the code to reproduce this benchmark in
[`notebooks/benchmarking.py`](https://github.com/adaptive-machine-learning/CapyMOA/blob/main/notebooks/benchmarking.py).
*CapyMOA has the speed of MOA with the flexibility of Python and the richness of
Python's data science ecosystem.*

## Cite Us 

If you use CapyMOA in your research, please cite us using the following BibTeX item.
```
@misc{
    gomes2025capymoaefficientmachinelearning,
    title={{CapyMOA}: Efficient Machine Learning for Data Streams in Python},
    author={Heitor Murilo Gomes and Anton Lee and Nuwan Gunasekara and Yibin Sun and Guilherme Weigert Cassales and Justin Jia Liu and Marco Heyden and Vitor Cerqueira and Maroua Bahri and Yun Sing Koh and Bernhard Pfahringer and Albert Bifet},
    year={2025},
    eprint={2502.07432},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2502.07432},
}
```
