# [CapyMOA](https://capymoa.org)

![Banner Image](https://github.com/adaptive-machine-learning/CapyMOA/raw/main/docs/images/CapyMOA.jpeg)

[![PyPi Version](https://img.shields.io/pypi/v/capymoa)](https://pypi.org/project/capymoa/)
[![Join the Discord](https://img.shields.io/discord/1235780483845984367?label=Discord)](https://discord.gg/spd2gQJGAb)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://capymoa.org)
[![GitHub](https://img.shields.io/github/stars/adaptive-machine-learning/CapyMOA?style=social)](https://github.com/adaptive-machine-learning/CapyMOA)


Machine learning library tailored for data streams. Featuring a Python API
tightly integrated with MOA (**Stream Learners**), PyTorch (**Neural
Networks**), and scikit-learn (**Machine Learning**). CapyMOA provides a
**fast** python interface to leverage the state-of-the-art algorithms in the
field of data streams.

To setup CapyMOA, simply install it via pip. If you have any issues with the
installation (like not having Java installed) or if you want GPU support, please
refer to the [installation guide](https://capymoa.org/installation). Once installed take a
look at the [tutorials](https://capymoa.org/tutorials.html) to get started.

```bash
# CapyMOA requires Java. This checks if you have it installed
java -version

# CapyMOA requires PyTorch. This installs the CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install CapyMOA and its dependencies
pip install capymoa

# Check that the install worked
python -c "import capymoa; print(capymoa.__version__)"
```

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
