![A cute capybara animal riding a moa like a horse](/docs/images/CapyMOA.jpeg)

# [CapyMOA](https://capymoa.org)
<img src="https://img.shields.io/pypi/v/capymoa" href="https://pypi.org/project/capymoa/" alt="PyPi Version"/>
<img src="https://img.shields.io/discord/1235780483845984367?label=Discord" href="https://discord.gg/spd2gQJGAb" alt="Join the discord"/>

Machine learning library tailored for data streams. Featuring a Python API
tightly integrated with MOA (**Stream Learners**), PyTorch (**Neural
Networks**), and scikit-learn (**Machine Learning**). CapyMOA provides a
**fast** python interface to leverage the state-of-the-art algorithms in the
field of data streams.

To setup CapyMOA, simply install it via pip. If you have any issues with the 
installation (like not having Java installed) or if you want GPU support, please
refer to the [installation guide](docs/installation.md). Once installed take a
look at the [tutorials](capymoa.org/notebooks/index.html) to get started.

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



> [!WARNING]  
> CapyMOA is still in the early stages of development. The API is subject to 
> change until version 1.0.0. If you encounter any issues, please report 
> them in [GitHub Issues](https://github.com/adaptive-machine-learning/CapyMOA/issues)
> or talk to us on [Discord](https://discord.gg/spd2gQJGAb).


![Benchmark of capymoa being faster than river.](/docs/images/benchmark_20240422_221824_performance_plot_wallclock.png)

## 🏗️ Contributing 

* **[How to install CapyMOA.](docs/installation.md)**
* **[How to add documentation.](docs/contributing/docs.md)**
* **[How to add tests.](docs/contributing/tests.md)**
* **[How to add new algorithms or methods.](docs/contributing/learners.md)**
* **[How to format a commit message for CapyMOA.](docs/contributing/vcs.md)**
