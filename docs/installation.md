# Installation
This document describes how to install CapyMOA and its dependencies. CapyMOA
is tested against Python 3.9, 3.10, and 3.11. Newer versions of Python will
likely work but have yet to be tested.

- [Installation](#installation)
  - [Environment](#environment)
  - [Dependencies](#dependencies)
    - [Java (Required)](#java-required)
    - [PyTorch (Required)](#pytorch-required)
  - [Install CapyMOA](#install-capymoa)
    - [Install CapyMOA for Development](#install-capymoa-for-development)

## Environment
We recommend using a virtual environment to manage your Python environment. Miniconda
is a good choice for managing Python environments. You can install Miniconda from
[here](https://docs.conda.io/en/latest/miniconda.html). Once you have Miniconda
installed, you can create a new environment with:

```bash
conda create -n capymoa python=3.9
conda activate capymoa
```

When your environment is activated, you can install CapyMOA by following the
instructions below.

## Dependencies
Most of the dependencies for CapyMOA are installed automatically with pip. 
However, there are a few dependencies that need to be installed manually.

### Java (Required)
CapyMOA requires Java to be installed and accessible in your environment. You
can check if Java is installed by running the following command in your terminal:

```bash
java -version
```

If Java is not installed, you can download OpenJDK (Open Java Development Kit) 
from [this link](https://openjdk.org/install/), or alternatively the Oracle 
JDK from [this link](https://www.oracle.com/java). **Linux users can also
install OpenJDK using their distribution's package manager.**

Now that Java is installed, you should see an output similar to the following
when you run the command ``java -version``:
```
openjdk version "17.0.9" 2023-10-17
OpenJDK Runtime Environment (build 17.0.9+8)
OpenJDK 64-Bit Server VM (build 17.0.9+8, mixed mode)
```

### PyTorch (Required)
The CapyMOA algorithms using deep learning require PyTorch. It
is not installed by default because different versions are required
for different hardware. If you want to use these algorithms, follow the
instructions [here](https://pytorch.org/get-started/locally/) to
get the correct version for your hardware.

For CPU only, you can install PyTorch with:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Install CapyMOA

```bash
pip install capymoa
```

### Install CapyMOA for Development
If you want to contribute to CapyMOA, you should clone the repository, install
development dependencies, and install CapyMOA in editable mode:

```bash
git clone git@github.com:adaptive-machine-learning/CapyMOA.git # or your fork
cd CapyMOA
pip install --editable ".[dev]"
```
