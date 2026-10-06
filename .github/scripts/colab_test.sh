#!/usr/bin/env bash
# Installs and tests CapyMOA the way a Google Colab user does: system Python,
# plain pip, no virtualenv. Run from the repository root inside the Colab image:
#
#   docker run --rm --user root -v "$PWD:/src" -w /src \
#     us-docker.pkg.dev/colab-images/public/cpu-runtime \
#     bash .github/scripts/colab_test.sh
set -euxo pipefail

# The build hook downloads MOA and generates stubs, so Java must exist before
# `pip install`.
if ! java -version; then
    apt-get update
    apt-get install -y --no-install-recommends default-jre-headless
fi

# Leaves Colab's preinstalled torch untouched.
pip install .

# Test dependencies are installed separately so `pip install .` stays what a
# user runs.
pip install pytest pytest-timeout pytest-subtests pytest-xdist

# Importing from the checkout would test the source tree, not the install.
(cd /tmp && python -c "import capymoa; print(capymoa.__file__); assert 'site-packages' in capymoa.__file__; capymoa.about()")

# `-o pythonpath=` overrides pyproject.toml's `pythonpath = ["src"]` so the
# tests import the pip-installed package.
python -m pytest tests -o pythonpath= --timeout=180 --exitfirst
