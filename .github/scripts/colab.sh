#!/usr/bin/env bash
# Install and test CapyMOA in a Google Colab like environment using system Python, pip,
# and no venv.
#
# Reproduce with docker:
#
# $ docker run --rm -v "$PWD:/src" -w /src \
#     --entrypoint bash \
#     us-docker.pkg.dev/colab-images/public/cpu-runtime \
#     .github/scripts/colab.sh
set -euxo pipefail

pip install . pytest-subtests

# Run from /tmp, not /src, so this imports the installed package rather than
# the checkout.
(cd /tmp && python -c "import capymoa; print(capymoa.__file__); capymoa.about()")

# `-o pythonpath=` overrides pyproject.toml's `pythonpath = ["src"]` so the
# tests import the pip-installed package.
python -m pytest tests -o pythonpath= --exitfirst
