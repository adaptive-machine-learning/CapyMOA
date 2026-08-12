# AGENTS.md

* Setup `uv -q sync --extra dev --extra torch-cpu --extra doc` to install development dependencies.
    - If you have a GPU, you can use `--extra torch` instead of `--extra torch-cpu`.
* Run in virtual environment with `uv run ...`.
* Run doctests with `uv run pytest -q --doctest-modules src`.
* Run unit tests with `uv run pytest -q tests`.
    * `pytest.mark.torch` - requires PyTorch to be installed.
* Run notebook tests with `uv run pytest -q --nbmake notebooks`.
* Build documentation with `uv run invoke docs`.
* Clean documentation with `uv run invoke docs.clean`.
* Formatter `uv run invoke fmt`.
