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
* SHOULD use conventional commit messages. You MAY introduce breaking changes, but MUST
  NOT include `!` or `BREAKING CHANGE` in the commit message since we are pre v1.0.0.
* MUST run formatter `uv run invoke fmt` before committing code.
* MUST NOT add `Co-Authored-By: Claude <noreply@anthropic.com>` (or similar) to commit messages.
* MUST ask the user before pushing to GitHub.

## Documentation

* SHOULD use simplified technical english in documentation and comments.
* SHOULD NOT use long words where a short one will do.
* SHOULD use everyday words except where technical terms are required.
* If it is possible to cut a word out, you SHOULD cut it out.
* SHOULD avoid complex sentence structures. 
* Consider using commas or parenthesis over em-dash or removing the clause entirely.
