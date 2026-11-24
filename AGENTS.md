# AGENTS.md

* Setup `uv -q sync --extra dev --extra torch-cpu --extra doc` to install development dependencies.
    - If you have a GPU, you can use `--extra torch` instead of `--extra torch-cpu`.
* Run in virtual environment with `uv run ...`.
* Run doctests with `uv run pytest -q --doctest-modules src`.
* Run unit tests with `uv run pytest -q tests`.
    * `pytest.mark.torch` - requires PyTorch to be installed.
* Check notebooks execute with `uv run invoke docs.nb`.
* Build documentation with `uv run invoke docs`.
* Clean documentation with `uv run invoke docs.clean`.
* MUST run formatter `uv run invoke fmt` before committing code.
* MUST ask the user before pushing to GitHub.

## Tests

* Before adding a test, name the behavior, a credible mistake it catches, and an expected result that does not depend on the code under test.
* Prefer small, fast tests whose failure points at the broken behavior. Use integration tests only for interactions that small tests cannot show.
* Do not copy implementation logic into assertions or mocks.
* Test contracts, not internal structure.
* Do not keep development scaffolding as tests.
* Never delete or weaken a failing test just to make the suite pass.
* CI runs in about 10 minutes. Do not add slow tests or parametrize over redundant cases (seeds, learner variants).

## Commit messages

* SHOULD use conventional commit messages. You MAY introduce breaking changes, but MUST
  NOT include `!` or `BREAKING CHANGE` in the commit message since we are pre v1.0.0.
* MUST NOT add `Co-Authored-By: Claude <noreply@anthropic.com>` (or similar) to commit messages.
* MUST add `Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL1] [TOOL2] ...` to the end of
  commit messages and PR descriptions.
    * `AGENT_NAME` is the name of the harness (/tool/framework)
    * `MODEL_VERSION` is the specific model version. SHOULD be as specific as possible
       including model family, version, and size.
    * `[TOOL1] [TOOL2]` are optional agent developer tools or MCP. SHOULD NOT include
       basic developer tools.
    * Example: `Assisted-by: claude-code:claude-sonnet-4.6 github-mcp-server`
    * Example: `Assisted-by: copilot:gemini-3.7-flash`

## Documentation

* SHOULD use simplified technical english in documentation and comments.
* SHOULD NOT use long words where a short one will do.
* SHOULD use everyday words except where technical terms are required.
* If it is possible to cut a word out, you SHOULD cut it out.
* SHOULD avoid complex sentence structures. 
* Consider using commas or parenthesis over em-dash or removing the clause entirely.
