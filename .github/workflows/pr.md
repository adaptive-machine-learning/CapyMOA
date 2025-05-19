# Pull Request Automation Workflow
Whenever a pull request is opened a number of automated checks are run to ensure breaking changes are not introduced and to ensure consistency of style and formatting.

These checks are defined by the `.github/workflows/pr.yml` file and this documentation is defined by `.github/workflows/pr.md`.

If you encounter any issues, reading the error messages and trying to reproduce them locally is a good first step.
Don't hesitate to ask for assistance in the pull request or join the Discord server.

## Tests

This job runs formatting, linting, tests, doctests, and checks notebooks.

- If "**Tests and Doctests**" step fails the [Adding Tests guide](https://capymoa.org/contributing/tests.html) may help.
- If "**Check Notebooks**" step fail the [Notebooks guide](https://capymoa.org/contributing/docs.html#notebooks) may help.

## Code Style

This job uses `ruff` to check the code style. If this job fail the [Linting and Formatting guide](https://capymoa.org/contributing/vcs.html#linting-and-formatting) may help.

## Commit Style

This job checks if commit messages are conventional commit compliant. If these checks fail, the [Commit Messages guide](https://capymoa.org/contributing/vcs.html#commit-messages) may help. **Don't worry too much about this check, as the reviewer can assist by squashing and merging commits with a compliant message.**

## Check Documentation

This job ensures that the documentation can be built successfully. If this check fails, the [Documentation guide](https://capymoa.org/contributing/docs.html) may help.
