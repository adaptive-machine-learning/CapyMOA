# Version Control

This document outlines the version control practices used in the CapyMOA project.

## Linting and Formatting

Linting is the process of automatically checking code for style, syntax, and other issues.
Code formatting ensures code meets a consistent style.
Together they help ensure the codebase is consistent and clean.
Developers don't have to worry about formatting and reviewers can focus on code rather than style preferences.

CapyMOA uses the [ruff](https://astral.sh/ruff) linter to enforce both.
Checks are run automatically using GitHub actions on every pull request.

You will need to run `ruff` locally before committing changes.  
Ruff is installed as part of the [development dependencies](../installation.rst). 

To format files run:
```bash
ruff format # or python -m invoke format
```

To lint files run:
```bash
ruff check # or python -m invoke lint
```

Furthermore, `python -m invoke commit` will run the linter and check that 
formatting is correct before committing.


## Commit Messages

**tldr; Run `python -m invoke commit` (or `invoke commit`, `python -m commitizen commit`) to commit changes.** (Requires that you've [installed the optional development dependencies](../installation.rst).)

CapyMOA uses conventional commit messages to streamline the release process.

> "The Conventional Commits specification is a lightweight convention on top of
> commit messages. It provides an easy set of rules for creating an explicit
> commit history; which makes it easier to write automated tools on top of.
> This convention dovetails with SemVer, by describing the features, fixes,
> and breaking changes made in commit messages." -- [conventionalcommits.org](https://www.conventionalcommits.org/en/v1.0.0/#summary)

Conventional commits are structured as follows:

    <type>[optional scope]: <description>

or

    <type>[optional scope]: <description>

    [optional body]

    [optional footer(s)]

Here are some basic examples:

    docs: correct spelling of CHANGELOG


    feat(lang): add Polish language

Where:

* `<type>` is one of
  * `feat`: New feature. **Will increment the MINOR version number.**
  * `fix`: Bug fix. **Will increment the patch version number.**
  * `build`: Changes that affect the build system or external dependencies
  * `chore`: Repetitive tasks such as updating dependencies
  * `ci`: Changes to continuous integration configuration files and scripts
  * `docs`: Documentation changes
  * `perf`: Performance improvement
  * `refactor`: Code changes that neither fix a bug nor add a feature
  * `revert`: Revert a previous commit
  * `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
  * `test`: Adding missing tests or correcting existing tests
* `[optional scope]` is a module or component affected by the commit. A top level python module is a good example of a scope:
  * `classifier`
  * `datasets`
  * `stream`
  * etc.

  Its okay to leave out the scope if its not obvious or not applicable.

* `<description>` This should be a short, concise lowercase description of the change in the imperative mood (e.g. "add ...", "change ...", "fix", "remove...").

## Breaking Changes

If the API changes in a way that is not backwards-compatible, the commit message
should include a `!` after the type/scope, e.g. `feat(classifiers)!: ...`.

You can and probably should include more information in the body and footer of
the commit message to explain the breaking change. See [conventionalcommits.org](https://www.conventionalcommits.org/en/v1.0.0/) for more information.

    chore!: drop support for python 3.9

    BREAKING CHANGE: use Python features only available in 3.10
