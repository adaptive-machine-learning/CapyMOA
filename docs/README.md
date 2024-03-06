# Contributing to Docs
There are three source of documentation in this repository:

 1. Auto-generated documentation from the source code using Autodoc. See this 
    [link](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) 
    for more information.

 2. Juptyer notebooks in the `/notebooks` directory are converted to markdown 
    files and included in the documentation with `nbsphinx`. See this 
    [link](https://nbsphinx.readthedocs.io) for more information.

    To add a notebook to the documentation, add the notebook to the 
    `/notebooks` directory and add the filename to the `toctree` in
    `index.rst`.

 3. Manually written documentation in the `/docs` directory.

## Prerequisites
Install the documentation dependencies by running the following command in the
root directory of the repository:

```bash
pip install --editable ".[doc,dev]"
```

## Building the Documentation
To build the documentation, run the following command in the project root:
```sh
invoke docs.build
```
To build the documentation, serve it locally, and watch for changes, run the
following command in the project root:
```sh
invoke docs.dev
```
