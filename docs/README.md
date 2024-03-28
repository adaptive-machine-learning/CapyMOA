# Adding Documentation

## Prerequisites
Install the documentation dependencies by running the following command in the
root directory of the repository:

```bash
pip install --editable ".[doc,dev]"
```

## Building Documentation
To build the documentation, run the following command in the project root:
```sh
invoke docs.build
```
To build the documentation, serve it locally, and watch for changes, run the
following command in the project root:
```sh
invoke docs.dev
```
To clean the documentation, run the following command in the project root:
```sh
invoke docs.clean
```

## Types of Documentation

### Auto-Generated Documentation
Auto-generated documentation from the source code using Autodoc. See this 
[link](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) 
for more information.

### Juptyer Notebooks

Juptyer notebooks in the `/notebooks` directory are converted to markdown 
files and included in the documentation with `nbsphinx`. See this [link](https://nbsphinx.readthedocs.io) for more information.

To add a notebook to the documentation, add the notebook to the `/notebooks` directory and add the filename to the `toctree` in `notebooks/index.rst`.

### Manual Documentation
Manually written documentation in the `/docs` directory. These can be written in
reStructuredText or Markdown. To add a new page to the documentation, add a new
file to the `/docs` directory and add the filename to the `toctree` in `index.rst`
or the appropriate location in the documentation.
