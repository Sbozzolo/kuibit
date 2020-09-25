![PostCactus-logo](logo.png)

[![codecov](https://codecov.io/gh/Sbozzolo/PostCactus/branch/master/graph/badge.svg)](https://codecov.io/gh/Sbozzolo/PostCactus)
![Tests](https://github.com/Sbozzolo/PostCactus/workflows/Tests/badge.svg)
![Docs](https://github.com/Sbozzolo/PostCactus/workflows/Docs/badge.svg)
[![GPLv3
license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/Sbozzolo/PostCactus/?ref=repository-badge)

# PostCactus

[PostCactus](https://github.com/wokast/PyCactus/tree/master/PostCactus) is a set
of tools to post-process simulations performed with the Einstein Toolkit
originally developed by Wolfgang Kastaun. This repository contains a fork of
PostCactus with the following differences:
1. We use Python 3 instead of Python 2
2. We drop support to some features, but add others (see documentation for a
   list of the available features)
3. We test and document all the code

The goal of this fork is to make PostCactus more robust and developer-friendly,
hence the emphasis on documentation and testing. We try to make the code easy to
understand and extend.

Documentation is found at the following link:
[sbozzolo.github.io/PostCactus](https://sbozzolo.github.io/PostCactus).

## Installation

``PostCactus`` is available in TestPyPI. To install it with `pip`
``` bash
   pip3 install --index-url https://test.pypi.org/simple/ postcactus
```
If they are not already available, `pip` will install the following packages:
- `numpy`
- `numba`
- `h5py`
- `scipy`.

The minimum version of Python required is 3.5.

If you intend to develop ``PostCactus``, follow the instruction below.

### Development

For development, we use [poetry](https://python-poetry.org/). Poetry simplifies
dependency management, building, and publishing the package.

To install `PostCactus` with poetry, clone this repo, move into the folder, and run:
``` sh
   poetry install
```
This will download all the needed dependencies in a sandboxed environment. When
you want to use ``PostCactus``, just run ``poetry shell``. This will drop you in
a shell in which you have full access to ``PostCactus`` in "development" version,
and its dependencies (also the one needed only for development).

## Documentation

`PostCactus` uses Sphinx to generate the documentation. To produce the documentation
```sh
cd docs && make html
```
Documentation is automatically generated after each commit by GitHub Actions.

We use [nbsphinx](https://nbsphinx.readthedocs.io/) to translate Jupyter
notebooks to the examples. The extension is required. Note: Jupyter notebooks
have to be un-evaluated.

## Tests

`PostCactus` comes with a suite of unit tests. To run the tests, (in a poetry shell),
```sh
poetry run python -m unittest
```
Tests are automatically run after each commit by GitHub Actions.

If you want to look at the coverage of your tests, run (in a poetry shell)
```sh
coverage run -m unittest
coverage html
```
This will produce a directory with the html files containing the analysis of
the coverage of the tests.
