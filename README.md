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

The goal of this fork is to make PostCactus more robust and user-friendly, hence
the emphasis on documentation and testing.

Documentation is found [here](https://sbozzolo.github.io/PostCactus).

## Installation

Clone this repo:
``` bash
   git clone https://github.com/Sbozzolo/PostCactus.git
```
Move into the folder and install with `pip`:
``` bash
   cd PostCactus && pip3 install --user .
```
For development, it is convenient to use `pip3 install -e . --user`, so that
modifying the files will have a direct effect on the library (otherwise it has
to be installed after every edit).

If they are not already available, `pip` will install the following packages:
- `numpy`
- `scipy`.

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

`PostCactus` comes with a suite of unit tests. To run the tests,
```sh
python3 -m unittest
```
Tests are automatically run after each commit by GitHub Actions.

