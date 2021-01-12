<p align="center">
<img src="https://github.com/Sbozzolo/kuibit/raw/master/logo.png" height="120">
</p>

[![codecov](https://codecov.io/gh/Sbozzolo/kuibit/branch/master/graph/badge.svg)](https://codecov.io/gh/Sbozzolo/kuibit)
![Tests and documentation](https://github.com/Sbozzolo/kuibit/workflows/Tests/badge.svg)
[![GPLv3
license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![Get help on Telegram](https://img.shields.io/badge/Get%20help%20on-Telegram-blue.svg)](https://t.me/kuibit)
[![PyPI version](https://badge.fury.io/py/kuibit.svg)](https://badge.fury.io/py/kuibit)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/Sbozzolo/kuibit/?ref=repository-badge)

# kuibit

`kuibit` is a Python library to analyze simulations performed with the Einstein
Toolkit largely inspired by
[PostCactus](https://github.com/wokast/PyCactus/tree/master/PostCactus).
`kuibit` can read simulation data and represent it with high-level classes. For
a list of features available, look at the [official
documentation](https://sbozzolo.github.io/kuibit).

## Installation

``kuibit`` is available in PyPI. To install it with `pip`
``` bash
pip3 install kuibit
```
If they are not already available, `pip` will install all the necessary dependencies.

The minimum version of Python required is 3.6.

If you intend to develop ``kuibit``, follow the instruction below.

### Development

For development, we use [poetry](https://python-poetry.org/). Poetry simplifies
dependency management, building, and publishing the package.

To install `kuibit` with poetry, clone this repo, move into the folder, and run:
``` sh
poetry install -E full
```
This will download all the needed dependencies in a sandboxed environment (the
`-E full` flag is for the optional dependencies). When you want to use
``kuibit``, just run ``poetry shell``. This will drop you in a shell in
which you have full access to ``kuibit`` in "development" version, and its
dependencies (also the one needed only for development).

## Documentation

`kuibit` uses Sphinx to generate the documentation. To produce the documentation
```sh
cd docs && make html
```
Documentation is automatically generated after each commit by GitHub Actions.

We use [nbsphinx](https://nbsphinx.readthedocs.io/) to translate Jupyter
notebooks to the examples. The extension is required. Note: Jupyter notebooks
have to be un-evaluated. `nbsphinx` requires [pandoc](https://pandoc.org/). If
don't have `pandoc`, you should comment out `nbsphinx` in `docs/conf.py`, or
compiling the documentation will fail.

## Tests

`kuibit` comes with a suite of unit tests. To run the tests, (in a poetry shell),
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

## Experimental branch

The git repo of `kuibit` has an `experimental` branch, which contains
modules for visualization and several general-purpose scripts (e.g., to plot a
given grid variable via command-line). It is worth to have a look at that branch
too.

## What is a _kuibit_?

A kuibit (harvest pole) is the tool traditionally used by the Tohono O'odham
people to reach the fruit of the Saguaro cacti during the harvesting season. In
the same way, this package is a tool that you can use to collect the fruit of
your `Cactus` simulations.

## Credits

`kuibit` follows the same designed as `PostCactus`, code developed by Wolfgang
Kastaun. This fork completely rewrites the original code, adding emphasis on
documentation, testing, and extensibility. The logo contains elements designed
by [freepik.com](freepik.com).

