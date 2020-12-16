<p align="center">
<img src="https://github.com/Sbozzolo/PostCactus/raw/master/logo.png" width="435" height="200">
</p>

[![codecov](https://codecov.io/gh/Sbozzolo/PostCactus/branch/master/graph/badge.svg)](https://codecov.io/gh/Sbozzolo/PostCactus)
![Tests and documentation](https://github.com/Sbozzolo/PostCactus/workflows/Tests/badge.svg)
[![GPLv3
license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/Sbozzolo/PostCactus/?ref=repository-badge)

# PostCactus-IL Compatible Fork!
This is a fork of the main PostCactus 3+ repository, with additions to support use with the
Illinois GRMHD code, which formats some of its output ever so slightly differently than the Einstein Toolkit,
and as a result evades PostCactus's attempts to read in all scalars, as well as Psi4 data.
Hopefully, I will keep this up-to-date with the main PostCactus branch, but it is worth remembering that much
of what I am supporting here is not publicly available, nor under active development, so there is a limit to how
much development effort should go into this fork.
Also, since the Illinois code is an example of a Cactus code that PostCactus doesn't natively support, should't
it really be called "PostEinsteinToolkit"? Discuss!

# PostCactus

PostCactus is a Python library to analyze simulations performed with the
Einstein Toolkit [originally developed by Wolfgang
Kastaun](https://github.com/wokast/PyCactus/tree/master/PostCactus). PostCactus
can read simulation data and represent it with high-level classes. For a list of
features available, look at the [official
documentation](https://sbozzolo.github.io/PostCactus).

## Installation

``PostCactus`` is available in TestPyPI. To install it with `pip`
``` bash
pip3 install --index-url https://test.pypi.org/simple/ postcactus
```
If they are not already available, `pip` will install all the necessary dependencies.

The minimum version of Python required is 3.6.

If you intend to develop ``PostCactus``, follow the instruction below.

### Development

For development, we use [poetry](https://python-poetry.org/). Poetry simplifies
dependency management, building, and publishing the package.

To install `PostCactus` with poetry, clone this repo, move into the folder, and run:
``` sh
poetry install -E full
```
This will download all the needed dependencies in a sandboxed environment (the
`-E full` flag is for the optional dependencies). When you want to use
``PostCactus``, just run ``poetry shell``. This will drop you in a shell in
which you have full access to ``PostCactus`` in "development" version, and its
dependencies (also the one needed only for development).

## Documentation

`PostCactus` uses Sphinx to generate the documentation. To produce the documentation
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

## Credits

The code was originally developed by Wolfgang Kastaun. This fork completely
rewrites the original code, adding emphasis on documentation, testing, and
extensibility. The icon in the logo was designed by [freepik.com](freepik.com).

