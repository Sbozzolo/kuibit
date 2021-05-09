<p align="center">
<img src="https://github.com/Sbozzolo/kuibit/raw/master/logo.png" height="120">
</p>

[![DOI](https://joss.theoj.org/papers/10.21105/joss.03099/status.svg)](https://doi.org/10.21105/joss.03099)
[![codecov](https://codecov.io/gh/Sbozzolo/kuibit/branch/master/graph/badge.svg)](https://codecov.io/gh/Sbozzolo/kuibit)
![Tests](https://github.com/Sbozzolo/kuibit/workflows/Tests/badge.svg)
![Documentation](https://github.com/Sbozzolo/kuibit/workflows/Document/badge.svg)
[![GPLv3
license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![Get help on Telegram](https://img.shields.io/badge/Get%20help%20on-Telegram-blue.svg)](https://t.me/kuibit)
[![PyPI version](https://badge.fury.io/py/kuibit.svg)](https://badge.fury.io/py/kuibit)
[![DeepSource](https://deepsource.io/gh/Sbozzolo/kuibit.svg/?label=active+issues)](https://deepsource.io/gh/Sbozzolo/kuibit/?ref=repository-badge)

# kuibit

`kuibit` is a Python library to analyze simulations performed with the Einstein
Toolkit largely inspired by
[PostCactus](https://github.com/wokast/PyCactus/tree/master/PostCactus).
`kuibit` can read simulation data and represent it with high-level classes. For
a list of features available, look at the [official
documentation](https://sbozzolo.github.io/kuibit). For examples and tools that
are ready to be used, you can download the archive that is attached to the
[release](https://github.com/sbozzolo/kuibit/releases/latest/download/examples.tar.gz),
or you can look the [relevant section of the
documentation](https://sbozzolo.github.io/kuibit/#id1). The [testimonials
page](https://sbozzolo.github.io/kuibit/testimonials.html) collects short
reviews about `kuibit`.

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
``kuibit``, just run ``poetry shell`` from within the `kuibit` directory.
This will drop you in a shell in
which you have full access to ``kuibit`` in "development" version, and its
dependencies (including the one needed only for development). Alternatively, you
can activate the virtual environment directly. You can find where the environment
in installed running the command `poetry env info --path` in the `kuibit` directory.
This is a standard virtual environment, which can be activated with the `activate`
scripts in the `bin` folder. Once you do that, you will be able to use `kuibit`
for anywhere.

## Help!

Users and developers of ``kuibit`` meet in the [Telegram
group](https://t.me/kuibit). If you have any problem or suggestion, that's a
good place where to discuss it. Alternatively, you can also open an issue on
GitHub.

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

## Videos

Here is a list of videos describing `kuibit` and how to use it:
- [Introduction on kuibit - Einstein Toolkit Seminar, 2021](https://www.youtube.com/watch?v=7-F2xh-m31A)

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

## What is a _kuibit_?

A kuibit (also known as _kukuipad_, meaning harvest pole) is the tool
traditionally used by the Tohono O'odham people to reach the fruit of the
Saguaro cacti during the harvesting season. In the same way, this package is a
tool that you can use to collect the fruit of your `Cactus` simulations.

## Credits

`kuibit` follows the same design and part of the implementation details of
`PostCactus`, code developed by Wolfgang Kastaun. This fork completely rewrites
the original code, adding emphasis on documentation, testing, and extensibility.
The logo contains elements designed by [freepik.com](freepik.com). We thank
``kuibit`` first users, Stamatis Vretinaris and Pedro Espino, for providing
comments to improve the code and the documentation.

## Citation

`kuibit` is built and maintained by the dedication of one graduate student. Please,
consider citing `kuibit` if you find the software useful. You can use the following
`bibtex` key (as provided by ADSABS).
``` bibtex
@article{kuibit,
       author = {{Bozzola}, Gabriele},
        title = "{kuibit: Analyzing Einstein Toolkit simulations with Python}",
      journal = {The Journal of Open Source Software},
     keywords = {numerical relativity, Python, Einstein Toolkit, astrophysics, Cactus, General Relativity and Quantum Cosmology, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2021,
        month = apr,
       volume = {6},
       number = {60},
          eid = {3099},
        pages = {3099},
          doi = {10.21105/joss.03099},
archivePrefix = {arXiv},
       eprint = {2104.06376},
 primaryClass = {gr-qc},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021JOSS....6.3099B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
You can find this entry in Python with `from kuibit import __bibtex__`.
