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
PostCactus with following differences:
    1. We use Python 3 instead of Python 2
    2. We test and document all the code
    3. We are overall fewer features, but also different ones
    
The goal of this fork is to make PostCactus more robust and user-friendly, hence
the emphasis on documentation and testing.

Documentation is found [here](https://sbozzolo.github.io/PostCactus). 

## Documentation

`PostCactus` uses Sphinx to generate the documentation. To produce the documentation
```sh
cd docs && make html
```
Documentation is automatically generated after each commit by GitHub Actions.

## Tests

`PostCactus` comes with a suite of unit tests. To run the tests, 
```sh
python3 -m unittest
```
Tests are automatically run after each commit by GitHub Actions.

