Overview
========

PostCactus is a set of tools to post-process simulations performed with the
`Einstein Toolkit <https://einsteintoolkit.org/>`. The code was originally
written by Wolfgang Kastaun. This is smaller fork with emphasis on documentation
and testing.

Features
--------

Features currently implemented:

- Handle unit conversion, in particular from geometrized to physical
  (``postcactus.unitconv``)

Installation
------------

Clone the repo_:

.. _repo: https://github.com/Sbozzolo/PostCactus:

.. code-block:: bash

   git clone https://github.com/Sbozzolo/PostCactus.git

Move into the folder and install with pip:

.. code-block:: bash

   cd PostCactus && pip3 install --user .

Usage
-----

.. toctree::
   :maxdepth: 1

   unitconv.rst
