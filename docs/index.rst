Overview
========

PostCactus is a set of tools to post-process simulations performed with the
`Einstein Toolkit <https://einsteintoolkit.org/>`_. The code was originally
written by Wolfgang Kastaun. This is smaller fork with emphasis on documentation
and testing.


Features
--------

Features currently implemented:

- Represent and manipulate time series (``postcactus.timeseries``). Time series
  can be manipulated with all the mathematical expressions and they can be
  called as normal functions (``rho(t0)`` with ``rho`` being a ``TimeSeries``).
  Examples of functions available for time series: ``zero_pad``. ``time_shift``,
  ``phase_shift``, ``mean_remove``, ``integrate``, ``derive``, ``resample``.
- Manipulate gravitational-waves (``postcactus.gw_utils``). Examples of
  functions available: 
- Handle unit conversion, in particular from geometrized to physical
  (``postcactus.unitconv``).

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

   timeseries.rst
   gw_utils.rst
   unitconv.rst

Examples
--------

.. toctree::
   :maxdepth: 1

   examples/timeseries.ipynb


Reference material (classes, functions, ...)
---------------------------------------------

.. toctree::
   :maxdepth: 1

   timeseries_ref.rst
   gw_utils_ref.rst
   unitconv_ref.rst
