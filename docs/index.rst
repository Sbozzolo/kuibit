Overview
========

PostCactus is a set of tools to post-process simulations performed with the
`Einstein Toolkit <https://einsteintoolkit.org/>`_. The code was originally
written by Wolfgang Kastaun. This is fork with emphasis on documentation and
testing.


Features
--------

Features currently implemented:

- Read and organize simulation data (:py:mod:`~.simdir`). Checkpoints and
  restarts are handled transparantely.
- Represent and manipulate time series (:py:mod:`~.timeseries`). Examples of
  functions available for time series: ``zero_pad``. ``time_shift``,
  ``phase_shift``, ``mean_remove``, ``integrate``, ``derive``, ``resample``,
  ``to_FrequencySeries`` (Fourier transform).
- Represent and manipulate frequency series (:py:mod:`~.frequencyseries`), like
  Fourier transforms of time series. Inverse Fourier transform is available.
- Manipulate and analyze gravitational-waves (:py:mod:`~.gw_utils`).
- Handle unit conversion, in particular from geometrized to physical
  (:py:mod:`~.unitconv`).

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

   simdir.rst
   series.rst
   cactus_scalars.rst
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

   simdir_ref.rst
   cactus_scalars_ref.rst
   series_ref.rst
   timeseries_ref.rst
   frequencyseries_ref.rst
   gw_utils_ref.rst
   unitconv_ref.rst
