Overview
========

PostCactus is a set of tools to post-process simulations performed with the
`Einstein Toolkit <https://einsteintoolkit.org/>`_. The code was originally
developed by Wolfgang Kastaun. This is a fork with emphasis on documentation and
testing.


Features
--------

Features currently implemented:

- Read and organize simulation data (:py:mod:`~.simdir`). Checkpoints and
  restarts are handled transparantely.
- Work with scalar data as produced by ``CarpetASCII`` (:py:mod:`~.cactus_scalars`).
- Analyze the multipololar decompositions output by ``Multipoles``
  (:py:mod:`~.cactus_multipoles`).
- Analyze gravitational waves extracted with the Newman-Penrose formalism
  (:py:mod:`~.cactus_waves`) computing, among the other things, strains, overlaps,
  energy lost.
- Work with the power spectral densities of known detectors
  (:py:mod:`~.sensitivity_curves`)
- Represent and manipulate time series (:py:mod:`~.timeseries`). Examples of
  functions available for time series: ``integrate``, ``derive``, ``resample``,
  ``to_FrequencySeries`` (Fourier transform).
- Represent and manipulate frequency series (:py:mod:`~.frequencyseries`), like
  Fourier transforms of time series. Inverse Fourier transform is available.
- Manipulate and analyze gravitational-waves (:py:mod:`~.gw_utils`,
  :py:mod:`~.gw_mismatch`). For example, compute energies, mismatches, or
  extrapolate waves to infinity.
- Work with grid functions (:py:mod:`~.grid_data`).
- Handle unit conversion, in particular from geometrized to physical
  (:py:mod:`~.unitconv`).

Installation
------------

`PostCactus` is available in TestPyPI. To install it with `pip`
.. code-block:: bash

   pip3 install --index-url https://test.pypi.org/simple/ postcactus

If they are not already available, ``pip`` will install the following packages:
- ``numpy``,
- ``numba``,
- ``h5py``,
- ``scipy``.

The minimum version of Python required is 3.5.

If you intend to extend/develop ``PostCactus``, follow the instruction on `GitHub <https://github.com/Sbozzolo/PostCactus>`_.

Usage
-----

.. toctree::
   :maxdepth: 1

   simdir.rst
   series.rst
   cactus_scalars.rst
   cactus_multipoles.rst
   cactus_waves.rst
   gw_utils.rst
   gw_mismatch.rst
   grid_data.rst
   sensitivity_curves.rst
   unitconv.rst

Examples
--------

.. toctree::
   :maxdepth: 1

   examples/simdir.ipynb
   examples/timeseries.ipynb
   examples/gravitational_waves.ipynb


Reference material (classes, functions, ...)
---------------------------------------------

.. toctree::
   :maxdepth: 1

   simdir_ref.rst
   series_ref.rst
   timeseries_ref.rst
   frequencyseries_ref.rst
   cactus_scalars_ref.rst
   cactus_multipoles_ref.rst
   cactus_waves_ref.rst
   gw_utils_ref.rst
   gw_mismatch_ref.rst
   grid_data_ref.rst
   sensitivity_curves_ref.rst
   unitconv_ref.rst
