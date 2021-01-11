Overview
========

``kuibit`` is a set of tools to post-process simulations performed with the
`Einstein Toolkit <https://einsteintoolkit.org/>`_.

The goal of this package is to enable you to pursue your scientific goals
without having to worry about computational details (e.g., handling simulation
restarts, reading HDF5 files, ...). ``kuibit`` represent simulation data in
a high-level and intuitive way, and provides some commonly used routines in
numerical-relativity (e.g., computing the strain of gravitational waves).

Summary of Features
-------------------

For a full list of available features, see the :doc:`features page <features>`.

- Read and organize simulation data (:py:mod:`~.simdir`). Checkpoints and
  restarts are handled transparently.
- Work with scalar data as produced by ``CarpetASCII``
  (:py:mod:`~.cactus_scalars`).
- Analyze the multipolar decompositions output by ``Multipoles``
  (:py:mod:`~.cactus_multipoles`).
- Analyze gravitational waves extracted with the Newman-Penrose formalism
  (:py:mod:`~.cactus_waves`) computing, among the other things, strains,
  overlaps, energy lost.
- Work with the power spectral densities of known detectors
  (:py:mod:`~.sensitivity_curves`).
- Represent and manipulate time series (:py:mod:`~.timeseries`). Examples of
  functions available for time series: ``integrate``, ``derive``, ``resample``,
  ``to_FrequencySeries`` (Fourier transform).
- Represent and manipulate frequency series (:py:mod:`~.frequencyseries`), like
  Fourier transforms of time series. Inverse Fourier transform is available.
- Manipulate and analyze gravitational-waves (:py:mod:`~.gw_utils`,
  :py:mod:`~.gw_mismatch`). For example, compute energies, mismatches, or
  extrapolate waves to infinity.
- Work with 1D, 2D, and 3D grid functions (:py:mod:`~.grid_data`,
  :py:mod:`~.cactus_grid_functions`) as output by ``CarpetIOHDF5`` or
  ``CarpetIOASCII``.
- Work with horizon data from (:py:mod:`~.cactus_horizons`) as output by
  ``QuasiLocalMeasures`` and ``AHFinderDirect``.
- Handle unit conversion, in particular from geometrized to physical
  (:py:mod:`~.unitconv`).

Installation
------------

``kuibit`` is available in TestPyPI. To install it with `pip`

.. code-block:: bash

   pip3 install --index-url https://test.pypi.org/simple/ kuibit

If they are not already available, ``pip`` will install all the necessary
dependencies.

The minimum version of Python required is 3.6.

If you intend to extend/develop ``kuibit``, follow the instruction on
`GitHub <https://github.com/Sbozzolo/kuibit>`_.

Usage
-----

.. toctree::
   :maxdepth: 1

   simdir.rst
   series.rst
   cactus_scalars.rst
   cactus_multipoles.rst
   cactus_horizons.rst
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
   examples/grid_data.ipynb
   examples/cactus_grid_functions.ipynb
   examples/cactus_horizons.ipynb
   examples/gravitational_waves.ipynb

Reference material (classes, functions, ...)
---------------------------------------------

.. toctree::
   :maxdepth: 1

   features.rst
   simdir_ref.rst
   series_ref.rst
   timeseries_ref.rst
   frequencyseries_ref.rst
   cactus_grid_functions_ref.rst
   cactus_scalars_ref.rst
   cactus_multipoles_ref.rst
   cactus_waves_ref.rst
   cactus_horizons_ref.rst
   gw_utils_ref.rst
   gw_mismatch_ref.rst
   grid_data_ref.rst
   sensitivity_curves_ref.rst
   unitconv_ref.rst

What is a kuibit?
-----------------

A kuibit (harvest pole) is the tool traditionally used by the Tohono O'odham
people to reach the fruit of the Saguaro cacti during the harvesting season. In
the same way, this package is a tool that you can use to collect the fruit of
your ``Cactus`` simulations.


Credits
-------

The code was originally developed by Wolfgang Kastaun. This fork completely
rewrites the original code, adding emphasis on documentation, testing, and
extensibility. The icon in the logo was designed by `freepik.com
<https://freepik.com/>`_.
