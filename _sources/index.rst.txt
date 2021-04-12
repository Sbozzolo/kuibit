Overview
========

``kuibit`` is a set of tools to post-process simulations performed with the
`Einstein Toolkit <https://einsteintoolkit.org/>`_.

The goal of this package is to enable you to pursue your scientific goals
without having to worry about computational details (e.g., handling simulation
restarts, reading HDF5 files, ...). ``kuibit`` represent simulation data in
a high-level and intuitive way, and provides some commonly used routines in
numerical-relativity (e.g., computing the strain of gravitational waves).
A video introduction about ``kuibit`` can be found on
`YouTube <https://www.youtube.com/watch?v=7-F2xh-m31A>`_.

The :doc:`testimonials page <testimonials>` collects short user's reviews about
``kuibit``.

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
  functions available for time series: ``integrate``, ``differentiate``, ``resample``,
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
- [EXPERIMENTAL] Visualize data with ``matplotlib``
  (:py:mod:`~.visualize_matplotlib`).
- [EXPERIMENTAL] Write command-line scripts (:py:mod:`~.argparse_helper`).
- [EXPERIMENTAL] Make movies with  `motionpicture`_.
- [EXPERIMENTAL] Run full analyses with  `ciak`_.

.. _motionpicture: https://github.com/Sbozzolo/motionpicture
.. _ciak: https://github.com/Sbozzolo/ciak

Installation
------------

``kuibit`` is available in PyPI. To install it with ``pip``

.. code-block:: bash

   pip3 install kuibit

If they are not already available, ``pip`` will install all the necessary
dependencies.

The minimum version of Python required is 3.6.

If you intend to extend/develop ``kuibit``, follow the instruction on
`GitHub <https://github.com/Sbozzolo/kuibit>`_.

Help!
------------

Users and developers of ``kuibit`` meet in the `Telegram group
<https://t.me/kuibit>`_. If you have any problem or suggestion, that's a good
place where to discuss it. Alternatively, you can also open an issue on GitHub.

Frequently asked questions are collected in the page :ref:`faq:Frequently Asked
Questions`.

In addition to the tutorials that are presented in this page, real-world
examples can be found `examples
<https://github.com/Sbozzolo/kuibit/tree/experimental/examples>`_ folder of the
``experimental`` branch. See the section `Experimental branch and examples
<https://github.com/Sbozzolo/kuibit/#experimental-branch-and-examples>`_ for
more information.

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
   visualize_matplotlib.rst
   argparse_helper.rst
   utils.rst

Tutorials
---------

.. toctree::
   :maxdepth: 1

   tutorials/simdir.ipynb
   tutorials/timeseries.ipynb
   tutorials/grid_data.ipynb
   tutorials/cactus_grid_functions.ipynb
   tutorials/cactus_horizons.ipynb
   tutorials/gravitational_waves.ipynb

In addition to these tutorials, you can find real world examples in the
`examples <https://github.com/Sbozzolo/kuibit/tree/experimental/examples>`_
folder of the ``experimental`` branch. See the section `Experimental branch and
examples
<https://github.com/Sbozzolo/kuibit/#experimental-branch-and-examples>`_ for
more information.

Reference material (classes, functions, ...)
---------------------------------------------

.. toctree::
   :maxdepth: 1

   testimonials.rst
   features.rst
   faq.rst
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
   visualize_matplotlib_ref.rst
   argparse_helper_ref.rst
   utils_ref.rst

What is a kuibit?
-----------------

A kuibit (also known as *kukuipad* harvest pole) is the tool traditionally used
by the Tohono O'odham people to reach the fruit of the Saguaro cacti during the
harvesting season. In the same way, this package is a tool that you can use to
collect the fruit of your ``Cactus`` simulations.


Credits
-------


``kuibit`` follows the same design and part of the implementation details of
``PostCactus``, code developed by Wolfgang Kastaun. This fork completely
rewrites the original code, adding emphasis on documentation, testing, and
extensibility. The logo contains elements designed by `freepik.com
<https://freepik.com/>`_. We thank ``kuibit`` first users, Stamatis Vretinaris
and Pedro Espino, for providing comments to improve the code and the
documentation.
