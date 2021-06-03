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
The YouTube series
`Using kuibit <https://www.youtube.com/playlist?list=PLIVVtc6RlFxpi3BiF6KTWd3z4TTQ5hY06>`_
contains video tutorials on ``kuibit``.

The :doc:`testimonials page <testimonials>` collects short user's reviews about
``kuibit``.

Summary of Features
-------------------

For a full list of available features, see the :doc:`features page <features>`.

- Read and organize simulation data (:py:mod:`~.simdir`). Checkpoints and
  restarts are handled transparently.
- Work with scalar data as produced by ``CarpetIOASCII``
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
- Write command-line scripts (:py:mod:`~.argparse_helper`).
- Visualize data with ``matplotlib`` (:py:mod:`~.visualize_matplotlib`).
- Make movies with  `motionpicture`_.

.. _motionpicture: https://github.com/Sbozzolo/motionpicture

Installation
------------

``kuibit`` is available in PyPI. To install it with `pip`

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
   motionpicture.rst

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

The YouTube series
`Using kuibit <https://www.youtube.com/playlist?list=PLIVVtc6RlFxpi3BiF6KTWd3z4TTQ5hY06>`_
contains video tutorials on ``kuibit``. Each video is focused on a single topic.


Examples
---------

Below you will find a list of examples to perform more or less common analysis.
You can immediately start doing science without writing one line of code using
these examples. The scripts provided can be used for plotting, extracting
gravitational waves, or other useful information. To get the most out of these
examples, check out the :doc:`recommendations on how to
use the examples <recommendation_examples>` page.

Note that all these examples contain a significant fraction of boilerplate that
is needed to keep them general and immediately useful. When learning ``kuibit``,
you can ignore all of this.

You can download these examples as archive from the `GitHub release page
<https://github.com/sbozzolo/kuibit/releases/latest/download/examples.tar.gz>`_,
which is automatically updated with each release.

Scripts
^^^^^^^

.. toctree::
   :maxdepth: 1

   examples/bins/plot_1d_vars.rst
   examples/bins/plot_ah_coordinate_velocity.rst
   examples/bins/plot_ah_found.rst
   examples/bins/plot_ah_radius.rst
   examples/bins/plot_ah_separation.rst
   examples/bins/plot_constraints.rst
   examples/bins/plot_em_energy.rst
   examples/bins/plot_grid_var.rst
   examples/bins/plot_gw_energy.rst
   examples/bins/plot_gw_linear_momentum.rst
   examples/bins/plot_phi_lm.rst
   examples/bins/plot_physical_time.rst
   examples/bins/plot_psi4_lm.rst
   examples/bins/plot_strain_lm.rst
   examples/bins/plot_timeseries.rst
   examples/bins/plot_total_luminosity.rst
   examples/bins/print_ah_formation_time.rst
   examples/bins/print_available_iterations.rst
   examples/bins/print_available_timeseries.rst
   examples/bins/print_qlm_properties_at_time.rst
   examples/bins/save_resampled_grid_data.rst

Movies
^^^^^^

`motionpicture`_ is a Python library to make animations used by ``kuibit``. To
learn more about ``motionpicture`` and how to use it, read the :doc:`quick
introduction to motionpicture <motionpicture>`.

.. _motionpicture: https://github.com/Sbozzolo/motionpicture

.. toctree::
   :maxdepth: 1

   examples/mopi_movies/grid_var.rst

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

Citation
------------

``kuibit`` is built and maintained by the dedication of one graduate student. Please,
consider citing ``kuibit`` if you find the software useful. You can use the following
``bibtex`` key.

.. code-block:: bibtex

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
