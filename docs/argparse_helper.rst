Building command-line scripts
=============================

``kuibit`` is a library and it is designed to assist you in performing your
analyses. Hence, ``kuibit`` supports multiple workflows. A possible choice
consists in writing command-line executables to perform specific tasks (most of
the scripts in ``examples/bins`` are built for this kind of workflow, so you can
use them as example, or you can directly use them). For example, it is useful to
have a script ``plot_grid.py`` that takes a simulation folder and a variable
name and plots a 2D snapshot of such variable. The module
:py:mod:`~.argparse_helper` contains functions to make writing such scripts
easier (:ref:`argparse_helper_ref:Reference on kuibit.argparse_helper`).

Argparse and argparse_helper
----------------------------

``argparse`` is a built-in module in Python used to read information from the
command line. :py:mod:`~.argparse_helper` provides functions that automatically
populate ``argparse`` with common options and reasonable defaults. The minimum
working example is simply

.. code-block:: python

   from kuibit import argparse_helper as kah

   desc = "This is an example"
   parser = kah.init_argparse(desc)
   args = kah.get_args(parser)

This will populate the options ``configfile``, ``verbose``, ``datadir``,
``outdir``, and ``ignore_symlinks``, which can be accessed with ``args.varname``
(e.g., ``args.verbose``), or with ``args.fields['varname']``. ``kuibit``
supports being configured with configfiles, which can be useful to control
scripts with many options. The method :py:meth:`~.get_args` takes the parser
object and process the command-line data.

``parser`` can be extended with more common options. For example, if you are
working with grid data, you will likely will want to specify grid extents. To
add common grid options, use the method :py:meth:`~.add_grid_to_parser`, which
adds arguments like ``origin``, ``corner``, and ``resolution``.

.. code-block:: python

   from kuibit import argparse_helper as kah

   desc = "This is an example"
   parser = kah.init_argparse(desc)
   kah.add_grid_to_parser(parser)
   args = kah.get_args(parser)

Often, the default arguments are not enough and you want to add your own. This
can be achieved with the method ``add_argument``, as in the following example:

.. code-block:: python

   from kuibit import argparse_helper as kah

   desc = "This is an example"
   parser = kah.init_argparse(desc)
   parser.add_argument(
        "--variable", type=str, required=True, help="Variable to plot"
   )
   kah.add_grid_to_parser(parser)
   args = kah.get_args(parser)


Groups of options available
----------------------------

The groups of options that are currently available are:

- :py:func:`~.add_grid_to_parser`, for operations that involve grids
- :py:func:`~.add_figure_to_parser`, for operations that involve making figures
- :py:func:`~.add_horizon_to_parser`, for operations that involve plotting
  horizons

You should check the :ref:`argparse_helper_ref:Reference on
kuibit.argparse_helper` to figure out exactly what options are added by each
function.
