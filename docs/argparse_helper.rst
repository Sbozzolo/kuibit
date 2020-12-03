Building command-line scripts
=============================

``PostCactus`` is a library and it is designed to assist you in performing your
analyses. Hence, ``PostCactus`` supports multiple workflows. A possible choice
consists in writing command-line executables to perform specific tasks. For
example, it is useful to have a script ``plot_grid.py`` that takes a simulation
folder and a variable name and plots a 2D snapshot of such variable. The module
:py:mod:`~.argparse_helper` contains functions to make writing such scripts
easier (:ref:`argparse_helper_ref:Reference on postcactus.argparse_helper`).

Argparse and argparse_helper
----------------------------

``argparse`` is a built-in module in Python used to read information from the
command line. :py:mod:`~.argparse_helper` provides functions that automatically
populate ``argparse`` with common options and reasonable defaults. The minimum
working example is simply

.. code-block:: python

   from postcactus import argparse_helper as pah

   desc = "This is an example"
   parser = pah.init_argparse(desc)
   args = pah.get_args(parser)

This will populate the options ``configfile``, ``verbose``, ``datadir``, and
``outdir``, which can be accessed with ``args.varname`` (e.g.,
``args.verbose``), or with ``args.fields['varname']``. ``PostCactus`` supports
being configured with configfiles, which can be useful to control scripts with
many options. The method :py:meth:`~.get_args` takes the parser object and
process the command-line data.

``parser`` can be extended with more common options. For example, if you are
working with grid data, you will likely will want to specify grid extents. To
add common grid options, use the method :py:meth:`~.add_grid_to_parser`, which
adds arguments like ``origin``, ``corner``, and ``resolution``.

.. code-block:: python

   from postcactus import argparse_helper as pah

   desc = "This is an example"
   parser = pah.init_argparse(desc)
   pah.add_grid_to_parser(parser)
   args = pah.get_args(parser)

Often, the default arguments are not enough and you want to add your own. This
can be achieved with the method ``add_argument``, as in the following example:

.. code-block:: python

   from postcactus import argparse_helper as pah

   desc = "This is an example"
   parser = pah.init_argparse(desc)
   parser.add_argument(
        "--variable", type=str, required=True, help="Variable to plot"
   )
   pah.add_grid_to_parser(parser)
   args = pah.get_args(parser)

Most of the scripts in ``example_bins`` are built for this kind of workflow, so
you can use them as examples (or you can directly use them).
