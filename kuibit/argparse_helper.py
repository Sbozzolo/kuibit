#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

"""The :py:mod:`~.argparse_helper` module provides helper functions to write
scripts that are controlled by command-line arguments or configuration files.

The intended way to use this module is, schematically

.. code-block:: python

    from kuibit import argparse_helper as kah

    parser = kah.init_argparse(desc="Description")
    # Next we add everything we need
    kah.add_figure_to_parser(parser)

    # Specific arguments
    parser.add_argument("--arg1", help="Specific argument")

    # Finally
    args = kah.get_args(parser)

    # args is Namespace that contains all the arguments provided via
    # command-line, configuration file, or environment variable

"""
import os
import sys

import configargparse

# We use configargparse instead of argparse because it gives us much more
# flexibility.


def init_argparse(*args, **kwargs):
    """Initialize a new argparse with given arguments.

    Unknown arguments are passed to ``configargparse.ArgParser``.

    :returns: Argparse parser.
    :rtype: configargparse.ArgumentParser

    """
    parser = configargparse.ArgParser(*args, **kwargs)
    parser.add(
        "-c", "--configfile", is_config_file=True, help="Config file path"
    )
    parser.add(
        "-v", "--verbose", help="Enable verbose output", action="store_true"
    )
    parser.add_argument("--datadir", default=".", help="Data directory")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument(
        "--ignore-symlinks",
        action="store_true",
        help="Ignore symlinks in the data directory",
    )
    return parser


def get_args(parser, args=None):
    """Process argparse arguments.

    If ``args`` is None, the command line arguments are used. Otherwise,
    ``args`` is used (useful for testing and debugging).

    :param args: List of command-line options.
    :type args: list

    :returns: Arguments as read from command line or from args.
    :rtype: argparse Namespace

    """

    if args is None:
        # Remove the name of the program from the list of arguments
        args = sys.argv[1:]
    return parser.parse_args(args)


def add_grid_to_parser(parser, dimensions=2):
    """Add parameters that have to do with grid configurations to the given parser.

    This function edits ``parser`` in place.

    The options added are:

    - ``resolution``
    - ``x0 (origin)``
    - ``x1 (corner)
    - ``axis`` (for ``dimension = 1``)
    - ``plane`` (for ``dimension = 2``)

    :param parser: Argparse parser to which the grid options have to be added.
    :type parser: configargparse.ArgumentParser

    :param dimensions: Number of grid dimensions to consider (1, 2, or 3).
    :type dimensions: int

    """
    if dimensions not in (1, 2, 3):
        raise ValueError("The number of dimensions has to be 1, 2, or 3")

    parser.add_argument(
        "--resolution",
        type=int,
        default=500,
        help=(
            (
                "Resolution of the grid in number of points "
                "(default: %(default)s)"
            )
        ),
    )

    parser.add_argument(
        "-x0",
        "--origin",
        type=float,
        nargs=dimensions,
        default=[0] * dimensions,
    )
    parser.add_argument(
        "-x1",
        "--corner",
        type=float,
        nargs=dimensions,
        default=[1] * dimensions,
    )

    if dimensions == 1:
        parser.add_argument(
            "--axis",
            type=str,
            choices=["x", "y", "z"],
            default="x",
            help="Axis to plot (default: %(default)s)",
        )

    if dimensions == 2:
        parser.add_argument(
            "--plane",
            type=str,
            choices=["xy", "xz", "yz"],
            default="xy",
            help="Plane to plot (default: %(default)s)",
        )


def add_figure_to_parser(parser, default_figname=None):
    """Add parameters that have to do with a figure as output to a given parser.

    This function edits ''parser'' in place.

    The options added are:

    - ``figname``
    - ``fig-extension
    - ``as-tikz

    :param default_figname: Default name of the output figure.
    :type default_figname: str
    :param parser: Argparse parser (generated with :py:func:`~.init_argparse`).
    :type parser: configargparse.ArgumentParser

    """
    parser.add_argument(
        "--figname",
        type=str,
        default=default_figname,
        help="Name of the output figure (not including the extension).",
    )
    parser.add_argument(
        "--fig-extension",
        type=str,
        default="png",
        env_var="KBIT_FIG_EXTENSION",
        help="Extension of the output figure (default: %(default)s)."
        " This is ignored when the output is a TikZ figure, "
        "in which case the extension is .tikz",
    )
    parser.add_argument(
        "--as-tikz", action="store_true", help="Save figure as TikZ figure"
    )


def add_horizon_to_parser(
    parser, color="k", edge_color="w", alpha=1, time_tolerance=0.1
):
    """Add parameters that have to do with a apparent horizons to a given parser.

    This function edits ``parser`` in place.

    The options added are:

    - ``ah-show``
    - ``ah-color``
    - ``ah-edge-color``
    - ``ah-alpha``
    - ``ah-time-tolerance``

    :param color: Color of the horizons.
    :type color: anything accepted by the drawing package
    :param edge_color: Color of the edge of the horizons.
    :type edge_color: anything accepted by the drawing package
    :param alpha: Number between 0 and 1 that identifies the opacity of the
                  horizon.
    :type alpha: float
    :param time_tolerance: Time tolerance allowed for finding an horizon.
    :type time_tolerance: float
    :param parser: Argparse parser (generated with init_argparse())
    :type parser: configargparse.ArgumentParser

    """
    ah_group = parser.add_argument_group("Horizon options")
    ah_group.add_argument(
        "--ah-show", action="store_true", help="Plot apparent horizons."
    )
    ah_group.add_argument(
        "--ah-color",
        default=color,
        help="Color name for horizons (default is '%(default)s').",
    )
    ah_group.add_argument(
        "--ah-edge-color",
        default=edge_color,
        help="Color name for horizons boundary (default is '%(default)s').",
    )
    ah_group.add_argument(
        "--ah-alpha",
        type=float,
        default=alpha,
        help="Alpha (transparency) for apparent horizons (default: %(default)s)",
    )
    ah_group.add_argument(
        "--ah-time-tolerance",
        type=float,
        default=time_tolerance,
        help="Tolerance for matching horizon time [simulation units]"
        " (default is '%(default)s').",
    )
    return parser


def get_program_name():
    """Return the name of the current script.

    :returns: Name of file executing the code.
    :rtype: str
    """
    return os.path.basename(sys.argv[0])
