#!/usr/bin/env python3

# Copyright (C) 2020 Gabriele Bozzola
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

"""
import sys

import configargparse

# We use configargparse instead of argparse because it gives us much more
# flexibility.


def init_argparse(description=None):
    """Initialize a new argparse with given description and some basic default
    options.

    :param description: Description in the --help
    :type description: str

    :returns: Argparse parser
    :rtype: configargparse.ArgumentParser

    """
    parser = configargparse.ArgParser(description=description)
    parser.add(
        "-c", "--configfile", is_config_file=True, help="Config file path"
    )
    parser.add(
        "-v", "--verbose", help="Enable verbose output", action="store_true"
    )
    parser.add_argument("--datadir", default=".", help="Data directory")
    parser.add_argument("--outdir", default=".", help="Output directory")
    return parser


def get_args(parser, args=None):
    """Process argparse arguments.

    If args is None, the command line arguments are used.
    Otherwise, args is used (useful for testing and debugging).

    :returns: Arguments as read from command line or from args
    :rtype: argparse Namespace
    """

    if args is None:
        # Remove the name of the program from the list of arguments
        args = sys.argv[1:]
    return parser.parse_args(args)


def add_grid_to_parser(parser):
    """Add parameters that have to do with grid configurations to a given
    parser.

    This function edits parser in place.

    :param parser: Argparse parser
    :type parser: configargparse.ArgumentParser

    """
    parser.add_argument(
        "--resolution",
        type=int,
        default=500,
        help=(
            "Resolution of the image in number of points "
            + "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--plane",
        type=str,
        choices=["xy", "xz"],
        default="xy",
        help="Plane to plot (default: %(default)s)",
    )
    parser.add_argument(
        "-x0",
        "--origin",
        type=float,
        nargs=2,
        default=[0, 0],
    )
    parser.add_argument(
        "-x1",
        "--corner",
        type=float,
        nargs=2,
        default=[1, 1],
    )


def add_figure_to_parser(parser, default_figname=None):
    """Add parameters that have to do with a figure as output to a given parser.

    This function edits parser in place.

    :param default_figname: Default name of the output figure.
    :type default_figname: str
    :param parser: Argparse parser (generated with init_argparse())
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
        help="Extension of the output figure (default: %(default)s)."
        " This is ignored when the output is a TikZ figure, "
        "in which case the extension is .tikz",
    )
    parser.add_argument(
        "--as-tikz",
        action='store_true',
        help="Save figure as TikZ figure"
    )
