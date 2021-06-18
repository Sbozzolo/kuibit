#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

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

import logging

import matplotlib.pyplot as plt
from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    get_figname,
    save_from_dir_filename_ext,
    set_axis_limits_from_args,
    setup_matplotlib,
)

if __name__ == "__main__":
    setup_matplotlib()

    desc = f"""\
{kah.get_program_name()} plots a given timeseries as output by CarpetIOASCII.
"""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "--variable", type=str, required=True, help="Variable to plot."
    )
    parser.add_argument(
        "--reduction",
        type=str,
        choices=[
            "scalar",
            "minimum",
            "maximum",
            "norm1",
            "norm2",
            "average",
            "infnorm",
        ],
        default="scalar",
        help="Reduction to plot.",
    )
    parser.add(
        "--logxaxis", help="Use a logarithmic x axis.", action="store_true"
    )
    parser.add(
        "--logyaxis", help="Use a logarithmic y axis.", action="store_true"
    )
    args = kah.get_args(parser)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    if args.reduction == "scalar":
        ext = ""
        red = ""
    else:
        ext = "_"
        red = args.reduction

    figname = get_figname(args, default=f"{args.variable}{ext}{red}")
    logger.debug(f"Using figname {figname}")

    logger.debug(f"Reading variable {args.variable}")
    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:

        logger.debug("Prepared SimDir")
        reader = sim.timeseries[args.reduction]
        logger.debug(f"Available variables {reader}")
        var = reader[args.variable]
        logger.debug(f"Read variable {args.variable}")

        logger.debug("Plotting timeseries")
        plt.plot(var)
        plt.xlabel("Time")
        plt.ylabel(f"{red} {args.variable}")
        if args.logxaxis:
            plt.xscale("log")
        if args.logyaxis:
            plt.yscale("log")
        set_axis_limits_from_args(args)
        logger.debug("Plotted")

        logger.debug("Saving")
        save_from_dir_filename_ext(args.outdir, figname, args.fig_extension)
        logger.debug("DONE")
