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

import logging
import os

import matplotlib.pyplot as plt
from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    plot_color,
    save_from_dir_filename_ext,
    setup_matplotlib,
    get_figname,
)

if __name__ == "__main__":
    setup_matplotlib()

    desc = f"""{kah.get_program_name()} plots or more 1D grid functions output by Carpet."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser)

    parser.add_argument(
        "--variables",
        type=str,
        required=True,
        help="Variables to plot.",
        nargs="+",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to plot. If -1, the latest.",
    )
    parser.add(
        "--logscale", help="Use a logarithmic y scale.", action="store_true"
    )
    parser.add(
        "--vmin",
        help=(
            "Minimum value of the variable. "
            "If logscale is True, this has to be the log."
        ),
        type=float,
    )
    parser.add(
        "--vmax",
        help=(
            "Maximum value of the variable. "
            "If logscale is True, this has to be the log."
        ),
        type=float,
    )
    parser.add(
        "--xmin",
        help=("Minimum coordinate."),
        type=float,
    )
    parser.add(
        "--xmax",
        help=("Maximum coordinate."),
        type=float,
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Whether to take the absolute value.",
    )
    parser.add_argument(
        "--axis",
        type=str,
        choices=["x", "y", "z"],
        default="x",
        help="Axis to plot (default: %(default)s).",
    )
    args = kah.get_args(parser)

    iteration = args.iteration

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)


    var_names = "_".join(args.variables)
    figname = get_figname(args, default=f"{var_names}_{args.axis}")

    sim = SimDir(args.datadir, ignore_symlinks=args.ignore_symlinks)
    logger.debug("Prepared SimDir")

    reader = sim.gridfunctions[args.axis]

    for var in args.variables:
        if var not in reader:
            raise ValueError(f"{var} is not available")

    for variable in args.variables:

        logger.debug(f"Reading variable {variable}")
        logger.debug(f"Variables available {reader}")
        var = reader[variable]
        logger.debug(f"Read variable {variable}")

        if iteration == -1:
            iteration = var.available_iterations[-1]

        time = var.time_at_iteration(iteration)

        logger.debug(f"Using iteration {iteration} (time = {time})")

        if args.absolute:
            data = abs(var[iteration])
            variable_name = f"abs({variable})"
        else:
            data = var[iteration]
            variable_name = variable

        logger.debug("Merging refinement levels")
        data = data.merge_refinement_levels().to_GridSeries()

        if args.logscale:
            label = f"log10({variable_name})"
            data = data.log10()
        else:
            label = variable_name

        logger.debug(f"Using label {label}")

        logger.debug(f"Plotting variable {variable}")
        plt.plot(data, label=label)
        logger.debug("Plotted")

    add_text_to_corner(fr"$t = {time:.3f}$")

    plt.legend()
    plt.xlabel(args.axis)
    plt.ylim(ymin=args.vmin, ymax=args.vmax)
    plt.xlim(xmin=args.xmin, xmax=args.xmax)


    logger.debug("Saving")
    save_from_dir_filename_ext(args.outdir, figname, args.fig_extension)
    logger.debug("DONE")
