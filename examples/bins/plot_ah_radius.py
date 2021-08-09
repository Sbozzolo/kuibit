#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2021 Gabriele Bozzola
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
    add_text_to_corner,
    get_figname,
    save_from_dir_filename_ext,
    set_axis_limits_from_args,
    setup_matplotlib,
)

if __name__ == "__main__":
    setup_matplotlib()

    desc = f"""{kah.get_program_name()} plots the coordinate radius of a given
apparent horizon as a function of time. If --dx is passed, then add a y-axis
indicating the number of points that resolve the radius, assuming that it is
all covered by the given resolution."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "-a",
        "--horizon",
        type=int,
        required=True,
        help="Apparent horizons to plot.",
    )

    parser.add_argument(
        "--dx",
        type=float,
        help=(
            "Grid resolution where the horizon is,"
            " assuming the entire horizon has this resolution."
        ),
    )

    args = kah.get_args(parser)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    ah = args.horizon
    figname = get_figname(args, default=f"ah_{ah}_radius")
    logger.debug(f"Figname: {figname}")

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:

        logger.debug("Prepared SimDir")
        sim_hor = sim.horizons

        logger.debug(
            f"Apparent horizons available: {sim_hor.available_apparent_horizons}"
        )

        # Check that the horizons are available
        if ah not in sim_hor.available_apparent_horizons:
            raise ValueError(f"Apparent horizons {ah} is not available")

        logger.debug("Reading horizons and computing radius")
        horizon = sim_hor.get_apparent_horizon(ah).ah

        # Plot
        logger.debug("Plotting")
        plt.ylabel(f"Radius of horizon {ah}")
        plt.xlabel("Time")
        plt.plot(horizon.mean_radius, label="Mean radius")
        plt.plot(horizon.min_radius, label="Min radius")
        plt.plot(horizon.max_radius, label="Max radius")
        plt.legend()

        if args.dx:
            logger.debug("Adding resolution y axis")
            plt.twinx()
            plt.plot(horizon.mean_radius / args.dx)
            plt.plot(horizon.min_radius / args.dx)
            plt.plot(horizon.max_radius / args.dx)
            plt.ylabel("Number of points on radius")

        add_text_to_corner(f"AH {ah}", anchor="SW", offset=0.005)

        set_axis_limits_from_args(args)
        logger.debug("Plotted")

        logger.debug("Saving")
        save_from_dir_filename_ext(
            args.outdir,
            figname,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
        logger.debug("DONE")
