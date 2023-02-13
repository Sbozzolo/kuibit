#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2021-2022 Gabriele Bozzola
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
from kuibit.tensor import Vector
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    get_figname,
    save_from_dir_filename_ext,
    set_axis_limits_from_args,
    setup_matplotlib,
)

if __name__ == "__main__":
    desc = f"""{kah.get_program_name()} plots the coordinate velocity of a
given apparent horizon as a function of time."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "-a",
        "--horizon",
        type=int,
        required=True,
        help="Apparent horizons to plot.",
    )

    args = kah.get_args(parser)
    setup_matplotlib(rc_par_file=args.mpl_rc_file)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    ah = args.horizon
    figname = get_figname(args, default=f"ah_{ah}_coordinate_velocity")
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

        # Check that the horizon is available
        if ah not in sim_hor.available_apparent_horizons:
            raise ValueError(f"Apparent horizon {ah} is not available")

        logger.debug("Reading horizons and computing radius")

        hor = sim_hor.get_apparent_horizon(ah).ah

        cen = Vector([hor.centroid_x, hor.centroid_y, hor.centroid_z])
        vel = cen.differentiated()

        # Plot
        logger.debug("Plotting")
        plt.ylabel(f"Coordinate velocity of horizon {ah}")
        plt.xlabel("Time")
        plt.plot(vel[0], label=r"$v^x$")
        plt.plot(vel[1], label=r"$v^y$")
        plt.plot(vel[2], label=r"$v^z$")
        plt.plot(vel.magn(), label=r"$\|v\|$")
        plt.legend()

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
