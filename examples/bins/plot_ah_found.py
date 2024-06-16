#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2021-2024 Gabriele Bozzola
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
    desc = f"""\
{kah.get_program_name()} plots the times intervals at which the different given apparent horizons were found. """

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "-a",
        "--horizons",
        type=int,
        required=True,
        help="Apparent horizons to plot",
        nargs="+",
    )

    args = kah.get_args(parser)
    setup_matplotlib(rc_par_file=args.mpl_rc_file)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    horizons = "_".join([str(h) for h in args.horizons])
    figname = get_figname(args, default=f"ah_{horizons}_found")
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

        for ah in args.horizons:
            if ah in sim_hor.available_apparent_horizons:
                logger.debug(f"Reading horizon {ah}")
                # We can use any index for the qlm index, it will be thrown away
                current_horizon = sim_hor.get_apparent_horizon(ah)
                time_found = current_horizon.ah.cctk_iteration.t
                # We prepare an array with the same length of time_found and with
                # constant value of ah
                ah_num = [ah] * len(time_found)
                plt.scatter(time_found, ah_num, marker="o", s=0.1)

        # Plot
        logger.debug("Plotting")
        plt.ylabel("Apparent horizon")
        plt.xlabel("Time")

        # Fix ticks
        plt.gca().tick_params(axis="y", which="minor", left=False)
        plt.ylim(min(args.horizons) - 1, max(args.horizons) + 1)
        plt.yticks(args.horizons)

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
