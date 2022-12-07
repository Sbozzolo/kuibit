#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2022 Gabriele Bozzola
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
import numpy as np

from kuibit import argparse_helper as kah
from kuibit.hor_utils import compute_angular_velocity_vector
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    get_figname,
    save_from_dir_filename_ext,
    set_axis_limits_from_args,
    setup_matplotlib,
)

if __name__ == "__main__":

    desc = f"""\
{kah.get_program_name()} plots the period as on the equatorial plane as computed
from Newtonian angular velocity of the equivalent Kepler problem. This
calculation typically has some very large numbers, this script does its best to
choose the bounds, but often you will need to manually specify --ymax and
--ymin."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "-a",
        "--horizons",
        type=int,
        required=True,
        help="Apparent horizons to plot",
        nargs=2,
    )

    args = kah.get_args(parser)
    setup_matplotlib(rc_par_file=args.mpl_rc_file)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(
        args, default=f"ah_{args.horizons[0]}_{args.horizons[1]}_period"
    )
    logger.debug(f"Using figname {figname}")

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
        for ah in args.horizons:
            if ah not in sim_hor.available_apparent_horizons:
                raise ValueError(f"Apparent horizons {ah} is not available")

        logger.debug("Reading horizons and computing velocity")

        h1 = sim_hor.get_apparent_horizon(args.horizons[0])
        h2 = sim_hor.get_apparent_horizon(args.horizons[1])

        Omega = compute_angular_velocity_vector(h1, h2, resample=True)

        logger.debug("Plotting period")
        period = 2 * np.pi / Omega[2]
        median = np.median(period.y)
        logger.debug(f"Median {median:.3f}")

        # Set args.ymin/args.ymax if they are None
        args.ymin = args.ymin or 0
        args.ymax = args.ymax or 2 * median

        plt.plot(period, label="Omega_z")

        plt.legend()
        plt.ylabel("Period")
        plt.xlabel("Time")
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
