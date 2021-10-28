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
import numpy as np
from scipy.optimize import curve_fit

from kuibit import argparse_helper as kah
from kuibit.cactus_horizons import compute_horizons_separation
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    get_figname,
    save_from_dir_filename_ext,
    set_axis_limits_from_args,
    setup_matplotlib,
)

if __name__ == "__main__":
    setup_matplotlib()

    desc = f"""\ {kah.get_program_name()} plots the time derivative of coordinate
separation between the centroids of two given apparent horizons. Use this
quantity to estimate the eccentricity with the method described in the paper
gr-qc/0702106. Pass --xmin and --xmax to control where the fit is performed."""

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

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(
        args,
        default=f"ah_{args.horizons[0]}_{args.horizons[1]}_separation_derivative",
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

        logger.debug("Reading horizons and computing separation")

        h1 = sim_hor.get_apparent_horizon(args.horizons[0])
        h2 = sim_hor.get_apparent_horizon(args.horizons[1])

        separation = compute_horizons_separation(h1, h2, resample=True)

        separation.crop(init=args.xmin, end=args.xmax)
        derivative = separation.differentiated()

        logger.debug("Plotting derivative of separation")
        plt.plot(derivative)
        plt.ylabel("Derivative of the coordinate separation")
        plt.xlabel("Time")
        set_axis_limits_from_args(args)
        logger.debug("Plotted")

        def f_ecc(t, A0, A1, B, omega, phi):
            return A0 + A1 * t + B * np.sin(omega * t + phi)

        p0 = [0, 0.1, 0.02, 0.02, 0]

        (A0, A1, B, omega, phi), _ = curve_fit(
            f_ecc, derivative.t, derivative.y, p0
        )

        eccentricty = B / (omega * separation.y[0])

        plt.plot(
            derivative.t,
            f_ecc(derivative.t, A0, A1, B, omega, phi),
            label=f"e = {eccentricty:.3f}",
        )
        plt.legend()

        logger.debug("Saving")
        save_from_dir_filename_ext(
            args.outdir,
            figname,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
        logger.debug("DONE")
