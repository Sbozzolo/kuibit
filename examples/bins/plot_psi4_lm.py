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
    save_from_dir_filename_ext,
    setup_matplotlib,
    get_figname,
)


if __name__ == "__main__":
    setup_matplotlib()

    desc = """{kah.get_program_name()} plots the multipolar decomposition of Psi4 as
    measured by a given detector and a given l and m."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser)

    parser.add_argument(
        "--detector-num",
        type=int,
        required=True,
        help="Number of the spherical surface over which to read Psi4.",
    )

    parser.add_argument(
        "--mult-l", type=int, default=2, help="Multipole number l."
    )
    parser.add_argument(
        "--mult-m", type=int, default=2, help="Multipole number m."
    )

    args = kah.get_args(parser)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(
        args,
        default=f"Psi4_{args.mult_l}{args.mult_m}_det{args.detector_num}"
    )
    logger.debug(f"Using figname {figname}")

    sim = SimDir(args.datadir, ignore_symlinks=args.ignore_symlinks)
    logger.debug("Prepared SimDir")

    reader = sim.gravitationalwaves

    radius = reader.radii[args.detector_num]
    logger.debug(f"Using radius: {radius}")
    detector = reader[radius]

    if (args.mult_l, args.mult_m) not in detector.available_lm:
        logger.debug(f"Available multipoles {detector.available_lm}")
        raise ValueError(
            f"Multipole {args.mult_l}, {args.mult_m} not available"
        )

    psi4 = detector[args.mult_l, args.mult_m]

    logger.debug("Plotting Psi4")

    plt.plot(
        psi4.real(), label=fr"$\Re \Psi_4^{{{args.mult_l}{args.mult_l}}}$"
    )
    plt.plot(
        psi4.imag(), label=fr"$\Im \Psi_4^{{{args.mult_l}{args.mult_l}}}$"
    )

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(r"$r \Psi_4$")
    logger.debug("Plotted")

    logger.debug("Saving")
    save_from_dir_filename_ext(args.outdir, figname, args.fig_extension)
    logger.debug("DONE")
