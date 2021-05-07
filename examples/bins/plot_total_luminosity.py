#!/usr/bin/env python3

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

    desc = f"""\
{kah.get_program_name()} plots the gravitational-wave plus the
electromagnetic-wave luminosity as a function of time for a given detector."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "--detector-num",
        type=int,
        required=True,
        help="Number of the spherical surface over which to read Psi4 and Phi2.",
    )

    parser.add_argument(
        "--pcut",
        type=int,
        required=True,
        help="Period that enters the fixed-frequency integration."
        " Typically, the longest physical period in the signal.",
    )

    args = kah.get_args(parser)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(
        args, default=f"tot_luminosity_det{args.detector_num}"
    )
    logger.debug(f"Using figname {figname}")

    sim = SimDir(args.datadir, ignore_symlinks=args.ignore_symlinks)
    logger.debug("Prepared SimDir")

    radius = sim.gravitationalwaves.radii[args.detector_num]
    logger.debug(f"Using radius {radius}")

    logger.debug("Computing GW power")
    power_gw = sim.gravitationalwaves[radius].get_total_power(args.pcut)
    logger.debug("Computed GW power")
    logger.debug("Computing EM power")
    power_em = sim.electromagneticwaves[radius].get_total_power()
    logger.debug("Computed EM power")

    logger.debug("Plotting")
    plt.plot((power_gw + power_em).time_shifted(-radius))
    plt.xlabel(r"Time - Detector distance $(t - r)$")
    plt.ylabel(r"$dE\slash dt (t)$")

    add_text_to_corner(f"Det {args.detector_num}", anchor="SW", offset=0.005)
    add_text_to_corner(fr"$r = {radius:.3f}$", anchor="NE", offset=0.005)

    set_axis_limits_from_args(args)
    logger.debug("Plotted")

    logger.debug("Saving")
    save_from_dir_filename_ext(args.outdir, figname, args.fig_extension)
    logger.debug("DONE")
