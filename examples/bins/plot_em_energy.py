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
import os

import matplotlib.pyplot as plt

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    get_figname,
    save_from_dir_filename_ext,
    setup_matplotlib,
)

if __name__ == "__main__":
    setup_matplotlib()

    desc = f"""{kah.get_program_name()} plots, for a given detector, the electromagnetic
    luminosity and cumulative energy as a function of time as computed from
    Phi2. """

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser)

    parser.add_argument(
        "--detector-num",
        type=int,
        required=True,
        help="Number of the spherical surface over which to read Phi2.",
    )

    args = kah.get_args(parser)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(args, default=f"em_energy_det{args.detector_num}")
    logger.debug(f"Using figname {figname}")

    sim = SimDir(args.datadir, ignore_symlinks=args.ignore_symlinks)
    logger.debug("Prepared SimDir")

    radius = sim.electromagneticwaves.radii[args.detector_num]
    logger.debug(f"Using radius: {radius}")

    logger.debug("Computing energy")
    energy = sim.electromagneticwaves[radius].get_total_energy()
    logger.debug("Computed energy")

    logger.debug("Computing power")
    power = sim.electromagneticwaves[radius].get_total_power()
    logger.debug("Computed power")

    logger.debug("Plotting")

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(power.time_shifted(-radius))
    # We set E = 0 at t - r = 0
    ax2.plot(energy.time_shifted(-radius) - energy(radius))
    ax2.set_xlabel(r"Time - Detector distance $(t - r)$")
    ax1.set_ylabel(r"$dE\slash dt (t)$")
    ax2.set_ylabel(r"$E^{<t}(t)$")
    logger.debug("Plotted")

    logger.debug("Saving")
    save_from_dir_filename_ext(args.outdir, figname, args.fig_extension)
    logger.debug("DONE")
