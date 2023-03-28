#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2021-2023 Gabriele Bozzola
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
    desc = f"""\
{kah.get_program_name()} plots the gravitational-wave luminosity and cumulative
energy as a function of time for a given detector. """

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "--detector-num",
        "--num-detector",
        type=int,
        required=True,
        help="Number of the spherical surface over which to read Psi4.",
    )

    parser.add_argument(
        "--pcut",
        type=int,
        required=True,
        help="Period that enters the fixed-frequency integration."
        " Typically, the longest physical period in the signal.",
    )

    args = kah.get_args(parser)
    setup_matplotlib(rc_par_file=args.mpl_rc_file)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(args, default=f"gw_energy_det{args.detector_num}")
    logger.debug(f"Using figname {figname}")

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        logger.debug("Prepared SimDir")

        radius = sim.gravitationalwaves.radii[args.detector_num]
        logger.debug(f"Using radius: {radius}")

        logger.debug("Computing energy")
        energy = sim.gravitationalwaves[radius].get_total_energy(args.pcut)
        logger.debug("Computed energy")

        logger.debug("Computing power")
        power = sim.gravitationalwaves[radius].get_total_power(args.pcut)
        logger.debug("Computed power ")

        logger.debug("Plotting")

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(power.time_shifted(-radius))
        # We set E = 0 at t - r = 0
        ax2.plot(energy.time_shifted(-radius) - energy(radius))

        ax2.set_xlabel(r"Time - Detector distance $(t - r)$")
        ax1.set_ylabel(r"$dE\slash dt (t)$")
        ax2.set_ylabel(r"$E^{<t}(t)$")

        add_text_to_corner(
            f"Det {args.detector_num}", anchor="SW", offset=0.005
        )
        add_text_to_corner(rf"$r = {radius:.3f}$", anchor="NE", offset=0.005)

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
