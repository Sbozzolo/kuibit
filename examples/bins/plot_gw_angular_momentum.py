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

    desc = f"""\
{kah.get_program_name()} plots the instantaneous and cumulative angular momentum
lost via emission of gravitational-wave as a function of time for a given
detector. """

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "--detector-num",
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

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(
        args, default=f"gw_angular momentum_z_det{args.detector_num}"
    )
    logger.debug(f"Using figname {figname}")

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:

        logger.debug("Prepared SimDir")

        radius = sim.gravitationalwaves.radii[args.detector_num]
        logger.debug(f"Using radius: {radius}")

        logger.debug("Computing angular momentum")
        linmom = sim.gravitationalwaves[radius].get_total_angular_momentum_z(
            args.pcut
        )
        logger.debug("Computed angular momentum")

        logger.debug("Computing torque")
        force_z = sim.gravitationalwaves[radius].get_total_torque_z(args.pcut)
        logger.debug("Computed torque")

        logger.debug("Plotting")

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        ax1.plot(force_z.time_shifted(-radius))
        # We set J = 0 at t - r = 0
        ax2.plot(linmom.time_shifted(-radius) - linmom(radius))
        ax2.set_xlabel(r"Time - Detector distance $(t - r)$")
        ax1.set_ylabel(r"$dJ^z\slash dt (t)$")
        ax2.set_ylabel(r"$J^z_{<t}(t)$")

        add_text_to_corner(
            f"Det {args.detector_num}", anchor="SW", offset=0.005
        )
        add_text_to_corner(fr"$r = {radius:.3f}$", anchor="NE", offset=0.005)

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
