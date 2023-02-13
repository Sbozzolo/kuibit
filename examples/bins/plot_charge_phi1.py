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
from numpy import pi

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
{kah.get_program_name()} plots the total charge as computed from Phi1."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "--detector-nums",
        "--num-detectors",
        type=int,
        required=True,
        nargs="+",
        help="Number of the spherical surface over which to read Phi."
        " Multiple are possible.",
    )

    args = kah.get_args(parser)
    setup_matplotlib(rc_par_file=args.mpl_rc_file)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(
        args, default=f"charge_phi1_{'_'.join(map(str, args.detector_nums))}"
    )
    logger.debug(f"Using figname {figname}")

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        logger.debug("Prepared SimDir")

        reader_mult = sim.multipoles

        if "Phi1" not in reader_mult:
            raise ValueError("Phi1 not available")

        reader = reader_mult["Phi1"]

        for num in args.detector_nums:
            try:
                radius = reader.radii[num]
            except IndexError:
                raise ValueError(f"Detector {num} not available") from None

            logger.debug(f"Plotting detector {num} (radius: {radius})")
            detector = reader[radius]

            if (0, 0) not in detector.available_lm:
                logger.debug(f"Available multipoles {detector.available_lm}")
                raise ValueError("Phi1 l=0, m=0 not available")

            phi = detector[0, 0].real()

            charge = phi.real() * radius**2 * pi**-0.5

            plt.plot(charge, label=f"{radius:.2f}")
            logger.debug("Plotted")

        set_axis_limits_from_args(args)
        plt.legend()
        plt.xlabel("Simulation Time")
        plt.ylabel("Charge")

        logger.debug("Saving")
        save_from_dir_filename_ext(
            args.outdir,
            figname,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
        logger.debug("DONE")
