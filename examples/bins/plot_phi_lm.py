#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

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
{kah.get_program_name()} plots the multipolar decomposition of one of the Phi
Newman-Penrose constants for Maxwell/Proca as measured by a given detector and
at a given l and m."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "--phi-num",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Why Phi to plot? Phi0, Phi1, or Phi2?",
    )

    parser.add_argument(
        "--detector-num",
        type=int,
        required=True,
        help="Number of the spherical surface over which to read Phi.",
    )

    parser.add_argument(
        "--mult-l", type=int, default=0, help="Multipole number l."
    )
    parser.add_argument(
        "--mult-m", type=int, default=0, help="Multipole number m."
    )

    args = kah.get_args(parser)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    phi_num = args.phi_num
    var_name = f"Phi{phi_num}"

    figname = get_figname(
        args,
        default=f"{var_name}_{args.mult_l}{args.mult_m}_det{args.detector_num}",
    )
    logger.debug(f"Using figname {figname}")

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:

        logger.debug("Prepared SimDir")

        reader_mult = sim.multipoles

        if var_name not in reader_mult:
            raise ValueError(f"{var_name} not available")

        reader = reader_mult[var_name]

        radius = reader.radii[args.detector_num]
        logger.debug(f"Using radius: {radius}")
        detector = reader[radius]

        if (args.mult_l, args.mult_m) not in detector.available_lm:
            logger.debug(f"Available multipoles {detector.available_lm}")
            raise ValueError(
                f"Multipole {args.mult_l}, {args.mult_m} not available"
            )

        phi = detector[args.mult_l, args.mult_m]

        logger.debug(f"Plotting {var_name}")

        plt.plot(
            phi.real(),
            label=fr"$\Re \Phi_{phi_num}^{{{args.mult_l}{args.mult_l}}}$",
        )
        plt.plot(
            phi.imag(),
            label=fr"$\Im \Phi_{phi_num}^{{{args.mult_l}{args.mult_l}}}$",
        )

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(fr"$r \Phi_{phi_num}$")
        set_axis_limits_from_args(args)

        add_text_to_corner(
            f"Det {args.detector_num}", anchor="SW", offset=0.005
        )
        add_text_to_corner(fr"$r = {radius:.3f}$", anchor="NE", offset=0.005)

        logger.debug("Plotted")

        logger.debug("Saving")
        save_from_dir_filename_ext(args.outdir, figname, args.fig_extension)
        logger.debug("DONE")
