#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2020-2023 Gabriele Bozzola
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
{kah.get_program_name()} plots the (l,m) gravitational-wave strain. Optionally,
a window function can be applied to the data before performing the integration.
To do this, use the --window flag passing the name of a method defined in
TimeSeries (e.g. 'tukey'). Then, pass all the arguments with --window-args in
the order as they appear in the TimeSeries method."""

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
        type=float,
        required=True,
        help="Period for the fixed frequency integration.",
    )

    parser.add_argument(
        "--mult-l", type=int, default=2, help="Multipole number l."
    )
    parser.add_argument(
        "--mult-m", type=int, default=2, help="Multipole number m."
    )

    parser.add_argument(
        "--window",
        help="Window function to apply before performing the integration.",
    )

    parser.add_argument(
        "--window-args",
        nargs="*",
        type=float,
        help="Arguments of the window function.",
    )

    args = kah.get_args(parser)
    setup_matplotlib(rc_par_file=args.mpl_rc_file)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(
        args,
        default=f"strain_{args.mult_l}{args.mult_m}_det{args.detector_num}",
    )
    logger.debug(f"Using figname {figname}")

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
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

        logger.debug("Computing strain")

        # We are going to do tuple unpacking, and we cannot unpack "None",
        # so we overwrite the value with an empty list
        if args.window_args is None:
            args.window_args = []

        strain = detector.get_strain_lm(
            args.mult_l,
            args.mult_m,
            args.pcut,
            *args.window_args,
            window_function=args.window,
        )

        logger.debug("Plotting")

        plt.plot(
            strain.real(),
            label=rf"$r_{{\mathrm{{ex}}}} h^{{{args.mult_l}{args.mult_m}}}_+$",
        )
        plt.plot(
            -strain.imag(),
            label=rf"$r_{{\mathrm{{ex}}}} h^{{{args.mult_l}{args.mult_m}}}_\times$",
        )

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(rf"$r_{{\mathrm{{ex}}}} h^{{{args.mult_l}{args.mult_l}}}$")

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
