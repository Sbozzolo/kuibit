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
    save_from_dir_filename_ext,
    setup_matplotlib,
)


if __name__ == "__main__":
    setup_matplotlib()

    desc = """{kah.get_program_name()} plots given reductions of the constraints."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser)
    parser.add_argument(
        "--reductions",
        default=["norm2", "maximum"],
        nargs="+",
        help="Reductions (norm2, maximum, ...) to plot.",
    )

    args = kah.get_args(parser)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    sim = SimDir(args.datadir, ignore_symlinks=args.ignore_symlinks)
    logger.debug("Prepared SimDir")

    reader = sim.timeseries
    logger.debug(f"Variables available {reader}")

    def plot_constraint(constraint_, reduction_):
        logger.debug(f"Reading {reduction_} of {constraint_}")
        var = reader[reduction_][constraint_]
        logger.debug(f"Read {reduction_} of {constraint_}")
        plt.semilogy(abs(var), label=f"{reduction_}(|{constraint_}|)")
        logger.debug(f"Plotted {reduction_} of {constraint_}")

    for reduction in args.reductions:
        logger.debug(f"Working with reduction: {reduction}")

        plt.clf()

        if args.figname is None:
            figname = f"constraints_{reduction}"
        else:
            figname = args.figname + f"_{reduction}"

        # We have multiple choices depending on the code used to compute the
        # constraint
        constraint_names = [
            ["H", "M1", "M2", "M3"],  # McLachlan
            ["hc", "mc", "my", "mz"],  # Lean
            ["hamc", "momcx", "momxy", "momcz", "divE"],  # ProcaConstraints
        ]

        logger.debug("Plotting")
        for names in constraint_names:
            for constraint in names:
                if constraint in reader[reduction]:
                    plot_constraint(constraint, reduction)

        plt.legend()
        logger.debug("Plotted")

        logger.debug("Saving")
        save_from_dir_filename_ext(args.outdir, figname, args.fig_extension)
        logger.debug("Saved")

    logger.debug("Done")
