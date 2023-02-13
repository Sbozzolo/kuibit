#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2021-2022 Gabriele Bozzola
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

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir

if __name__ == "__main__":
    desc = f"""\
{kah.get_program_name()} prints the point where a given grid function at a given
iteration is maximum or minimum."""

    parser = kah.init_argparse(desc)

    dimensions = ("x", "y", "z", "xy", "xz", "yz", "xyz")

    parser.add_argument(
        "--variable",
        required=True,
        help="Consider this variable.",
    )
    parser.add_argument(
        "--dimension",
        help="Print only for the given dimension.",
        choices=dimensions,
        required=True,
    )
    parser.add_argument(
        "--type", required=True, choices=["maximum", "max", "minimum", "min"]
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Whether to take the absolute value.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to consider. If -1, the latest.",
    )

    args = kah.get_args(parser)

    iteration = args.iteration
    absolute = args.absolute

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    logger.debug(f"Reading variable {args.variable}")
    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        logger.debug("Prepared SimDir")
        reader = sim.gridfunctions
        logger.debug(f"Variables available {reader}")

        if args.variable not in reader[args.dimension]:
            raise RuntimeError(
                f"Variable {args.variable} of dimension {args.dimension} not available"
            )

        var = reader[args.dimension][args.variable]
        logger.debug(f"Read variable {args.variable}")

        if iteration == -1:
            iteration = var.available_iterations[-1]

        data = var[iteration]

        if args.type in ("maximum", "max"):
            logger.debug("Working with maximum")
            maximum = data.abs_max() if absolute else data.max()
            logger.debug(
                f"Absolute maximum {maximum:.3f}"
                if absolute
                else f"Maximum {maximum:.3f}"
            )
            print(data.coordinates_at_maximum(absolute))
        else:
            logger.debug("Working with minimum")
            minimum = data.abs_min() if absolute else data.min()
            logger.debug(
                f"Absolute minimum {minimum:.3f}"
                if absolute
                else f"Minimum {minimum:.3f}"
            )
            print(data.coordinates_at_minimum(absolute))

        logger.debug("DONE")
