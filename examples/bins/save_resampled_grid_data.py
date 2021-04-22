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

from kuibit import argparse_helper as kah

from kuibit.simdir import SimDir

if __name__ == "__main__":

    desc = f"""{kah.get_program_name()} dumps a specific grid variable
    resampled to a given grid into a file. Saving as .npz files guarantees
    the best performances. """

    parser = kah.init_argparse(description=desc)
    parser.add_argument(
        "--variable", type=str, required=True, help="Variable to save."
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to plot. If -1, the latest.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=500,
        help=(("Resolution of the resampled data" "(default: %(default)s)")),
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["x", "y", "z", "xy", "xz", "yz", "xyz"],
        default="xyz",
        help="Type of data (default: %(default)s)",
    )

    parser.add_argument(
        "--outname",
        type=str,
        help="Name of the output file.",
    )

    parser.add_argument(
        "-x0",
        "--origin",
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "-x1",
        "--corner",
        type=float,
        nargs="+",
    )

    args = kah.get_args(parser)

    if len(args.type) != len(args.origin):
        raise ValueError(
            f"x0 ({args.origin}) and type ({args.type}) are incompatible"
        )

    if len(args.type) != len(args.corner):
        raise ValueError(
            f"x1 ({args.corner}) and type ({args.type}) are incompatible"
        )

    if args.outname is None:
        outname = f"{args.variable}_{args.type}.npz"
    else:
        outname = args.outname

    output_path = os.path.join(args.outdir, outname)

    iteration = args.iteration
    x0, x1, res = args.origin, args.corner, args.resolution
    shape = [res] * len(args.type)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    logger.debug(f"Reading variable {args.variable}")
    sim = SimDir(args.datadir, ignore_symlinks=args.ignore_symlinks)
    reader = sim.gridfunctions[args.type]
    logger.debug(f"Variables available {reader}")
    var = reader[args.variable]
    logger.debug(f"Read variable {args.variable}")

    if iteration == -1:
        iteration = var.available_iterations[-1]

    logger.debug(f"Reading {iteration} and resampling")

    data = var[iteration].to_UniformGridData(
        shape, x0, x1, iteration=iteration
    )

    logger.debug(f"Saving to {output_path}")
    data.save(output_path)
