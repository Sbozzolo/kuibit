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

from kuibit.simdir import SimDir
from kuibit import argparse_helper as kah

if __name__ == "__main__":

    desc = f"""{kah.get_program_name()} prints the list of the available
    iterations given a grid function. This can be used for shell scripts."""
    parser = kah.init_argparse(desc)

    dimensions = ("x", "y", "z", "xy", "xz", "yz", "xyz")

    parser.add_argument(
        "--variable",
        required=True,
        help="Show iterations of this variable.",
    )
    parser.add_argument(
        "--dimension",
        help="Print only for the given dimension.",
    )
    args = kah.get_args(parser)

    reader = SimDir(
        args.datadir, ignore_symlinks=args.ignore_symlinks
    ).gridfunctions

    # We loop over dimensions
    if args.dimension is not None:
        if args.variable not in reader[args.dimension]:
            raise RuntimeError(
                f"Variable {args.variable} of dimension {args.dimension} not available"
            )
        for it in reader[args.dimension][args.variable].available_iterations:
            print(it, end=" ")
        print()
    else:
        # First we check that we have the variable
        if not any(args.variable in reader[dim] for dim in dimensions):
            raise RuntimeError(f"Variable {args.variable} not available")
        # Okay, we have something
        for dim in dimensions:
            if args.variable in reader[dim]:
                print(f"# {dim}")
                for it in reader[dim][args.variable].available_iterations:
                    print(it, end=" ")
                print()
