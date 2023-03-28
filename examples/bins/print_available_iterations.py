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

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir

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
        choices=dimensions,
    )
    parser.add_argument(
        "--with-time",
        help="Print also the corresponding time (takes longer to process).",
        action="store_true",
    )
    args = kah.get_args(parser)

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        reader = sim.gridfunctions

        # We loop over dimensions
        if args.dimension is not None:
            if args.variable not in reader[args.dimension]:
                raise RuntimeError(
                    f"Variable {args.variable} of dimension {args.dimension} not available"
                )
            for it in reader[args.dimension][
                args.variable
            ].available_iterations:
                print(it, end=" ")
                if args.with_time:
                    time = reader[args.dimension][
                        args.variable
                    ].time_at_iteration(it)
                    print(f"({time:.2f})", end=" ")
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
                        if args.with_time:
                            time = reader[dim][
                                args.variable
                            ].time_at_iteration(it)
                            print(f"({time:.2f})", end=" ")
                    print()
