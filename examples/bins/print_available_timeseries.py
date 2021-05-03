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

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir

if __name__ == "__main__":

    desc = f"""{kah.get_program_name()} prints the list of timeseries
    available to kuibit in the given data folder."""
    parser = kah.init_argparse(desc)
    args = kah.get_args(parser)
    print(
        SimDir(args.datadir, ignore_symlinks=args.ignore_symlinks).timeseries
    )
