#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2022-2025 Gabriele Bozzola
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
    desc = f"""\
{kah.get_program_name()} describes the content of a simdir"""

    parser = kah.init_argparse(desc)

    # For this script, we need a pickle file, so we are going to provide a
    # default value
    parser.set_defaults(pickle_file="sim.pickle")

    args = kah.get_args(parser)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    logger.debug(f"Using pickle file {args.pickle_file}")

    if os.path.exists(args.pickle_file):
        logger.debug("Found existing pickle file, it will be overwritten")

    with SimDir(args.datadir, ignore_symlinks=args.ignore_symlinks) as sim:
        logger.debug("Prepared SimDir")
        print(sim)
        logger.debug("Filled SimDir")

    logger.debug("DONE")
