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
from math import sqrt

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir

if __name__ == "__main__":

    desc = f"""{kah.get_program_name()} prints some of the interesting properties (as from
QuasiLocalMeasures) for a given horizon at a given time. Cubic splines are used
to interpolate between timesteps."""

    parser = kah.init_argparse(desc)
    parser.add_argument(
        "--qlm-index",
        required=True,
        type=int,
        help="Index of the horizon according to QuasiLocalMeasures.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0,
        help="Time to consider.",
    )
    parser.add_argument(
        "--estimate-gamma",
        action="store_true",
        help=(
            "Estimate the Lorentz factor using qlm_w_momentum. "
            "Ignore this if you do not know the details"
        ),
    )
    args = kah.get_args(parser)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:

        sim_hor = sim.horizons

        logger.debug(
            f"QuasiLocalMeasures horizons available: {sim_hor.available_qlm_horizons}"
        )

        horizon = sim_hor.get_qlm_horizon(args.qlm_index)

        time = args.time

        irr_mass = horizon["irreducible_mass"](time)

        print(
            f"""\
QuasiLocalMeasures index: {args.qlm_index}
Time:                     {time:4.5f}
Irreducible mass:         {irr_mass:4.5f}
Christodoulou mass:       {horizon['mass'](time):4.5f}
Angular momentum:         {horizon['spin'](time):4.5f}"""
        )

        try:
            print(f"Charge:                   {horizon['charge'](time):4.5f}")
        except KeyError:
            pass

        if args.estimate_gamma:
            # Estimate gamma, the Lorentz factor
            #
            # We use the Weinberg momentum because it matches exactly the one in
            # TwoPunctures. Probably it does not make sense after t = 0.
            #
            momentum_sq = (
                horizon["w_momentum_x"](time) * horizon["w_momentum_x"](time)
                + horizon["w_momentum_y"](time) * horizon["w_momentum_y"](time)
                + horizon["w_momentum_z"](time) * horizon["w_momentum_z"](time)
            )

            gamma = sqrt(1 + momentum_sq / (irr_mass * irr_mass))

            print(f"Gamma:                    {gamma:4.5f}")
