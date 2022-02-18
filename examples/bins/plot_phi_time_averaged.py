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

import matplotlib.pyplot as plt
import numpy as np

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    save_from_dir_filename_ext,
    set_axis_limits,
    setup_matplotlib,
)

if __name__ == "__main__":
    setup_matplotlib()

    desc = """\

{kah.get_program_name()} plots the azimuthal and time average of a given grid
function around a given center. The script interpolates the data onto concentric
rings and takes the average at a fixed radius. Then, this is averaged over a
window of time defined by tmin and tmax."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

    parser.add_argument(
        "--variable", type=str, required=True, help="Variable to plot."
    )
    parser.add_argument(
        "--plane",
        type=str,
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Plane to consider (default: %(default)s).",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        required=True,
        help="Minimum time in the time window.",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        required=True,
        help="Minimum time in the time window.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=300,
        help="Resolution to use for the x axis.",
    )
    parser.add_argument(
        "--phi-resolution",
        type=int,
        default=100,
        help="Resolution to use for the phi averaging.",
    )
    parser.add_argument(
        "--time-every",
        type=int,
        default=1,
        help="Use one snapshot every this number of available ones (default: %(default)s).",
    )
    parser.add(
        "--logscale", help="Use a logarithmic y scale.", action="store_true"
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Whether to take the absolute value.",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        default=(0, 0),
        help="Center around which to perform the average (default: %(default)s)",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=1,
        help="Maximum radius at which perform the average (default: %(default)s)",
    )
    args = kah.get_args(parser)

    tmin, tmax, var_name, = (
        args.tmin,
        args.tmax,
        args.variable,
    )
    X0, Y0 = args.center

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    if args.figname is None:
        figname = f"{var_name}_{args.plane}_phi_time_averaged"
    else:
        figname = args.figname

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        logger.debug("Prepared SimDir")

        reader = sim.gridfunctions[args.plane]
        logger.debug(f"Variables available {reader}")

        var = reader[var_name]
        logger.debug(f"Read variable {args.variable}")

        times = var.available_times
        logger.debug(f"Available time {times}")

        selected_times = [time for time in times if tmin <= time <= tmax]
        selected_times = selected_times[:: args.time_every]
        logger.debug(f"Selected times {selected_times}")

        # The equations are
        # X = X0 + R * cos(phi)
        # Y = Y0 + R * sin(phi)

        # We prepare the polar grid that we want evaluate
        angles = np.linspace(0, 2 * np.pi, args.phi_resolution)
        # R_min is not set to zero because it would be pointless
        radii = np.linspace(1e-5, args.max_radius, args.resolution)

        # These are the points where we are going to evaluate the variable. They
        # are a series of concentric rings.
        points = np.asarray(
            [
                [
                    (X0 + R * np.cos(phi), Y0 + R * np.sin(phi))
                    for phi in angles
                ]
                for R in radii
            ]
        )

        # We are going to collect all the results in values, and the end we do
        # the time averaging. So first, we do the phi averaging, then the time
        # one. The variable ret_values contains the phi-averaged quantity.
        ret_values = np.zeros(args.resolution)

        for time in selected_times:
            logger.debug(f"Working on time {time}")

            var_at_time = var.get_time(time)

            values_at_time = var_at_time(points)

            if args.logscale:
                val_time = np.log10(values_at_time)

            # Phi-averaging
            phi_averaged_at_time = np.mean(values_at_time, axis=1)

            ret_values += phi_averaged_at_time

            plt.plot(radii, phi_averaged_at_time, linewidth=0.2, c="gray")

        # Time-averaging
        ret_values /= len(selected_times)

        if args.logscale:
            label = f"log10({var_name})"
        else:
            label = f"{var_name}"

        logger.debug(f"Using label {label}")

        logger.debug(f"Plotting variable {var_name}")
        plt.plot(radii, ret_values)

        add_text_to_corner(
            rf"$t \in ({selected_times[0]:.3f}, {selected_times[-1]:.3f})$"
        )
        add_text_to_corner(rf"Center = {X0:.3f}, {Y0:.3f}", anchor="NW")

        plt.xlabel("Radius")
        plt.ylabel(label)
        set_axis_limits(xmin=radii[0], xmax=radii[-1])

        logger.debug("Saving")
        save_from_dir_filename_ext(
            args.outdir,
            figname,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
        logger.debug("DONE")
