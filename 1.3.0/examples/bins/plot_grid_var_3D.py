#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

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

from mayavi import mlab

from kuibit import argparse_helper as pah
from kuibit.simdir import SimDir
from kuibit.visualize_mayavi import (
    add_text_to_figure_corner,
    disable_interactive_window,
    plot_apparent_horizon,
    save,
)

"""This script plots a grid function with options specified via command-line. """

if __name__ == "__main__":
    disable_interactive_window()

    desc = __doc__

    parser = pah.init_argparse(desc)
    pah.add_grid_to_parser(parser, dimensions=3)
    pah.add_figure_to_parser(parser)
    pah.add_horizon_to_parser(parser)

    parser.add_argument(
        "--variable", type=str, required=True, help="Variable to plot"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to plot. If -1, the latest.",
    )
    parser.add_argument(
        "--multilinear-interpolate",
        action="store_true",
        help="Whether to interpolate to smooth data with multlinear"
        " interpolation before plotting.",
    )
    # parser.add_argument(
    #     "--colorbar",
    #     action="store_true",
    #     help="Whether to draw the color bar.",
    # )
    parser.add_argument(
        "--logscale",
        action="store_true",
        help="Whether to use log scale.",
    )
    # parser.add(
    #     "--vmin",
    #     help=(
    #         "Minimum value of the variable. "
    #         "If logscale is True, this has to be the log."
    #     ),
    #     type=float,
    # )
    # parser.add(
    #     "--vmax",
    #     help=(
    #         "Maximum value of the variable. "
    #         "If logscale is True, this has to be the log."
    #     ),
    #     type=float,
    # )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Whether to take the absolute value.",
    )

    args = pah.get_args(parser)

    # Parse arguments

    iteration = args.iteration
    x0, x1, res = args.origin, args.corner, args.resolution
    shape = [res, res, res]
    if args.figname is None:
        figname = f"{args.variable}_3D"
    else:
        figname = args.figname

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

        reader = sim.gridfunctions.xyz
        logger.debug(f"Variables available {reader}")
        var = reader[args.variable]
        logger.debug(f"Read variable {args.variable}")

        if iteration == -1:
            iteration = var.available_iterations[-1]

        time = var.time_at_iteration(iteration)

        logger.debug(f"Using iteration {iteration} (time = {time})")

        logger.debug(
            f"Plotting on grid with x0 = {x0}, x1 = {x1}, shape = {shape}"
        )

        if args.absolute:
            data = abs(var[iteration])
            variable = f"abs({args.variable})"
        else:
            data = var[iteration]
            variable = args.variable

        if args.logscale:
            label = f"log10({variable})"
        else:
            label = variable

        logger.debug(f"Using label {label}")

        add_text_to_figure_corner(fr"$t = {time:.3f}$")

        data = data.to_UniformGridData(shape, x0, x1)

        mlab.contour3d(
            *data.coordinates_from_grid(as_same_shape=True),
            data.data,
            transparent=True,
        )

        if args.ah_show:
            for ah in sim.horizons.available_apparent_horizons:
                # We don't care about the QLM index here
                hor = sim.horizons[0, ah]
                logger.debug(f"Plotting apparent horizon {ah}")
                plot_apparent_horizon(hor, iteration, color=args.ah_color)

        output_path = os.path.join(args.outdir, figname)
        logger.debug(f"Saving in {output_path}")
        save(
            output_path,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
