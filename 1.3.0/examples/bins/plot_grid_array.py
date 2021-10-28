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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    get_figname,
    plot_color,
    plot_horizon_on_plane_at_iteration,
    save_from_dir_filename_ext,
    setup_matplotlib,
)

if __name__ == "__main__":
    setup_matplotlib()

    desc = f"""\
{kah.get_program_name()} plot a given grid array.

Unfortunately, the files have information about the "coordinates" of grid
arrays, so, there are three options for labeling the axis: (1) do not label the
axis (default), (2) use a provided linear scale (if --x0 or --x1 are provided),
(3) plot on a sphere (with --on-sphere).

By default, no interpolation is performed so the image may look pixelated.
There are two available modes of interpolation. The first is activated
with --multilinear-interpolation. With this, the data from the simulation
is interpolated with a multilinear interpolation onto the plotting grid.
This is accurate and uses all the information available, but it is slow.
A second way to perform interpolation is passing a --interpolation-method
argument (e.g., bicubic). With this, the plotting data is interpolated.
This is much faster but it is not as accurate."""

    parser = kah.init_argparse(desc)

    kah.add_figure_to_parser(parser)
    kah.add_horizon_to_parser(parser)

    parser.add_argument(
        "--variable", type=str, required=True, help="Variable to plot."
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
        help="Whether to interpolate to smooth data with multilinear"
        " interpolation before plotting.",
    )
    parser.add_argument(
        "--interpolation-method",
        type=str,
        default="none",
        help="Interpolation method for the plot. See docs of np.imshow."
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--colorbar",
        action="store_true",
        help="Whether to draw the color bar.",
    )
    parser.add_argument(
        "--logscale",
        action="store_true",
        help="Whether to use log scale.",
    )
    parser.add_argument("-x0", "--origin", type=float, nargs=2)
    parser.add_argument("-x1", "--corner", type=float, nargs=2)
    parser.add(
        "--vmin",
        help=(
            "Minimum value of the variable. "
            "If logscale is True, this has to be the log."
        ),
        type=float,
    )
    parser.add(
        "--vmax",
        help=(
            "Maximum value of the variable. "
            "If logscale is True, this has to be the log."
        ),
        type=float,
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Whether to take the absolute value.",
    )
    parser.add_argument(
        "--on-sphere",
        action="store_true",
        help="Assume grid array is defined on a sphere.",
    )

    args = kah.get_args(parser)

    # Parse arguments

    iteration = args.iteration

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    figname = get_figname(args, default=f"{args.variable}")

    logger.debug(f"Reading variable {args.variable}")
    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:

        logger.debug("Prepared SimDir")

        var_name = args.variable

        if var_name in sim.gridfunctions.xy:
            var = sim.gridfunctions.xy[var_name]
            logger.debug("Data read from xy files")
        elif var_name in sim.gridfunctions.xz:
            var = sim.gridfunctions.xz[var_name]
            logger.debug("Data read from xz files")
        elif var_name in sim.gridfunctions.yz:
            var = sim.gridfunctions.yz[var_name]
            logger.debug("Data read from yz files")
        else:
            logger.debug(f"{sim.gridfunctions}")
            raise RuntimeError(
                f"Variable {var_name} cannot be found in 2D files"
            )

        logger.debug(f"Read variable {args.variable}")

        if iteration == -1:
            iteration = var.available_iterations[-1]

        time = var.time_at_iteration(iteration)

        logger.debug(f"Using iteration {iteration} (time = {time})")

        if args.absolute:
            data = abs(var[iteration])
            variable = f"abs({args.variable})"
        else:
            data = var[iteration]
            variable = args.variable

        data = data.merge_refinement_levels().data
        # Now data is a NumPy array

        if args.logscale:
            label = f"log10({variable})"
        else:
            label = variable

        logger.debug(f"Using label {label}")

        if args.on_sphere:

            # From https://stackoverflow.com/a/39535813

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            u, v = np.mgrid[
                0 : np.pi : 1j * data.shape[0],
                0 : 2 * np.pi : 1j * data.shape[1],
            ]

            norm = colors.Normalize(
                vmin=np.min(data), vmax=np.max(data), clip=False
            )

            x = np.sin(u) * np.cos(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(u)

            surf = ax.plot_surface(
                x,
                y,
                z,
                rstride=1,
                cstride=1,
                cmap=cm.inferno,
                linewidth=0,
                antialiased=False,
                facecolors=cm.inferno(norm(data)),
            )

            ax.set_axis_off()

            fig.colorbar(surf, label=label)

        elif args.origin and args.corner:

            x0, x1 = args.origin, args.corner

            # For grid arrays, the coordinate are the numbers from 0, 1, ...
            # (in 2D)

            print(data.grid.shape)
            len_x, len_y = data.grid.shape

            coordinates = (
                np.linspace(x0[0], x1[0], len_x),
                np.linspace(x0[1], x1[1], len_y),
            )

            # But we rescale the origin to x0 and the corner to x1

            logger.debug(f"Plotting on grid with x0 = {x0}, x1 = {x1}")

            plot_color(
                data,
                x0=x0,
                x1=x1,
                coordinates=coordinates,
                colorbar=args.colorbar,
                logscale=args.logscale,
                vmin=args.vmin,
                vmax=args.vmax,
                label=label,
                interpolation=args.interpolation_method,
            )

    add_text_to_corner(fr"$t = {time:.3f}$")

    logger.debug("Plotted")

    logger.debug("Saving")
    save_from_dir_filename_ext(
        args.outdir,
        figname,
        args.fig_extension,
        tikz_clean_figure=args.tikz_clean_figure,
    )
    logger.debug("DONE")
