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

import logging
import os

import matplotlib.pyplot as plt

from kuibit import argparse_helper as kah
from kuibit.grid_data import HierarchicalGridData
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    get_figname,
    plot_color,
    save_from_dir_filename_ext,
    setup_matplotlib,
)

"""This script plots lambda_MRI over the max(dx,dy,dz) on a given plane. Note,
the script requires a thorn that outputs the wavelength of the MRI using the
variable name lambda_MRI."""

if __name__ == "__main__":
    setup_matplotlib()

    desc = __doc__

    parser = kah.init_argparse(desc)
    kah.add_grid_to_parser(parser, dimensions=2)
    kah.add_figure_to_parser(parser)

    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to plot. If -1, the latest.",
    )
    parser.add_argument(
        "--multilinear-interpolate",
        action="store_true",
        help="Whether to interpolate to smooth data with multinear"
        " interpolation before plotting.",
    )
    parser.add_argument(
        "--interpolation-method",
        type=str,
        default="none",
        help="Interpolation method for the plot. See docs of np.imshow."
        " (default: %(default)s)",
    )
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
    args = kah.get_args(parser)

    # Parse arguments

    iteration = args.iteration
    x0, x1, res = args.origin, args.corner, args.resolution
    shape = [res, res]
    if args.figname is None:
        figname = f"MRI_quality_{args.plane}"
    else:
        figname = args.figname

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    logger.debug("Reading variable MRI_lambda")
    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:

        reader = sim.gridfunctions[args.plane]
        logger.debug(f"Variables available {reader}")
        var = reader["MRI_lambda"]
        logger.debug("Read variable MRI_lambda")

        if iteration == -1:
            iteration = var.available_iterations[-1]

        time = var.time_at_iteration(iteration)

        logger.debug(f"Using iteration {iteration} (time = {time})")

        logger.debug(
            f"Plotting on grid with x0 = {x0}, x1 = {x1}, shape = {shape}"
        )

        data = var[iteration]

        # Now we have to divide by the resolution (max(dx,dy,dz)) for each level. To
        # do this, we loop over all the components in data.
        for _, _, component in data:
            # Take the max of the resolution
            max_dx = max(component.dx)
            # Divide the actual data
            component.data /= max_dx

        plot_color(
            data,
            x0=x0,
            x1=x1,
            shape=shape,
            xlabel=args.plane[0],
            ylabel=args.plane[1],
            resample=args.multilinear_interpolate,
            colorbar=True,
            vmin=args.vmin,
            vmax=args.vmax,
            label=r"$\lambda_{\mathrm{MRI}}\slash\max(\Delta x,\Delta y,\Delta z)$",
            interpolation=args.interpolation_method,
        )

        add_text_to_corner(fr"$t = {time:.3f}$")

        output_path = os.path.join(args.outdir, figname)
        logger.debug(f"Saving in {output_path}")
        plt.tight_layout()
        save_from_dir_filename_ext(
            args.outdir,
            figname,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
