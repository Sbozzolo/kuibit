#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

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

from kuibit import argparse_helper as pah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_figure_corner,
    plot_color,
    plot_colorbar,
    save,
    save_from_dir_filename_ext,
    setup_matplotlib,
)

"""This script plots a color plot of the rest-mass density rho_b on the xz
plane. On top of it, it overplots MRI_lambda as measured on the equatorial
plane. Note, the script requires a thorn that outputs the wavelength of the MRI
using the variable name lambda_MRI. For the rest-mass density, IllinoisGRMHD is
assumed."""

if __name__ == "__main__":
    setup_matplotlib()

    desc = __doc__

    parser = pah.init_argparse(desc)
    pah.add_grid_to_parser(parser, dimensions=2)
    pah.add_figure_to_parser(parser)

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

    args = pah.get_args(parser)

    # Parse arguments

    iteration = args.iteration
    x0, x1, res = args.origin, args.corner, args.resolution
    shape = [res, res]
    if args.figname is None:
        figname = f"MRI_lambda_rho_b_xz"
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

        reader = sim.gridfunctions["xz"]
        logger.debug(f"Variables available {reader}")
        var = reader["rho_b"]
        logger.debug("Read variable rho_b")

        if iteration == -1:
            iteration = var.available_iterations[-1]

        time = var.time_at_iteration(iteration)

        logger.debug(f"Using iteration {iteration} (time = {time})")

        logger.debug(
            f"Plotting on grid with x0 = {x0}, x1 = {x1}, shape = {shape}"
        )

        data = var[iteration]

        fig, ax = plt.subplots()

        image = plot_color(
            data,
            axis=ax,
            x0=x0,
            x1=x1,
            shape=shape,
            xlabel="x",
            ylabel="z",
            resample=args.multilinear_interpolate,
            # colorbar=True,
            label=r"$rho_0$",
            interpolation=args.interpolation_method,
            aspect_ratio=None,
        )

        # We are going to read MRI_lambda from the xy output

        reader_xy = sim.gridfunctions["xy"]
        logger.debug(f"Variables available {reader_xy}")
        MRI_lambda = reader["MRI_lambda"][iteration]
        logger.debug("Read variable MRI_lambda")

        # Now, we slice the x axis
        MRI_lambda = MRI_lambda.sliced([None, 0])
        xmin, xmax = x0[0], x1[0]
        # Now resample and divide by 2
        MRI_lambda_1d = 0.5 * MRI_lambda.to_UniformGridData(
            shape=[res],
            x0=[xmin],
            x1=[xmax],
            resample=args.multilinear_interpolate,
        )

        ax.plot(
            MRI_lambda_1d.coordinates_from_grid()[0],
            MRI_lambda_1d.data,
            color="white",
        )
        ax.set_aspect(3)

        # plot_colorbar(image)

        add_text_to_figure_corner(fr"$t = {time:.3f}$")

        output_path = os.path.join(args.outdir, figname)
        logger.debug(f"Saving in {output_path}")
        plt.tight_layout()
        save(
            output_path,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
            as_tikz=args.as_tikz,
        )
