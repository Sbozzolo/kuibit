#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2021-2025 Gabriele Bozzola
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

from py_expression_eval import Parser as math_parser

from kuibit import argparse_helper as kah
from kuibit import masks as km
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    get_figname,
    plot_color,
    plot_components_boundaries,
    plot_horizon_on_plane_at_iteration,
    save_from_dir_filename_ext,
    set_axis_limits,
    setup_matplotlib,
)

# NOTE: This example is also implemented in a movie file with the same name. If
#       you update this file, you probably want to update the movie file as
#       well.

if __name__ == "__main__":
    desc = f"""\
{kah.get_program_name()} plot a given expression involving grid functions.
Use the name of the grid functions, and algebraic expressions. For example,
sin(rho_b) * log10(P).

By default, no interpolation is performed so the image may look pixelated.
There are two available modes of interpolation. The first is activated
with --multilinear-interpolation. With this, the data from the simulation
is interpolated with a multilinear interpolation onto the plotting grid.
This is accurate and uses all the information available, but it is slow.
A second way to perform interpolation is passing a --interpolation-method
argument (e.g., bicubic). With this, the plotting data is interpolated.
This is much faster but it is not as accurate."""

    parser = kah.init_argparse(desc)
    kah.add_grid_to_parser(parser, dimensions=2)
    kah.add_figure_to_parser(parser)
    kah.add_horizon_to_parser(parser)
    kah.add_grid_structure_to_parser(parser)

    parser.add_argument("expr", nargs=1, type=str, help="Expression to plot.")

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
    setup_matplotlib(rc_par_file=args.mpl_rc_file)

    # Parse arguments

    iteration = args.iteration
    x0, x1, res = args.origin, args.corner, args.resolution
    shape = [res, res]

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    expr = args.expr[0]
    figname = get_figname(
        args, default=f"{expr}_{args.plane}".replace("/", "_")
    )

    logger.debug(f"Expression {expr}")
    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        logger.debug("Prepared SimDir")
        reader = sim.gridfunctions[args.plane]
        logger.debug(f"Variables available {reader}")

        exp_parser = math_parser()

        # We need to make sure that the correct methods are being used. That is,
        # not the ones in python.math.
        exp_parser.ops1.update(
            {
                "exp": km.exp,
                "sin": km.sin,
                "cos": km.cos,
                "tan": km.tan,
                "log": km.log,
                "log10": km.log10,
                "asin": km.arcsin,
                "acos": km.arccos,
                "atan": km.arctan,
                "sqrt": km.sqrt,
                "abs": abs,
            }
        )

        variable_names = exp_parser.parse(expr).variables()

        # vars_ is a dict that contains NumPy arrays for all the various
        # variables involved. We use NumPy arrays because py_expression_eval
        # doesn't support complex types. Hence, we need to resample more often
        # than we could have done.
        variables = {}
        for var_name in variable_names:
            var = reader[var_name]

            if iteration == -1:
                iteration = var.available_iterations[-1]

            variables.update({var_name: var[iteration]})
            logger.debug(f"Read variable {var_name}")

        time = var.time_at_iteration(iteration)
        logger.debug(f"Using iteration {iteration} (time = {time})")

        logger.debug(
            f"Plotting on grid with x0 = {x0}, x1 = {x1}, shape = {shape}"
        )

        data = exp_parser.parse(expr).evaluate(variables)

        logger.debug("Resampling and plotting")
        plot_color(
            data,
            x0=x0,
            x1=x1,
            shape=shape,
            resample=args.multilinear_interpolate,
            xlabel=args.plane[0],
            ylabel=args.plane[1],
            colorbar=args.colorbar,
            vmin=args.vmin,
            vmax=args.vmax,
            label=expr,
            interpolation=args.interpolation_method,
        )

        add_text_to_corner(rf"$t = {time:.3f}$")

        if args.ah_show:
            for ah in sim.horizons.available_apparent_horizons:
                logger.debug(f"Plotting apparent horizon {ah}")
                plot_horizon_on_plane_at_iteration(
                    sim.horizons.get_apparent_horizon(ah),
                    iteration,
                    args.plane,
                    color=args.ah_color,
                    edgecolor=args.ah_edge_color,
                    alpha=args.ah_alpha,
                )

        if args.rl_show:
            logger.debug("Plotting grid structure")
            plot_components_boundaries(
                data, edgecolor=args.rl_edge_color, alpha=args.rl_alpha
            )

        set_axis_limits(xmin=x0[0], xmax=x1[0], ymin=x0[1], ymax=x1[1])

        logger.debug("Plotted")

        logger.debug("Saving")
        save_from_dir_filename_ext(
            args.outdir,
            figname,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
        logger.debug("DONE")
