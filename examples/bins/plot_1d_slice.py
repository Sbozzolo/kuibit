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

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    save_from_dir_filename_ext,
    set_axis_limits_from_args,
    setup_matplotlib,
)

if __name__ == "__main__":
    desc = """\
{kah.get_program_name()} plots a grid function on one of the coordinate axis. 1D
data is used if available, otherwise higher dimensional data is used."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser, add_limits=True)

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
        "--resolution",
        type=int,
        default=1000,
        help="Resolution to use for the plot.",
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
        "--axis",
        type=str,
        choices=["x", "y", "z"],
        default="x",
        help="Axis to plot (default: %(default)s)",
    )
    args = kah.get_args(parser)
    setup_matplotlib(rc_par_file=args.mpl_rc_file)

    iteration, var_name, axis = args.iteration, args.variable, args.axis

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    if args.figname is None:
        figname = f"{var_name}_{axis}_sliced"
    else:
        figname = args.figname

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        logger.debug("Prepared SimDir")

        # Okay, we need to understand where to read the data from

        # Do we have 1D data?
        if var_name not in sim.gridfunctions[axis]:
            # We do not have 1D data, so let's check if we have 2D data

            logger.debug(f"{var_name} not available in {axis}")

            # Depending on what axis the user requested, we need to look at
            # different data. The variable containing_axis maps the axis
            # to 2D data that might contain it.
            containing_axis = {
                "x": ("xy", "xz"),
                "y": ("xy", "yz"),
                "z": ("xz", "yz"),
            }

            # Now we loop over the two possible containing axis finding where
            # the variable is available
            var_in_2D = [
                var_name in sim.gridfunctions[ax]
                for ax in containing_axis[axis]
            ]

            # Do we any the data among the 2D data?
            if not any(var_in_2D):
                logger.debug(f"{var_name} not available in 2D data")

                # We don't have the data in the 2D files, so we should check 3D
                # data
                if var_name not in sim.gridfunctions.xyz:
                    # We don't have the data in the 3D files either, hence we
                    # don't have the data at all.
                    raise ValueError(
                        f"{var_name} is not available in 1D, 2D, and 3D data"
                    )

                # We have the data in the 3D files
                logger.debug("Using 3D data")

                # Let's read it
                var_3D = sim.gridfunctions.xyz[var_name]

                if iteration == -1:
                    iteration = var_3D.available_iterations[-1]

                time = var_3D.time_at_iteration(iteration)

                logger.debug(f"Using iteration {iteration} (time = {time})")

                # Now we have to slice it. The variable cuts tells me how to.
                # We keep one dimension, and set 0, 0 to the others.
                cuts = {
                    "x": [None, 0, 0],
                    "y": [0, None, 0],
                    "z": [0, 0, None],
                }
                var = var_3D[iteration].sliced(cuts[axis])
            else:  # var_name available in 2D data
                logger.debug("Using 2D data")

                # var_in_2D contains two elements. These can be both True, or
                # only one of the two (in that case, the other must be False).
                # We need to pick one of the Trues to know what 2D data to read.

                # var_in_2D.index(True) returns the index of the first True in
                # the iterable.
                which_axes = containing_axis[axis][var_in_2D.index(True)]

                var_2D = sim.gridfunctions[which_axes][var_name]

                if iteration == -1:
                    iteration = var_2D.available_iterations[-1]

                time = var_2D.time_at_iteration(iteration)

                logger.debug(f"Using iteration {iteration} (time = {time})")

                # Now we have to slice the 2D data. We could be clever, but we
                # will be clear. cuts is a dictionary that spells out all the
                # possible options, depending on the containing axes and the
                # requested one.

                cuts = {
                    "xy": {"x": [None, 0], "y": [0, None]},
                    "xz": {"x": [None, 0], "z": [0, None]},
                    "yz": {"y": [None, 0], "z": [0, None]},
                }

                # Now we have to slice it
                var = var_2D[iteration].sliced(cuts[which_axes][axis])

        else:  # var_name available in 1D data
            logger.debug("Using 1D data")

            # Here we just have to read the data

            var = sim.gridfunctions[axis][var_name]

            if iteration == -1:
                iteration = var.available_iterations[-1]

            time = var.time_at_iteration(iteration)

            logger.debug(f"Using iteration {iteration} (time = {time})")

            var = var[iteration]

        if args.absolute:
            data = abs(var)
            variable_name = f"abs({var_name})"
        else:
            data = var
            variable_name = var_name

        logger.debug("Resampling to UniformGridData")

        # We cast to a GridSeries, which is easier to plot.

        data = data.to_UniformGridData(
            [args.resolution],
            x0=[args.xmin],
            x1=[args.xmax],
            resample=args.multilinear_interpolate,
        ).to_GridSeries()

        if args.logscale:
            label = f"log10({variable_name})"
            data = data.log10()
        else:
            label = variable_name

        logger.debug(f"Using label {label}")

        logger.debug(f"Plotting variable {var_name}")

        plt.plot(data, label=label)

        add_text_to_corner(rf"$t = {time:.3f}$")

        plt.xlabel(axis)
        plt.ylabel(label)
        set_axis_limits_from_args(args)

        logger.debug("Saving")
        save_from_dir_filename_ext(
            args.outdir,
            figname,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
        logger.debug("DONE")
