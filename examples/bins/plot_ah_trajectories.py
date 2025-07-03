#!/usr/bin/env python3

# Copyright (C) 2020-2025 Gabriele Bozzola
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
from kuibit.series import sample_common
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    get_figname,
    plot_horizon_on_plane_at_time,
    save_from_dir_filename_ext,
    setup_matplotlib,
)

if __name__ == "__main__":
    desc = f"""{kah.get_program_name()} plots the trajectories of given apparent horizons on
a plane. Optionally, it also plots the outline of the horizons."""

    parser = kah.init_argparse(desc)
    kah.add_figure_to_parser(parser)

    parser.add_argument(
        "--plane",
        type=str,
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Plane to plot (default: %(default)s)",
    )

    parser.add_argument(
        "-a",
        "--horizons",
        type=int,
        required=True,
        help="Apparent horizons to consider.",
        nargs="+",
    )

    parser.add_argument(
        "--draw-ah",
        help="Draw the outlines of the apparent horizons.",
        action="store_true",
    )
    parser.add_argument(
        "--force-com-at-origin",
        help="Force the Newtonian center of mass to be in the origin "
        "(this uses the irreducible masses). In this case, everything "
        "will be resampled to times that are common across the horizons.",
        action="store_true",
    )
    args = kah.get_args(parser)
    setup_matplotlib(rc_par_file=args.mpl_rc_file)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    # For the figure name, we chain all the horizon numbers
    horizon_name = "_".join([str(h) for h in args.horizons])
    figname = get_figname(
        args, default=f"ah_{horizon_name}_trajectory_{args.plane}"
    )
    logger.debug(f"Figname: {figname}")

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:
        logger.debug("Prepared SimDir")
        sim_hor = sim.horizons

        logger.debug(
            f"Apparent horizons available: {sim_hor.available_apparent_horizons}"
        )

        # Check that the horizons are available
        for ah in args.horizons:
            if ah not in sim_hor.available_apparent_horizons:
                raise ValueError(f"Apparent horizons {ah} is not available")

        # Now, we prepare the list with the centroids. The keys are the
        # horizon numbers and the values are TimeSeries. We prepare this so that
        # we can compute the center of mass. For that, we also need the masses
        # (x_cm = sum m_i / M x_i). We use the irreducible mass for this.
        ah_coords = {"x": [], "y": [], "z": []}
        masses = []
        # If force_com_at_origin, these objects will be rewritten so that we can
        # assume that they house the quantities we want to plot.

        for ah in args.horizons:
            logger.debug(f"Reading horizon {ah}")

            hor = sim_hor.get_apparent_horizon(ah)

            for coord, ah_coord in ah_coords.items():
                ah_coord.append(hor.ah[f"centroid_{coord}"])

            if args.force_com_at_origin:
                logger.debug("Reading mass")
                masses.append(hor.ah.m_irreducible)

        if args.force_com_at_origin:
            # x_cm = sum m_i / M x_i

            logger.debug("Computing center of mass")

            logger.debug("Resampling to common times")

            # We have to resample everything to a common time interval. This is
            # because we are going to combine the various objects with
            # mathematical operations.

            # Loop over the three coordinates and overwrite the list of
            # TimeSeries (note that ahs here are lists of TimeSeries and
            # sample_common(ahs) returns a new list of TimeSeries)
            ah_coords = {
                coord: sample_common(ahs) for coord, ahs in ah_coords.items()
            }

            masses = sample_common(masses)

            # First, we compute the total mass (as TimeSeries)
            total_mass = sum(mass for mass in masses)

            # Loop over the three coordinates
            for coord, ah_coord in ah_coords.items():
                # For each coordinate, compute the center of mass along that
                # coordinate
                com = sum(
                    mass * ah / total_mass
                    for mass, ah in zip(masses, ah_coords[coord])
                )
                # Now, we update ah_coords over that coordinate by subtracting
                # the center of mass from each apparent horizon
                ah_coord = [ah - com for ah in ah_coords[coord]]

        to_plot_x, to_plot_y = args.plane
        logger.debug(f"Plotting on the x axis {to_plot_x}")
        logger.debug(f"Plotting on the y axis {to_plot_y}")

        # Now we loop over all the horizons
        for ind, ah in enumerate(args.horizons):
            plt.plot(
                ah_coords[to_plot_x][ind].y,
                ah_coords[to_plot_y][ind].y,
                label=f"Horizon {ah}",
            )

            # We save the time to plot the horizon outline
            time = ah_coords[to_plot_x][ind].tmax

            # Try to draw the shape of the horizon
            if args.draw_ah:
                logger.debug(f"Drawing shape at time {time} for ah {ah}")
                plot_horizon_on_plane_at_time(
                    sim_hor.get_apparent_horizon(ah), time, args.plane
                )

        if args.force_com_at_origin:
            xlabel = f"{to_plot_x} - {to_plot_x}_CM"
            ylabel = f"{to_plot_y} - {to_plot_y}_CM"
        else:
            xlabel, ylabel = to_plot_x, to_plot_y

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.gca().set_aspect("equal")

        plt.legend()
        add_text_to_corner(rf"$t = {time:.3f}$")

        logger.debug("Saving")
        save_from_dir_filename_ext(
            args.outdir,
            figname,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
        logger.debug("DONE")
