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
from kuibit.grid_data_utils import load_UniformGridData
from kuibit.simdir import SimDir
from kuibit.visualize_mayavi import (
    add_text_to_figure_corner,
    disable_interactive_window,
    plot_apparent_horizon,
    save,
)

"""This script plots a grid function as read from a saved file."""

if __name__ == "__main__":

    desc = __doc__

    parser = pah.init_argparse(desc)
    pah.add_figure_to_parser(parser)
    pah.add_horizon_to_parser(parser)

    parser.add_argument(
        "--datafile",
        required=True,
        help="File with the data",
    )

    args = pah.get_args(parser)

    # Parse arguments
    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    logger.debug(f"Reading file {args.datafile}")
    data = load_UniformGridData(args.datafile)

    if data.num_dimensions != 3:
        raise NotImplementedError(
            "Only 3D plotting is implemented at the moment"
        )

    disable_interactive_window()

    if args.figname is None:
        figname = f"plot3D"
    else:
        figname = args.figname

    mlab.contour3d(
        *data.coordinates_from_grid(as_same_shape=True),
        data.data,
        transparent=True,
    )

    if args.ah_show:
        with SimDir(
            args.datadir,
            ignore_symlinks=args.ignore_symlinks,
            pickle_file=args.pickle_file,
        ) as sim:

            for ah in sim.horizons.available_apparent_horizons:
                # We don't care about the QLM index here
                hor = sim.horizons[0, ah]
                logger.debug(f"Plotting apparent horizon {ah}")
                plot_apparent_horizon(hor, data.iteration)

        output_path = os.path.join(args.outdir, figname)
        logger.debug(f"Saving in {output_path}")
        save(
            output_path,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
