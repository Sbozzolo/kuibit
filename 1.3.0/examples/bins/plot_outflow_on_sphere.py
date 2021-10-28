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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from kuibit import argparse_helper as pah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_corner,
    save,
    save_from_dir_filename_ext,
    setup_matplotlib,
)

"""Plot the flux on a sphere as output by the Outflow thorn.
"""

if __name__ == "__main__":
    setup_matplotlib()

    desc = __doc__

    parser = pah.init_argparse(desc)
    pah.add_figure_to_parser(parser)

    parser.add_argument(
        "--detector-num",
        type=int,
        required=True,
        help="Number of the spherical surface over which to read the flux",
    )
    parser.add_argument(
        "--logscale",
        action="store_true",
        help="Whether to use log scale.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to plot. If -1, the latest.",
    )

    args = pah.get_args(parser)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    if args.figname is None:
        figname = f"outflow_det{args.detector_num}"
    else:

        figname = args.figname

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:

        var_name = f"fluxdens_projected[{args.detector_num}]"
        logger.debug(f"Using var name {var_name}")

        logger.debug("Trying to read data from 2D files")
        # TODO: Check that iteration is present

        # From https://stackoverflow.com/a/39535813

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # ref_level -1, component 0

        iteration = args.iteration
        if iteration == -1:
            iteration = var.available_iterations[-1]

        logger.debug(f"Working with iteration {iteration}")

        # TODO: Check if data has to be transposed
        data = var[iteration][-1][0].data
        if args.logscale:
            data = np.ma.log10(data)
            label = "log10(Flux)"
        else:
            label = "Flux"

        u, v = np.mgrid[
            0 : np.pi : 1j * data.shape[0], 0 : 2 * np.pi : 1j * data.shape[1]
        ]

        norm = colors.Normalize(
            vmin=np.min(data), vmax=np.max(data), clip=False
        )

        x = 10 * np.sin(u) * np.cos(v)
        y = 10 * np.sin(u) * np.sin(v)
        z = 10 * np.cos(u)

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

        time = var.time_at_iteration(iteration)
        add_text_to_corner(fr"$t = {time:.3f}$")

        output_path = os.path.join(args.outdir, figname)
        logger.debug(f"Saving in {output_path}")
        save_from_dir_filename_ext(
            args.outdir,
            figname,
            args.fig_extension,
            tikz_clean_figure=args.tikz_clean_figure,
        )
        logger.debug("DONE")
