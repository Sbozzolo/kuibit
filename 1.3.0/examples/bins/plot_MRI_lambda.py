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
import numpy as np

from kuibit import argparse_helper as pah
from kuibit.simdir import SimDir
from kuibit.visualize_matplotlib import (
    add_text_to_figure_corner,
    plot_color,
    save,
    save_from_dir_filename_ext,
    setup_matplotlib,
)

# TODO: Improve error checking

"""Plot an estimate of the wavelength of the magneto-rotational instability on
the equatorial plane.
"""

if __name__ == "__main__":
    setup_matplotlib()

    desc = __doc__

    parser = pah.init_argparse(desc)
    pah.add_grid_to_parser(parser, dimensions=2)
    pah.add_figure_to_parser(parser)
    parser.add_argument(
        "--K_poly",
        type=float,
        required=True,
        help="Polytropic index.",
    )
    parser.add_argument(
        "--Gamma",
        type=float,
        required=True,
        help="Polytropic Gamma.",
    )
    parser.add_argument(
        "--Gamma_th",
        type=float,
        required=True,
        help="Gamma thermal.",
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
    parser.add_argument(
        "--colorbar",
        action="store_true",
        help="Whether to draw the color bar.",
    )
    parser.add(
        "--vmin",
        help=("Minimum value of the variable. "),
        type=float,
    )
    parser.add(
        "--vmax",
        help=("Maximum value of the variable. "),
        type=float,
    )

    args = pah.get_args(parser)

    # Parse arguments

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    with SimDir(
        args.datadir,
        ignore_symlinks=args.ignore_symlinks,
        pickle_file=args.pickle_file,
    ) as sim:

        reader = sim.gridfunctions.xy

        iteration = args.iteration
        x0, x1, res = args.origin, args.corner, args.resolution
        shape = [res, res]

        if iteration == -1:
            iteration = reader["rho_b"].available_iterations[-1]

        if args.figname is None:
            figname = f"MRI_lambda_{iteration}"
        else:
            figname = args.figname

        time = reader["rho_b"].time_at_iteration(iteration)

        logger.debug(f"Using iteration {iteration} (time = {time})")

        vx = reader["vx"][iteration]
        vy = reader["vy"][iteration]
        Bx = reader["Bx"][iteration]
        By = reader["By"][iteration]
        Bz = reader["Bz"][iteration]
        smallb2 = reader["smallb2"][iteration]
        rho_b = reader["rho_b"][iteration]
        P = reader["P"][iteration]

        gxx = reader["gxx"][iteration]
        gxy = reader["gxy"][iteration]
        gxz = reader["gxz"][iteration]
        gyy = reader["gyy"][iteration]
        gyz = reader["gyz"][iteration]
        gzz = reader["gzz"][iteration]

        detg = (
            gxx * gyy * gzz
            + gxy * gyz * gxz
            + gxz * gxy * gyz
            - gxz * gyy * gxz
            - gxy * gxy * gzz
            - gxx * gyz * gyz
        )
        psi4 = detg ** (1.0 / 3.0)
        psi6 = detg.sqrt()
        x, y = detg.coordinates()

        partial_phi_magnitude = (
            psi4 * (gxx * y ** 2.0 + gyy * x ** 2.0 - 2.0 * gxy * x * y)
        ).sqrt()
        e_hat_phix = -y / partial_phi_magnitude
        e_hat_phiy = x / partial_phi_magnitude

        B_x = psi4 * (gxx * Bx + gxy * By + gxz * Bz)
        B_y = psi4 * (gxy * Bx + gyy * By + gyz * Bz)

        B_toroidal = abs(B_x * e_hat_phix + B_y * e_hat_phiy)
        B2 = psi4 * (
            gxx * Bx * Bx
            + 2.0 * gxy * Bx * By
            + 2.0 * gxz * Bx * Bz
            + gyy * By * By
            + 2.0 * gyz * By * Bz
            + gzz * Bz * Bz
        )
        B_poloidal = (B2 - B_toroidal ** 2.0).sqrt()

        Omega = (x * vy - y * vx) / (x ** 2.0 + y ** 2.0)

        K_poly = args.K_poly
        Gamma = args.Gamma
        Gamma_th = args.Gamma_th

        hrho_b = (
            rho_b
            + (K_poly * rho_b ** Gamma) / (Gamma - 1.0)
            + (P - K_poly * rho_b ** Gamma) / (Gamma_th - 1.0)
            + P
        )
        lambda_MRI = (
            2
            * np.pi
            * (B_poloidal * B_poloidal / (smallb2 + hrho_b + 1e-20)).sqrt()
            / (abs(Omega) + 1e-20)
        )

        logger.debug(
            f"Plotting on grid with x0 = {x0}, x1 = {x1}, shape = {shape}"
        )

        data = lambda_MRI.to_UniformGridData(shape=[2000, 10], x0=x0, x1=x1)
        data.slice([None, 0])

        plt.plot(data.coordinates_from_grid()[0], data.data)

        # plot_color(
        #     data,
        #     x0=x0,
        #     x1=x1,
        #     shape=shape,
        #     xlabel="x",
        #     ylabel="y",
        #     resample=args.multilinear_interpolate,
        #     colorbar=args.colorbar,
        #     vmin=args.vmin,
        #     vmax=args.vmax,
        #     label=r"$\lambda_{\mathrm{MRI}}$",
        #     interpolation=args.interpolation_method,
        # )

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
