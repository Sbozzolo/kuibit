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

import concurrent.futures
import logging
import os

import h5py
import numpy as np

from kuibit import argparse_helper as pah
from kuibit.simdir import SimDir


def fisheye(unnormalized_coords, xmin, xmax, num_points):
    """Fisheye transformation for 1D arrays.

    Takes in the logically-Cartesian coordinates, and returns the corresponding fisheye
    coordinates as defined from xmin to xmax.

    :param unnormalized_coords: Unnormalized (pixel) coordinates.
    :type unnormalized_coords: 1D NumPy array of ints.
    :param xmin: Minimum value of the physical coordinate. coord[0] will be mapped
                 to this value.
    ;type xmin: float
    :param xmin: Maximum value of the physical coordinate. coord[num_points - 1] will be
                 mapped to this value.
    ;type xmin: float

    :param num_points: Maximum value of the physical coordinate. coord[num_points - 1] will be
                 mapped to this value.
    ;type num_points: float

    """
    # NOTE: Here we are hard-coding a specific coordinate transformation.
    # x = sinh((A xi + B))^3, with x physical coordinate and xi unnormalized one.

    B = np.cbrt(np.arcsinh(xmin))
    A = (np.cbrt(np.arcsinh(xmax)) - B) / (num_points - 1)
    return np.sinh((A * unnormalized_coords + B) ** 3)


"""This script takes a simulation, and prepares a HDF5 in the format required by
GRay2 (for radiative-transfer)."""

desc = __doc__

parser = pah.init_argparse(desc)

parser.add_argument(
    "--output",
    type=str,
    default="data.h5",
    help="Path of the output file.",
)
parser.add_argument(
    "--num-points",
    type=int,
    default=100,
    help="Number of points that the final HDF5 has to have along each direction.",
)
parser.add_argument(
    "-x0",
    "--origin",
    type=float,
    nargs=3,
    default=[-100, -100, -100],
)
parser.add_argument(
    "-x1",
    "--corner",
    type=float,
    nargs=3,
    default=[100, 100, 100],
)
parser.add_argument(
    "--precision",
    type=str,
    choices=["single", "double"],
    default="single",
    help="Floating-point precision (default: %(default)s)",
)
parser.add_argument(
    "--parallel",
    action="store_true",
    help="Do this computation using all the cores available",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="If there's an output file already present, ignore it and overwrite it."
    " If this is not set, the groups are added to the HDF5 file, but no consistency check is performed.",
)

args = pah.get_args(parser)

# Parse arguments
output_file = args.output

logger = logging.getLogger(__name__)

if args.verbose:
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger.setLevel(logging.DEBUG)

if args.precision == "single":
    precision = np.single
    logger.debug("Using single precision")
else:
    precision = np.double
    logger.debug("Using double precision")

if not args.overwrite:
    logger.debug("Checking for output file")
    if os.path.exists(output_file) and os.path.isfile(output_file):
        with h5py.File(output_file, "r") as f:
            groups = set(f.keys())
            if "grid" not in groups:
                raise RuntimeError(
                    "Output file does not contain the grid group. It is invalid"
                )
            groups.remove("grid")
            times_in_output = set(map(float, groups))
            if len(times_in_output) > 0:
                logger.debug("Times found:")
                for t in times_in_output:
                    logger.debug(f"{t}")
            else:
                logger.debug("No times found")
    else:
        times_in_output = set()
else:
    times_in_output = set()


sim = SimDir(
    args.datadir,
    ignore_symlinks=args.ignore_symlinks,
    pickle_file=args.pickle_file,
)

reader = sim.gf.xyz
hor_reader = sim.horizons

num_points = args.num_points

# Values 0, 1, 2, 3, .... num_points - 1
unnormalized_coords = np.linspace(
    0, num_points - 1, num_points, dtype=precision
)

# Physical coordinates
xx = fisheye(unnormalized_coords, args.origin[0], args.corner[0], num_points)
yy = fisheye(unnormalized_coords, args.origin[1], args.corner[1], num_points)
zz = fisheye(unnormalized_coords, args.origin[2], args.corner[2], num_points)

var_names = [
    "alp",
    "betax",
    "betay",
    "betaz",
    "gxx",
    "gxy",
    "gxz",
    "gyy",
    "gyz",
    "gzz",
    "Gamma_ttt",
    "Gamma_ttx",
    "Gamma_tty",
    "Gamma_ttz",
    "Gamma_txx",
    "Gamma_txy",
    "Gamma_txz",
    "Gamma_tyy",
    "Gamma_tyz",
    "Gamma_tzz",
    "Gamma_xtt",
    "Gamma_xtx",
    "Gamma_xty",
    "Gamma_xtz",
    "Gamma_xxx",
    "Gamma_xxy",
    "Gamma_xxz",
    "Gamma_xyy",
    "Gamma_xyz",
    "Gamma_xzz",
    "Gamma_ytt",
    "Gamma_ytx",
    "Gamma_yty",
    "Gamma_ytz",
    "Gamma_yxx",
    "Gamma_yxy",
    "Gamma_yxz",
    "Gamma_yyy",
    "Gamma_yyz",
    "Gamma_yzz",
    "Gamma_ztt",
    "Gamma_ztx",
    "Gamma_zty",
    "Gamma_ztz",
    "Gamma_zxx",
    "Gamma_zxy",
    "Gamma_zxz",
    "Gamma_zyy",
    "Gamma_zyz",
    "Gamma_zzz",
    "rho_b",
]

for v in var_names:
    if v not in reader:
        logger.debug(f"Variables available: {reader}")
        raise RuntimeError(f"Variable {v} not available")

logger.debug(f"Using {var_names[0]} to find times")
times = set(reader[var_names[0]].available_times)
logger.debug(f"Available times: {times}")

# Remove the times we already have in the output
for t in times_in_output:
    if t in times:
        times.remove(t)
        logger.debug(f"Removed {t}")

logger.debug(f"Selected times: {times}")

# hor_list. The keys are horizon numbers, the values are dictionaries with keys
# the time and as value a 5-dimensional array: the first three elements are the
# centroid, the other two are minimum and maximum radii.
hor_list = {}


# TODO: We should check that all the variables are defined at all the iterations

var_list = {}
for v in var_names:
    # var_list is a dictionary with keys the var names and values other
    # dictionaries. These latter dictionaries have as keys the time
    # and as values the NumPy arrays of the data at that time.
    var_list[v] = {}


def evaluate_var(args):
    """Resample variable with name at the given iteration on the desired
    physical coordinates. We don't need to be fancy here because resampling
    from a HierarchicalGridData is not really parallelized anyways.

    :param args: Name of the variable to resample and time that has to
                 be read and resampled.
    :type var_name: tuple(str, int)

    :returns: The data on the fisheye coordinates for the requested variable at
              the requested iteration.
    :rtype: NumPy array

    """
    var_name, time = args

    # Read the HierarchicalGridData
    hg_data = reader[var_name].get_time(time)
    data = [[[hg_data((x, y, z)) for x in xx] for y in yy] for z in zz]
    return np.nan_to_num(np.array(data, dtype=precision))


for time in times:
    if args.parallel:
        with concurrent.futures.ProcessPoolExecutor() as exe:
            arguments = [(v, time) for v in var_names]
            for arg, data in zip(arguments, exe.map(evaluate_var, arguments)):
                v, time = arg
                logger.debug(f"Done with variable {v} at time {time}")
                var_list[v][time] = data
    else:
        for v in var_names:
            logger.debug(f"Working on variable {v} at time {time}")
            var_list[v][time] = evaluate_var((v, time))

    # Now we have to compute the 4 metric out of the 3 metric
    beta_x = (
        var_list["gxx"][time] * var_list["betax"][time]
        + var_list["gxy"][time] * var_list["betay"][time]
        + var_list["gxz"][time] * var_list["betaz"][time]
    )
    beta_y = (
        var_list["gxy"][time] * var_list["betax"][time]
        + var_list["gyy"][time] * var_list["betay"][time]
        + var_list["gyz"][time] * var_list["betaz"][time]
    )
    beta_z = (
        var_list["gxz"][time] * var_list["betax"][time]
        + var_list["gyz"][time] * var_list["betay"][time]
        + var_list["gzz"][time] * var_list["betaz"][time]
    )

    beta2 = (
        beta_x * var_list["betax"][time]
        + beta_y * var_list["betay"][time]
        + beta_z * var_list["betaz"][time]
    )

    var_list.setdefault("g_tt", {})
    var_list.setdefault("g_tx", {})
    var_list.setdefault("g_ty", {})
    var_list.setdefault("g_tz", {})

    var_list["g_tt"][time] = (
        beta2 - var_list["alp"][time] * var_list["alp"][time]
    )
    var_list["g_tx"][time] = beta_x
    var_list["g_ty"][time] = beta_y
    var_list["g_tz"][time] = beta_z

    # Horizons
    for hor in hor_reader.available_apparent_horizons:
        ah = hor_reader[0, hor].ah
        if time >= hor_reader[0, hor].formation_time:
            logger.debug(f"Adding horizon {hor} at time {time}")
            ah_x, ah_y, ah_x = (
                ah.centroid_x(time),
                ah.centroid_y(time),
                ah.centroid_z(time),
            )
            min_rad, max_rad = ah.min_radius(time), ah.max_radius(time)

            val = np.array(
                (ah_x, ah_y, ah_x, min_rad, max_rad), dtype=precision
            )
        else:
            val = np.zeros(5, dtype=precision)

        hor_list.setdefault(hor, {})[time] = val


var_names.extend(["g_tt", "g_tx", "g_ty", "g_tz"])
var_names.remove("alp")
var_names.remove("betax")
var_names.remove("betay")
var_names.remove("betaz")


with h5py.File(output_file, "w") as f:
    grid_group = f.create_group("grid")
    grid_group.create_dataset("x", data=xx)
    grid_group.create_dataset("y", data=yy)
    grid_group.create_dataset("z", data=zz)

    for time in times:
        it_group = f.create_group(str(time))

        # Loop over all the horizons
        for hor in hor_list:
            it_group.attrs.create(f"ah_{hor}", hor_list[hor][time])

        for var in var_names:
            data = var_list[var][time]

            #  We have to rename gij to g_ij (sigh)
            if var[1:3] in ("xx", "xy", "xz", "yy", "yz", "zz"):
                var = "g_" + var[1:3]

            it_group.create_dataset(var, data=data)

        def compute_rho(x, y, z):
            """Compute the density as 1/r."""
            return 1 / np.sqrt(x * x + y * y + z * z + 1)

        it_group.create_dataset(
            "rho",
            data=np.array(
                [[[compute_rho(x, y, z) for x in xx] for y in yy] for z in zz]
            ),
            dtype=precision,
        )
