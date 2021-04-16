#!/usr/bin/env python3

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

"""The :py:mod:`~.grid_data_utils` module provides supporting functions to
use the classes in :py:mod:`~.grid_data`.

The functions available are:
- :py:func:`~.common_bounding_box`: takes a list of :py:class:`~.UniformGrid`
  and returns the smallest bounding box that contains all the grids (returning
  the origin and the corner of the box).
- :py:func:`~.merge_uniform_grids`: takes a list of :py:class:`~.UniformGrid`
  with the same grid spacing and returns a new grid that covers all of them.
- :py:func:`~.load_UniformGridData`: read a :py:class:`~.UniformGridData` from
  a file.
- :py:func:`~.sample_function_from_uniformgrid` samples a given function to
  a given :py:class:`~.UniformGrid`.
- :py:func:`~.sample_function` samples a given function to a given
  :py:class:`~.UniformGrid`, specifying the details of the grid in the function
  call.

"""
import ast  # To read metadata in ASCII files
import re
from bz2 import open as bopen
from gzip import open as gopen
from os.path import splitext

import numpy as np

from kuibit import grid_data as gd


def common_bounding_box(grids):
    """Return the corners of smallest common bounding box of a list of
    :py:class:`~.UniformGrid`.

    :param geoms: list of uniform grids.
    :type geoms:  list of :py:class:`~.UniformGrid`.
    :returns: The common bounding box (``x0`` and ``x1``) of all the grids.
    :rtype: tuple of 1d NumPy arrays

    """
    # Let's check that grids is a list like objects
    if not hasattr(grids, "__len__"):
        raise TypeError("common_bounding_box takes a list")

    # Check that they are all UniformGrids
    if not all(isinstance(g, gd.UniformGrid) for g in grids):
        raise TypeError("common_bounding_boxes takes a list of UniformGrid")

    # We have to check that the number of dimensions is the same
    num_dims = {g.num_dimensions for g in grids}

    if len(num_dims) != 1:
        raise ValueError("Grids have different dimensions")

    # Let's consider three 2d grids as example with
    # x0 being respectively (0, 0), (-1, 0), (-3, 3)
    # x1 being (5, 5), (2, 2), (1, 2)
    x0s = np.array([g.x0 for g in grids])
    x1s = np.array([g.x1 for g in grids])
    # We put x0 and x1 in arrays:
    # x0s = [[0, 0], [-1, 0], [-3, 3]]
    # x1s = [[5, 5], [2, 2], [1, 2]]
    # Now, if we transpose that, we have
    # x0sT = [[0, -1, 0],
    #         [0, 0, 3]]
    # x1sT = [[5, 2, 1],
    #         [5, 2, 2]]
    # In this way we put all the same coordinates in a single
    # row. We can take the minimum and maximum along these rows
    # to find the common bounding box.
    x0 = np.array([min(b) for b in np.transpose(x0s)])
    x1 = np.array([max(b) for b in np.transpose(x1s)])
    return (x0, x1)


def merge_uniform_grids(grids, component=-1):
    """Return a new grid that covers all the grids in the list.

    All geometries must belong to the same refinement level and have the same
    grid spacing.

    :param geoms: list of grid geometries.
    :type geoms:  list of :py:class:`~.UniformGrid`
    :returns: Grid geometry covering all input grids.
    :rtype: :py:class:`~.UniformGrid`

    """
    # Let's check that grids is a list like objects
    if not hasattr(grids, "__len__"):
        raise TypeError("merge_uniform_grids takes a list")

    if not all(isinstance(g, gd.UniformGrid) for g in grids):
        raise TypeError("merge_uniform_grids works only with UniformGrid")

    # Check that all the grids have the same refinement levels
    ref_levels = {g.ref_level for g in grids}

    if len(ref_levels) != 1:
        raise ValueError("Can only merge grids on same refinement level.")

    # Extract the only element from the set (using tuple unpacking)
    (ref_level,) = ref_levels

    dx = [g.dx for g in grids]

    if not np.allclose(dx, dx[0]):
        raise ValueError("Can only merge grids with the same spacing.")

    # Find the bounding box
    x0, x1 = common_bounding_box(grids)

    # The additional 1.5 and 0.5 factors are because the points are
    # cell-centered, so the cells have size

    # dx here is a list of all the dx, we just want one (they are all the same)
    shape = ((x1 - x0) / dx[0] + 1.5).astype(np.int64)

    return gd.UniformGrid(
        shape, x0=x0, dx=dx[0], ref_level=ref_level, component=component
    )


def load_UniformGridData(path, *args, **kwargs):
    """Load file to :py:class:`~.UniformGridData`.

    The file can be a (optionally compressed) ASCII file or a ``.npz``.

    In the first case, the file has to start with the following pattern:
    # shape: {shape}
    # x0: {x0}
    # dx: {dx}
    # ref_level: {ref_level}
    # component: {component}
    # num_ghost: {num_ghost}
    # time: {time}
    # iteration: {iteration}
    where the curly parentheses contain the actual data.
    This metadata is essential to reconstruct the grid information.
    Files like this are generated by the :py:meth:`~.grid_data.save` method.

    :param path: Path of the file to be loaded.
    :type path: str
    :returns: Loaded data.
    :rtype: :py:class:`~.UniformGridData`

    """
    # Note, this function is not tested in test_grid_data_utils but in
    # test_grid_data.

    # Let us start with the easy case, the .npz file
    if splitext(path)[-1] == ".npz":
        # We read everything, then we split off the data and rename
        # what is left as metadata
        grid_details = np.load(path)
        data = grid_details["data"]
        metadata = {
            key: value for key, value in grid_details.items() if key != "data"
        }
    else:  # ASCII
        # We read the header to fill in the grid information
        # The colon separates data from description
        metadata = {
            "shape": None,
            "x0": None,
            "dx": None,
            "ref_level": None,
            "component": None,
            "num_ghost": None,
            "time": None,
            "iteration": None,
        }
        lines_to_read = len(metadata.keys())

        # We try to understand if the file is compressed
        # 1. (.+?) matches any character in a non-greedy way
        #    (which means, if we find .gz or .bz2 we don't match that)
        # 2. (\.(gz|bz2))? matches if we have compression
        # 3. ^ $ means that we match the entire name
        rx_filename = re.compile(r"^(.+?)(\.(gz|bz2))?$")

        filename_match = rx_filename.match(path)

        # What function to use to open the file?
        # What mode?
        decompressor = {
            None: (open, "r"),
            "gz": (gopen, "rt"),
            "bz2": (bopen, "rt"),
        }

        opener, open_mode = decompressor[filename_match.group(3)]

        with opener(path, open_mode) as f:
            # Here we read the first lines_to_read into header
            header = [f.readline().strip() for _ in range(lines_to_read)]

        # The metadata looks like
        # # shape: [50 50 50]
        # # x0: [-10. -10. -10.]
        # # dx: [0.40816327 0.40816327 0.40816327]
        # # ref_level: -1
        # # component: -1
        # # num_ghost: [0 0 0]
        # # time: None
        # # iteration: None

        for line in header:
            # TODO (REFACTORING): Make this into a regex
            #
            # At the moment, we are assuming specific spacing in the line.
            # We can make this much more robust by using a regular expression
            # instead.

            # We read what is after the colon (with space)
            var_data = line.split(": ")

            # The first part of var_data contains the var_name
            # (Note the space)
            var_name = var_data[0].split("# ")[-1]

            # Next we evaluate the second part of var_data as a Python literal
            # using ast, we save this into metadata
            metadata[var_name] = ast.literal_eval(var_data[-1])

        data = np.loadtxt(path).reshape(metadata["shape"])

    # We remove the shape key from metadata because it is not needed
    del metadata["shape"]

    # skipcq: PYL:E1124
    return gd.UniformGridData.from_grid_structure(data, **metadata)


def sample_function_from_uniformgrid(function, grid):
    """Create a regular dataset by sampling a scalar function of the form
    ``f(x, y, z, ...)`` on a grid.

    :param function:  The function to sample.
    :type function:   A callable that takes as many arguments as the number
                      of dimensions (in shape).
    :param grid:   Grid over which to sample the function.
    :type grid:    :py:class:`~.UniformGrid`
    :returns:     Sampled data.
    :rtype:       :py:class:`~.UniformGridData`

    """
    if not isinstance(grid, gd.UniformGrid):
        raise TypeError("grid has to be a UniformGrid")

    # The try except block checks that the function supplied has the correct
    # signature for the grid provided. If you try to pass a function that takes
    # too many of too few arguments, you will get a TypeError

    try:
        ret = gd.UniformGridData(
            grid, np.vectorize(function)(*grid.coordinates(as_same_shape=True))
        )
    except TypeError as type_err:
        # Too few arguments, type_err = missing N required positional arguments: ....
        # Too many arguments, type_err = takes N positional arguments but M were given
        ret = str(type_err)

    # TODO (REFACTORING): This is fragile way to do error parsing
    #
    # We are reading the error message to understand what is going on. This
    # can is not the best way to do this.
    if isinstance(ret, str):
        if "missing" in ret:
            raise TypeError(
                "Provided function takes too few arguments for requested grid"
            )
        if "takes" in ret:
            raise TypeError(
                "Provided function takes too many arguments for requested grid"
            )
        raise TypeError(ret)

    return ret


def sample_function(function, shape, x0, x1, *args, **kwargs):
    """Create a regular dataset by sampling a scalar function of the form
    ``f(x, y, z, ...)`` on a grid.

    You cannot use this function to initialize grids with flat dimensions
    (dimensions with only one grid point).

    :param function:  The function to sample.
    :type function:   A callable that takes as many arguments as the number
                      of dimensions (in shape).
    :param shape: Number of sample points in each dimension.
    :type shape:  1d NumPy array or list of int
    :param x0:    Minimum corner of regular sample grid.
    :type x0:     1d NumPy array or list of float
    :param x0:    Maximum corner of regular sample grid.
    :type x0:     1d NumPy array or list of float
    :returns:     Sampled data.
    :rtype:       :py:class:`~.UniformGridData`

    """
    grid = gd.UniformGrid(shape, x0=x0, x1=x1, *args, **kwargs)
    return sample_function_from_uniformgrid(function, grid)
