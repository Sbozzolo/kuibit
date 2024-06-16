#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
#
# Inspired by code originally developed by Wolfgang Kastaun. This file may
# contain algorithms and/or structures first implemented in
# GitHub:wokast/PyCactus/PostCactus/grid_data.py
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

"""The :py:mod:`~.grid_data` module provides representations of data on uniform
grids as well as for data on refined grid hierarchies. Standard arithmetic
operations are supported for those data grids, further methods to interpolate
and resample. The number of dimensions is arbitrary.

The important classes defined here are
- :py:class:`~.UniformGridData` represents data on a uniform grid.
- :py:class:`~.HierarchicalGridData` represents data on a refined grid
hierarchy (AMR).

A :py:class:`~.UniformGridData` object contains a :py:class:`~.UniformGrid` one.
Similarly, a :py:class:`~.HierarchicalGridData` contains multiple
:py:class:`~.UniformGridData`.

We also define :py:class:`~.GridSeries`. This is intended to be used for 1D grid
data and it is a way to use the infrastructure for ``Series`` for grid data. The
reason this is useful is that ``Series`` are much simpler and leaner to work
with.

"""
from __future__ import annotations

import warnings
from bisect import bisect_right
from os.path import splitext
from typing import Iterable, Optional

import numpy as np
from scipy import interpolate, linalg

from kuibit import grid_data_utils as gdu
from kuibit.numerical import BaseNumerical
from kuibit.series import BaseSeries
from kuibit.tensor import Tensor
from kuibit.uniform_grid import UniformGrid


class GridSeries(BaseSeries):
    """One-dimensional grid data, handled with the Series infrastructure.

    When the data is one dimensional, sometimes it is more convenient to treat
    it a series instead of grid data. This class is uses the same infrastructure
    as :py:class:`~.TimeSeries` and :py:class:`~.FrequencySeries` and has more
    or less the same features.

    :ivar x: Coordinates.
    :type x: 1D NumPy array
    :ivar y: Values.
    :type y: 1D NumPy array
    """

    def __init__(self, x, y, _=None):
        """Constructor.

        The third argument can be anything. It is required to ensure
        compatibility with other series, but it is not used.

        :param x: Coordinates.
        :type x: 1D NumPy array
        :param y: Values.
        :type y: 1D NumPy array

        """
        super().__init__(x, y, guarantee_x_is_monotonic=True)


class UniformGridData(BaseNumerical):
    """Represents a rectangular data grid with coordinates, supporting
    common arithmetic operations.

    :py:class:`~.UniformGridData` is a combination of a
    :py:class:`~.UniformGrid` (in ``grid`` attribute) and the actual data (in
    the ``data`` attribute). :py:class:`~.UniformGridData` makes sure that all
    the operations on these objects are intuitive, meaningful, and consistent.

    A :py:class:`~.UniformGridData` can be initialized with the default
    constructor (which takes grid and data), of with the alternative constructor
    :py:meth:`~.from_grid_structure` (which takes grid details and data).

    :ivar grid: Uniform grid over which the data is defined.
    :type grid: :py:class:`~.UniformGrid`
    :ivar data: The actual data.
    :type data: NumPy array.

    :ivar invalid_spline: Whether the spline stored is valid.
    :type invalid_spline: bool

    :ivar spline_real: Spline representation of the real part of the data.
    :type spline_real: SciPy's RegularGridInterpolator, or None

    :ivar spline_imag: Spline representation of the imaginary part of the data.
    :type spline_imag: SciPy's RegularGridInterpolator, or None

    """

    # We are deriving this from BaseNumerical. This will give all the
    # mathematical operators for free, as long as we defined _apply_unary
    # and _apply_binary.

    def __init__(self, grid, data):
        """
        :param grid: Uniform grid over which the data is defined.
        :type grid: :py:class:`~.UniformGrid`
        :param data: The data.
        :type data: A NumPy array.
        """
        if not isinstance(grid, UniformGrid):
            raise TypeError("grid has to be a UniformGrid")

        if not np.array_equal(data.shape, grid.shape):
            raise ValueError(
                f"grid and data shapes differ {grid.shape} vs {data.shape}"
            )

        self.grid = grid.copy()
        self.data = data.copy()

        # We keep this flag around to know when we have to recompute the
        # splines
        self.invalid_spline = True
        # Here we also define the splines as empty objects so that we know that
        # they are attributes of the class and they are not uninitialized.
        # These attributes will store data relevant for evaluating the spline.
        # This will be an object of type SciPy's RegularGridInterpolator.
        self.spline_real = None
        self.spline_imag = None

    # This is a class method. It doesn't depend on the specific instance, and
    # it is used as an alternative constructor.
    @classmethod
    def from_grid_structure(
        cls,
        data,
        x0,
        x1=None,
        dx=None,
        ref_level=-1,
        component=-1,
        num_ghost=None,
        time=None,
        iteration=None,
    ):
        """
        :param x0:    Position of cell center with lowest coordinate.
        :type x0:     1d NumPy array or list of float.
        :param dx:     If not None, specifies grid spacing, else grid
                          spacing is computed from x0, x1, and shape.
        :type dx:      1d NumPy array or list of float.
        :param data:      The data.
        :type data:       A NumPy array.
        :param ref_level:  Refinement level if this belongs to a hierarchy,
                          else -1.
        :type ref_level:   int
        :param component: Component number if this belongs to a hierarchy,
                          else -1.
        :type component:  int
        :param num_ghost:    Number of ghost zones (default=0)
        :type num_ghost:     1d NumPy arrary or list of int.
        :param time:      Time if that makes sense, else None.
        :type time:       float or None
        :param iteration: Iteration if that makes sense, else None.
        :type iteration:  float or None

        """
        geom = UniformGrid(
            data.shape,
            x0,
            x1=x1,
            dx=dx,
            ref_level=ref_level,
            component=component,
            num_ghost=num_ghost,
            time=time,
            iteration=iteration,
        )
        return cls(geom, data)

    def coordinates(self):
        """Return coordinates of the grid points as list of
        :py:class:`~.UniformGridData`.

        This can be used for computations involving the coordinates.

        :returns: Coordinates along each direction.
        :rtype: list of :py:class:`~.UniformGridData`

        """
        return [
            type(self)(self.grid, coord)
            for coord in self.coordinates_from_grid(as_same_shape=True)
        ]

    def coordinates_from_grid(self, as_meshgrid=False, as_same_shape=False):
        """Return coordinates of the grid points.

        This is equivalent to ``self.grid.coordinates()``.

        If ``as_meshgrid`` is True, the coordinates are returned as NumPy
        meshgrid. Otherwise, return the coordinates of the grid points as 1D
        arrays (schematically, [array for x coordinates, array for y
        coordinates, ...]).

        If ``as_same_shape`` is True return the coordinates as an array with the
        same shape of self and with values the coordinates. This is useful for
        computations involving the coordinates.

        :param as_meshgrid: If True, return the coordinates as meshgrid.
        :type as_meshgrid: bool
        :param as_same_shape: If True, return the coordinates as a list
                              or coordinates with the same shape of self
                              and with values of a given coordinate.
                              For instance, if ``self.num_dimension = 3`` there
                              will be three lists with ``shape = self.shape``.
        :type as_same_shape: bool
        :returns:  Grid coordinates.
        :rtype:   list of NumPy arrays with the same shape as grid

        """
        return self.grid.coordinates(
            as_meshgrid=as_meshgrid, as_same_shape=as_same_shape
        )

    def coordinates_meshgrid(self):
        """Return coordinates of the grid points as NumPy meshgrid.

        This is syntactic sugar useful for plotting with matplotlib.

        :returns:  Grid coordinates.
        :rtype:   list of NumPy arrays
        """
        return self.coordinates_from_grid(as_meshgrid=True)

    @property
    def data_xyz(self):
        """Return the data, but transposed.

        This is useful when plotting, because we store data in a matrix form,
        which is the transposed of what we are used to thinking about coordinates
        (ie, the first index is not ``x``).

        :returns: Data in a coordinate-friendly form.
        :rtype: NumPy array

        """
        return np.transpose(self.data)

    def save(self, file_name, *args, **kwargs):
        """Save data and grid information to a file.

        Unless the file extension is ``npz``, the output file will ASCII. In
        this case, compression is supported. To enable compression, just append
        ``bz`` or ``gz`` to the extension. All the unknown arguments are passed
        to ``np.savetxt``. The backend used in this case does not support
        writing 3D or larger arrays to disk as ASCII, so all the arrays are
        reshaped to 1D.

        If the file extension is ``npz``, then save the grid with this
        NumPy-specific format (compressed).

        If you look for performance, use ``npz``, if you want a file that you
        can easily read everywhere, use ASCII.

        The file output with this method can be read with the
        :py:func:`~.load_UniformGridData` function.

        :param file_name: Path (with extension) of the output file.
        :type file_name: str

        """
        if self.is_masked():
            warnings.warn(
                "Discarding mask information.",
                RuntimeWarning,
            )

        if splitext(file_name)[-1] == ".npz":
            # Time and iterations could be None, in that case, we don't add them
            others = {}
            if self.time is not None:
                others.update({"time": self.time})
            if self.iteration is not None:
                others.update({"iteration": self.iteration})
            np.savez_compressed(
                file_name,
                shape=self.shape,
                x0=self.x0,
                dx=self.dx,
                ref_level=self.ref_level,
                component=self.component,
                num_ghost=self.num_ghost,
                data=self.data,
                **others,
            )
            return

        # ASCII file
        #
        # In the header we save all the metadata for the grid.
        # We will use colons to read the data from the comment
        header = f"shape: {list(self.shape)}\n"
        header += f"x0: {list(self.x0)}\n"
        header += f"dx: {list(self.dx)}\n"
        header += f"ref_level: {self.ref_level}\n"
        header += f"component: {self.component}\n"
        header += f"num_ghost: {list(self.num_ghost)}\n"
        header += f"time: {self.time}\n"
        header += f"iteration: {self.iteration}"
        np.savetxt(
            file_name,
            self.data.reshape(
                np.prod(self.shape),
            ),
            *args,
            header=header,
            **kwargs,
        )

    @property
    def x0(self):
        """Lower corner.

        :returns: Center of lowest corner grid point.
        :rtype: 1d NumPy array
        """
        return self.grid.x0

    @property
    def shape(self):
        """Number of cells across each dimension.

        :returns: Number of cells across each dimension.
        :rtype: 1d NumPy array
        """
        return self.grid.shape

    @property
    def x1(self):
        """Upper corner.

        :returns: Center of top corner grid point.
        :rtype: 1d NumPy array
        """
        return self.grid.x1

    @property
    def origin(self):
        """Lower corner.

        Alias for :py:meth:`~.x0`.

        :returns: Center of lowest corner grid point.
        :rtype: 1d NumPy array
        """
        return self.x0

    @property
    def dx(self):
        """Grid spacing.

        :returns: Cell size across each dimension.
        :rtype: 1d NumPy array
        """
        return self.grid.dx

    @property
    def delta(self):
        """Grid spacing.

        Alias for :py:meth:`~.dx`.

        :returns: Cell size across each dimension.
        :rtype: 1d NumPy array
        """
        return self.dx

    @property
    def num_ghost(self):
        """Number of ghost zones.

        :returns: Number of ghost zones across each dimension.
        :rtype: 1d NumPy array
        """
        return self.grid.num_ghost

    @property
    def ref_level(self):
        """Refinement level number.

        :returns: Refinement level number.
        :rtype: int
        """
        return self.grid.ref_level

    @property
    def component(self):
        """Component number.

        :returns: Component number.
        :rtype: int
        """
        return self.grid.component

    @property
    def time(self):
        """Time.

        :returns: Time.
        :rtype: float
        """
        return self.grid.time

    @property
    def iteration(self):
        """Iteration number

        :returns: Iteration number.
        :rtype: float
        """
        return self.grid.iteration

    def __getitem__(self, key):
        return self.data[key]

    def _make_spline(self, k=1):
        """Private function to make spline representation of the data using
        ``scipy.interpolate.RegularGridInterpolator``.

        Only nearest neighbor or multilinear interpolations are available.

        Computing spline is memory intenstive: 150 MB/million points.

        This function is not meant to be called directly.

        :param k: Order of the interpolation (k = 0 or 1).
        :type k:  int

        """

        if self.is_masked():
            raise RuntimeError("Splines with masked data are not supported.")

        coords = self.grid.coordinates()

        if k not in (0, 1):
            raise ValueError(
                "Order for splines for dimensions > 2 must be 0 or 1"
            )

        # Here k is 0 or 1
        method = "nearest" if k == 0 else "linear"

        # Our grid is cell-centered, so it is perfecly valid to evaluate a point
        # that it is outside coords, as long as it is within 0.5 * dx. For
        # example, if the grid is linear from 0 to 10 with dx = 1, the point
        # -0.25 is in the grid. To account for this in splines (to avoid that
        # they throw an error), we add another point at the two boundaries. This
        # point as the same value as the last point (which is equivalent to a
        # 0th order interpolation at the very last half cell).

        # With this, the grid has uneven spacing.
        # Add an element at the beginning and end
        for index, coord in enumerate(coords):
            coords[index] = np.concatenate(
                (
                    [coord[0] - 0.5 * self.dx[index]],
                    coord,
                    [coord[-1] + 0.5 * self.dx[index]],
                )
            )

        # Add the border
        data_real = np.pad(self.data.real, pad_width=1, mode="edge")

        self.spline_real = interpolate.RegularGridInterpolator(
            coords,
            data_real,
            method=method,
            fill_value=0,
            bounds_error=True,
        )

        if self.is_complex():
            data_imag = np.pad(self.data.imag, pad_width=1, mode="edge")
            self.spline_imag = interpolate.RegularGridInterpolator(
                coords,
                data_imag,
                method=method,
                fill_value=0,
                bounds_error=True,
            )

        self.invalid_spline = False

    def _nearest_neighbor_interpolation(self, points, ext=2):
        """Return data of nearest neighbors of given points x.

        :param x: Points where to evaluate the data.
        :type x: 1D NumPy array of float, or :py:class:`~.UniformGrid`

        :param ext: How to deal values outside the boundaries. Values outside
                    the interval are set to 0 if ``ext=1``,
                    or an error is raised if ``ext=2``.
        :type ext:  int

        :returns: Values of the data evaluated on the input ``x``.
        :rtype:   1D NumPy array or float

        """
        # To implement the piecewise constant spline, we just lookup the
        # data, so first we get the corresponding indices.
        indices = self.grid.coordinates_to_indices(points)
        # We need the array version because we are going to modify the values
        indices_arr = np.array(indices)

        # Here we have to directly implement the support for ext = 1 and
        # ext = 2. We find all the points that are outside the
        # boundaries.

        # Here we check for every point if they have have negative index
        # or index larger than the shape (number of points)
        outside_indices = np.logical_or(
            np.any(indices_arr < 0, axis=-1),
            np.any(indices_arr >= self.shape),
        )

        if ext == 2:
            if np.any(outside_indices):
                # For ext = 2, we simply have the raise an error if we have
                # any outside_index
                raise ValueError("Point outside the grid")
            # NumPy fancy indexing consists in a list of N tuples each
            # representing a coordinate, so we have to reshape the indices.
            # Here we use this trick:
            # *indices unpacks the indices so that the iterator is over each
            # point. Then, we zip them, which means that we take one element
            # at the time from each dimension. Finally, we convert this iterator
            # to a tuple
            take_indices = tuple(zip(*indices))
            return self.data[take_indices]

        # Here we are with ext = 1. If we were to call self.data[indices], we
        # would have errors because we are trying to access elements outside the
        # array. Therefore, we change these indices to a value. We will
        # overwrite that value with 0 later.

        # Here we substitute those elements that are outside with the point
        # (0, 0, 0, ...) (N zeros with 0 is num dimension)
        indices_arr[outside_indices] = np.zeros(self.num_dimensions, dtype=int)

        # See comment ~10 lines above for what this means
        take_indices = tuple(zip(*indices_arr))
        ret = self.data[take_indices]

        ret[outside_indices] = 0

        return ret

    def evaluate_with_spline(self, x, ext=2, piecewise_constant=False):
        """Evaluate the spline on the points ``x``.

        Values outside the interval are set to 0 if ``ext=1``, or a
        ``ValueError`` is raised if ``ext=2``.

        This method is meant to be used only if you want to use a different ext
        for a specific call, otherwise, just use __call__.

        :param x: Points where to evaluate the data.
        :type x: 1D NumPy array of float, or :py:class:`~.UniformGrid`

        :param ext: How to deal values outside the boundaries. Values outside
                    the interval are set to 0 if ``ext=1``,
                    or an error is raised if ``ext=2``.
        :type ext:  int

        :returns: Values of the data evaluated on the input ``x``.
        :rtype:   1D NumPy array or float

        """
        # ext = 0 is extrapolation and ext = 3 is setting the boundary
        # value. We cannot do this with RegularGridInterpolator

        # TODO (FEATURE): Implement ext = 3
        #
        # We can implement ext = 3 by clamping the indices.

        if ext not in (1, 2):
            raise ValueError("Only ext=1 or ext=2 are available")

        if isinstance(x, UniformGrid):
            if x.num_dimensions != self.num_dimensions:
                raise ValueError(
                    "Incompatible dimensions between input and self"
                )
            # The way we want the coordinates is like as an array with the same
            # shape of the grid and with values the coordinates (as arrays).
            # This is similar to as_same_shape, but the coordinates have to be
            # the value, and not the first index.
            x = np.moveaxis(x.coordinates(as_same_shape=True), 0, -1)

        # Now we make sure we have an array
        x_arr = np.atleast_1d(x)

        # We determine what is the shape of the points forgetting about
        # their dimensionality (which we don't need). We use this reshape
        # the output.
        points_shape = x_arr.shape[:-1]

        # Next, we reshape up to the last axis, which means that
        # now we have a collection of points
        x_arr = x_arr.reshape(-1, x_arr.shape[-1])

        if piecewise_constant or (
            self.num_dimensions != self.num_extended_dimensions
        ):
            ret = self._nearest_neighbor_interpolation(x_arr, ext=ext)

        else:
            # We are here only with method = linear

            if self.invalid_spline:
                self._make_spline()

            # ext = 1 is setting to 0. We set fill_value to 0, so this is the
            # default behavior. We change the bounds_error attribute in
            # RegularGridInterpolator that controls this. By default, we set it
            # to raise an error. We reset it to True when we are done.
            if ext == 1:
                self.spline_real.bounds_error = False
                if self.is_complex():
                    self.spline_imag.bounds_error = False

            def apply_spline(points):
                y_real = self.spline_real(points)
                if self.is_complex():
                    y_imag = self.spline_imag(points)
                    ret = y_real + 1j * y_imag
                else:
                    ret = y_real
                return ret

            ret = apply_spline(x_arr)

            if ext == 1:
                self.spline_real.bounds_error = True
                if self.is_complex():
                    self.spline_imag.bounds_error = True

        # Now we have to reconstruct the correct return shape.
        # First, we determine what is the dimensionality of the output
        # of function
        shape_function_return = ret[0].shape
        # And append this to the shape of the points
        ret_shape = tuple(points_shape + shape_function_return)
        # Finally, we reshape
        ret = ret.reshape(ret_shape)

        return ret

    def __call__(self, x):
        # TODO (FEATURE): Avoid splines when the data is already available.
        #
        # At the moment, the splines are calculated even on points in which
        # data is available. This is probably inefficient.
        return self.evaluate_with_spline(x)

    def sliced(self, cut, resample=False):
        """Return a new :py:class:`~.UniformGridData` obtained slicing the current one.

        ``cut`` specifies how to slice the data. It has to be an array with the
        same number of dimensions of the data. In the entries where ``cut`` is
        None, that dimension is kept, where it is a number, the data is cut
        fixing that coordinate. For example, for a 2D array, if ``cut`` is
        ``[None, 2]``, the cut will be with ``y = 2``.

        If ``resample`` is True, you can cut at any point and we will compute
        the values with multilinear interpolation. If ``resample`` is False, we
        will use the data already available.

        In doing this, dimensions that are only one grid point are lost.

        :param cut: How to slice the array. None entries mean "keep that dimension".
        :type cut:  array or list with dimension
        :param resample: Whether to use multilinear interpolation to compute the
                         data or simply use the value of the closest point.
        :type resample: bool

        :returns: A sliced :py:class:`~.UniformGridData`.
        :rtype: :py:class:`~.UniformGridData`

        """

        # TODO (REFACTORING): Don't repet yourself!
        #
        # There is redundancy in how this function is written. It should be easy
        # to simplify it.

        if np.asarray(cut).shape != (self.num_dimensions,):
            raise ValueError(
                f"{cut} has wrong dimension. Cut has to have the same"
                " dimensions as the grid, and has to have None on the"
                " dimension you want to keep"
            )

        # First we check that we actually have to resample and cut is not all
        # None
        if len(set(cut)) == 1 and cut[0] is None:
            return self.copy()

        # If we have to resample, we simply prepare a new UniformGrid with the
        # coordiantes that we want. We are going to resample keeping the number
        # of dimensions fixed, but setting shape of 1 grid point on those
        # dimensions that have to be cut. Then, we flatten the UniformGridData
        # (which removes dimensions with one grid point).

        if resample:
            new_shape = [
                self.shape[dim] if cut[dim] is None else 1
                for dim in range(self.num_dimensions)
            ]
            new_x0 = [
                self.x0[dim] if cut[dim] is None else cut[dim]
                for dim in range(self.num_dimensions)
            ]
            new_ghost = [
                self.num_ghost[dim] if cut[dim] is None else 0
                for dim in range(self.num_dimensions)
            ]

            new_grid = UniformGrid(
                new_shape,
                x0=new_x0,
                dx=self.dx,
                ref_level=self.ref_level,
                component=self.component,
                num_ghost=new_ghost,
                time=self.time,
                iteration=self.iteration,
            )

            # We used "resampled" to make a copy, then "flat_dimensions_remove"
            # to modify that (so that we don't make a new copy)
            sliced_data = self.resampled(new_grid)
            sliced_data.flat_dimensions_remove()
            return sliced_data

        new_shape = [
            self.shape[dim]
            for dim in range(self.num_dimensions)
            if cut[dim] is None
        ]
        new_x0 = [
            self.x0[dim]
            for dim in range(self.num_dimensions)
            if cut[dim] is None
        ]
        new_dx = [
            self.dx[dim]
            for dim in range(self.num_dimensions)
            if cut[dim] is None
        ]
        new_ghost = [
            self.num_ghost[dim]
            for dim in range(self.num_dimensions)
            if cut[dim] is None
        ]

        new_grid = UniformGrid(
            new_shape,
            x0=new_x0,
            dx=new_dx,
            ref_level=self.ref_level,
            component=self.component,
            num_ghost=new_ghost,
            time=self.time,
            iteration=self.iteration,
        )

        # Here we are not resampling, so we just have to properly cut the array.
        # We prepare a slicer array which defines where to cut.
        slicer = []
        # We walk through all the dimensions, if some unreasonable cuts are
        # requsted, we throw an error, otherwise we find the index of the
        # data element where to cut.
        for dim in range(self.num_dimensions):
            if cut[dim] is None:
                slicer.append(slice(None))
            else:
                if not (
                    self.grid.lowest_vertex[dim]
                    <= cut[dim]
                    < self.grid.highest_vertex[dim]
                ):
                    # The slice method in HierarchicalGridData matches this
                    # error message, so you change it, update the corresponding
                    # method.
                    raise ValueError("Cut point is outside the grid")
                # Transform from coordinate to index
                index = int((cut[dim] - self.x0[dim]) / self.dx[dim] + 0.5)
                slicer.append(index)

        slicer = tuple(slicer)
        return type(self)(new_grid, self.data[slicer])

    def slice(self, cut, resample=False):
        """Slice the data along given direction.

        ``cut`` specifies how to slice the data. It has to be an array with the
        same number of dimensions of the data. In the entries where ``cut`` is
        None, that dimension is kept, where it is a number, the data is cut
        fixing that coordinate. For example, for a 2D array, if ``cut`` is
        ``[None, 2]``, the cut will be with ``y = 2``.

        If ``resample`` is True, you can cut at any point and we will compute
        the values with multilinear interpolation. If ``resample`` is False, we
        will use the data already available.

        In doing this, dimensions that are only one grid point are lost.

        :param cut: How to slice the array. None entries mean "keep that dimension".
        :type cut:  array or list with dimension
        :param resample: Whether to use multilinear interpolation to compute the
                         data or simply use the value of the closest point.
        :type resample: bool

        """
        self._apply_to_self(self.sliced, cut=cut, resample=resample)

    def resampled(self, new_grid, ext=2, piecewise_constant=False):
        """Return a new :py:class:`~.UniformGridData` resampled to ``new_grid``.

        If you want to resample without using the spline, and you want a nearest
        neighbor resampling, pass the keyword ``piecewise_constant=True``. This
        may be a good choice for data with large discontinuities, where the
        splines are ineffective.

        :param new_grid: New independent variable.
        :type new_grid:  1D NumPy array or list of float
        :param ext: How to handle points outside the data interval.
        :type ext: 1 for returning zero, 2 for ``ValueError``,
        :param piecewise_constant: Do not use splines, use the nearest neighbors.
        :type piecewise_constant: bool
        :returns: Resampled data.
        :rtype:   :py:class:`~.UniformGridData`

        """
        if not isinstance(new_grid, UniformGrid):
            raise TypeError("Resample takes another UniformGrid")

        # If grid is the same, there's no need to resample
        if self.grid == new_grid:
            return self.copy()

        return type(self)(
            new_grid,
            self.evaluate_with_spline(
                new_grid, ext=ext, piecewise_constant=piecewise_constant
            ),
        )

    def is_complex(self):
        """Return whether the data is complex.

        :returns:  True if the data is complex, false if it is not.
        :rtype:   bool

        """
        return issubclass(self.data.dtype.type, complex)

    @property
    def mask(self):
        """Return where the data is valid (according to the mask).

        :returns: Array of True/False of the same shape of the data.
                  False where the data is valid, True where is not.
        :rtype: array of bool
        """
        if self.is_masked():
            return self.data.mask
        return np.zeros(self.data.shape, dtype=bool)

    def is_masked(self):
        """Return whether the data is masked.

        :returns:  True if the data is masked, false if it is not.
        :rtype:   bool

        """
        return np.ma.is_masked(self.data)

    @property
    def dtype(self):
        return self.data.dtype

    def _apply_to_self(self, f, *args, **kwargs):
        """Apply the method ``f`` to ``self``, modifying ``self``.

        This function is used to implement those methods that act on the object
        starting from methods that return a new object. The function has to
        return a new copy of the object (not a reference).

        :param f: Method to apply.
        :type f: callable

        """
        ret = f(*args, **kwargs)
        self.grid, self.data = ret.grid, ret.data
        # We have to recompute the splines
        self.invalid_spline = True

    def flat_dimensions_removed(self):
        """Return a new :py:class:`~.UniformGridData` with dimensions of one grid point
        removed.

        :returns: New :py:class:`UniformGridData` without flat dimensions.
        :rtype: :py:class:`UniformGridData`

        """
        new_grid = self.grid.flat_dimensions_removed()
        new_data = self.data.reshape(new_grid.shape)
        return type(self)(new_grid, new_data)

    def flat_dimensions_remove(self):
        """Remove dimensions which are only one gridpoint large."""
        self._apply_to_self(self.flat_dimensions_removed)

    def ghost_zones_removed(self):
        """Return a new :py:class:`UniformGridData` with all the ghost zones removed.

        :returns: New :py:class:`UniformGridData` without ghostzones.
        :rtype: :py:class:`UniformGridData`
        """
        if np.amax(self.num_ghost) == 0:
            return self.copy()

        new_grid = self.grid.ghost_zones_removed()
        # We remove the borders from the data using the slicing operator
        slicer = tuple(
            slice(ghost_zones, -ghost_zones) for ghost_zones in self.num_ghost
        )
        new_data = self.data[slicer]
        return type(self)(new_grid, new_data)

    def ghost_zones_remove(self):
        """Remove all the ghost zones."""
        self._apply_to_self(self.ghost_zones_removed)

    def _roto_reflection_symmetry_undone(
        self,
        dimension: int,
        parity: int = 1,
        second_reflect_dimension: Optional[int] = None,
    ) -> UniformGridData:
        """Core routine to undo reflections and rotations.

        Undo a reflection about the given dimension.

        The parameter ``parity`` determines how to fill the data.

        This method works only if the data crosses the value 0 along the given
        dimension.

        We assume that the reflection will always be from the positive side to
        the negative. Pre-existing data in the negative side will be overwritten.

        Under certain conditions (which are always met in simulations with
        Carpet), undoing a rotation180 symmetry is the same as undoing a
        reflection symmetry combined with reversing the new data along the other
        dimension involved in the rotation.

        Consider the pictorial representation of a grid:

                 y
                 |
                 |    O
                 |
        ------------------- x
                 |
                 |
                 |

        First, we perform a reflection along x, and we get

                 y
                 |
            O    |    O
                 |
        ------------------- x
                 |
                 |
                 |

        Next, we reverse the newly copied data

                 y
                 |
                 |    O
                 |
        ------------------- x
                 |
            O    |
                 |


        This is a rotation!

        Note that this makes sense only if the grid is symmetric along the other
        axes. So, what we have to do is reverse the array for the other axis in
        the plane of rotation.

        This is why there's a second parameter ``second_reflect_axis``. This is
        a second reflection that occurs only in the part of data that has been
        newly minted. For it to work, the grid has to be symmetric in this
        direction.

        :param dimension: Along which dimension to fill in the data
        :type parity: int
        :param parity: Multiplied the filled that with this value. Useful to
                       change sign of vectors.
        :type parity: int
        :param second_reflect_dimension:
        :type second_reflect_dimension: int

        :returns: New :py:class:`UniformGridData` with data filled with points
                  obtained by applying one reflection or two.
        :rtype: :py:class:`UniformGridData`

        """
        if parity not in (-1, 1):
            raise ValueError(
                f"Parity has to be either 1 or -1, cannot be {parity}"
            )

        if not (0 <= dimension < self.num_dimensions):
            raise ValueError(
                f"Dimension has to be between 0 and {self.num_dimensions} (it is {dimension})"
            )

        # We assume that we are going to reflect from the positive side to the
        # negative

        # 0 is not in the data
        if self.x0[dimension] > 0:
            raise ValueError("Cannot reflect data that does not cross 0")

        if second_reflect_dimension is not None:
            # The grid has to be symmetric if we reflect along a second dimension
            if not np.isclose(
                self.x0[second_reflect_dimension],
                -self.x1[second_reflect_dimension],
                atol=1e-14,
            ):
                raise RuntimeError(
                    f"Grid is not symmetric in the {second_reflect_dimension} direction"
                )

        # See self.grid.coordinates_to_indices():
        # We convert the coordinate 0.0 to the index, this will always overestimate, so
        # it will always be the first positive
        index_first_positive = int(
            (0.0 - self.x0[dimension]) / self.dx[dimension] + 0.5
        )

        # Next we check that the grid is indeed symmetric about 0. So, the first
        # positive point has to be such that, when reflected, it has the correct
        # dx.
        #
        # For example: -2, -0.75, 0.5, 1.75, 3 is not a good grid. The dx here
        # is 1.25, but, when reflected, 0.5 -> -0.5, and the dx would be 1.
        #
        # But -2, 1, 3 is because 1 -> -1, and dx = 2

        # Compare UniformGrid.indices_to_coordinates
        coordinates_first_positive_dim = (
            index_first_positive * self.dx[dimension] + self.x0[dimension]
        )
        # This could actually be 0. If it is zero, we have to take this into
        # account. The zero value should not be copied.
        first_element_is_zero = (
            1
            if np.isclose(coordinates_first_positive_dim, 0, atol=1e-14)
            else 0
        )

        # 2 * coordinates_first_positive_dim = first_positive - (-first_positive)
        if not (
            first_element_is_zero
            or np.isclose(
                2 * coordinates_first_positive_dim,
                self.dx[dimension],
                atol=1e-14,
            )
        ):
            raise ValueError(
                f"Grid is not symmetric along reflection axis {dimension}"
            )

        num_elements_to_copy = (
            self.shape[dimension]
            - index_first_positive
            - first_element_is_zero
        )

        new_shape = self.shape.copy()
        # The new shape is the old shape + num_elements_to_copy - the negative
        # values (that are going to be overwritten) and the possible zero.
        new_shape[dimension] += num_elements_to_copy - index_first_positive

        # Prepare the output array
        # We overwrite it with the data
        new_data = np.zeros(new_shape, dtype=self.dtype)

        # First, we copy the new region

        # The following are slicers that copy everything. This is what we want,
        # except for the given dimension.
        destination = [slice(None, None) for _ in self.shape]
        source = [slice(None, None) for _ in self.shape]
        # Copy in indices 0, 1, ..., num_elements_to_copy - 1
        destination[dimension] = slice(0, num_elements_to_copy)

        # If we have a second reflection to perform, we want the data to be
        # reversed in this region
        if second_reflect_dimension:
            destination[second_reflect_dimension] = slice(None, None, -1)

        # We have to read the data backwards from the source, so we read from the end
        # to index_first_positive - 1 (not included)
        source[dimension] = slice(
            self.shape[dimension],
            index_first_positive + first_element_is_zero - 1,
            -1,
        )

        new_data[tuple(destination)] = parity * self.data[tuple(source)]

        # If we have a second reflection, the data has to be the same for the
        # old region, so we restore a copy slicer.
        if second_reflect_dimension:
            destination[second_reflect_dimension] = slice(None, None)

        # Now we copy the data we already had
        #
        # num_elements_to_copy - 1 because we start counting from 0
        destination[dimension] = slice(
            num_elements_to_copy, new_shape[dimension]
        )
        source[dimension] = slice(index_first_positive, self.shape[dimension])

        new_data[tuple(destination)] = self.data[tuple(source)]

        # And we need to extend the grid too

        # Compute the new x0
        new_x0 = self.x0
        new_x0[dimension] = -self.x1[dimension]

        new_grid = UniformGrid(
            shape=new_shape,
            x0=new_x0,
            x1=self.grid.x1,
            dx=self.grid.dx,
            ref_level=self.grid.ref_level,
            component=self.grid.component,
            num_ghost=self.grid.num_ghost,
            time=self.grid.time,
            iteration=self.grid.iteration,
        )

        return type(self)(new_grid, new_data)

    def rotation180_symmetry_undone(
        self,
        dimension: int = 0,
        plane: Iterable[int] = (0, 1),
        parity: int = 1,
    ) -> UniformGridData:
        """Return a new UniformGridData with rotational 180 symmetry undone.

        `dimension` identifies the region for which we only have half of the
        data (in Carpet it is always going to be 0--the x axis). `plane`
        specifies where we are performing the rotation (in Carpet it is always
        (0, 1)--rotation about the z axis). `plane` is specified by providing a
        tuple with the two dimensions involved in the rotation (one of the two has to be `dimension`).

        This method works only if the data crosses the value 0 along the given
        dimension, and if the coordinates are symmetric with respect to the
        other dimension involved in the rotation.

        We assume that the rotation will always be from the positive side to
        the negative. Pre-existing data in the negative side will be overwritten.

        This will change the shape of the object.

        :param dimension: Dimension about which to fill in missing data
        :type dimension: int
        :param plane: Plane on which the rotation occurs specified by specifying
                      the two dimensions involved in the rotation.
        :type plane: tuple of ints
        :param parity: Fill the data assuming that the function is even (parity = 1),
                       or odd (parity = -1).
        :type parity: 1 or -1

        """
        if dimension not in plane:
            raise ValueError(
                "Rotation plane does not include dimension for which the symmetry has to be undone"
            )

        if len(plane) != 2:
            raise ValueError(
                "Rotation plane has to be identified by its two dimensions"
            )

        # We need to figure out what is the second axis involved in the rotation
        # (in addition to `dimension`). So, we take the set difference between
        # `set(plane)` and `{dimension}`, which returns a set with only one
        # element. We use tuple unpacking to extract this element.
        (second_reflect_dimension,) = set(plane) - {dimension}

        return self._roto_reflection_symmetry_undone(
            dimension,
            parity=parity,
            second_reflect_dimension=second_reflect_dimension,
        )

    def rotation180_symmetry_undo(
        self,
        dimension: int = 0,
        plane: Iterable[int] = (0, 1),
        parity: int = 1,
    ) -> None:
        """Undo rotational 180 symmetry on the given plane for given dimension.

        `dimension` identifies the region for which we only have half of the
        data (in Carpet it is always going to be 0--the x axis). `plane`
        specifies where we are performing the rotation (in Carpet it is always
        (0, 1)--rotation about the z axis). `plane` is specified by providing a
        tuple with the two dimensions involved in the rotation (one of the two has to be `dimension`).

        This method works only if the data crosses the value 0 along the given
        dimension, and if the coordinates are symmetric with respect to the
        other dimension involved in the rotation.

        We assume that the rotation will always be from the positive side to
        the negative. Pre-existing data in the negative side will be overwritten.

        This will change the shape of the object.

        :param dimension: Dimension about which to fill in missing data
        :type dimension: int
        :param plane: Plane on which the rotation occurs specified by specifying
                      the two dimensions involved in the rotation.
        :type plane: tuple of ints
        :param parity: Fill the data assuming that the function is even (parity = 1),
                       or odd (parity = -1).
        :type parity: 1 or -1
        """
        self._apply_to_self(
            self.rotation180_symmetry_undone,
            dimension=dimension,
            plane=plane,
            parity=parity,
        )

    def reflection_symmetry_undone(
        self, dimension: int, parity: int = 1
    ) -> UniformGridData:
        """Return a new UniformGridData with reflection symmetry undo for the given dimension.

        The parameter ``parity`` determines how to fill the data.

        This method works only if the data crosses the value 0 along the given
        dimension.

        We assume that the reflection will always be from the positive side to
        the negative. Pre-existing data in the negative side will be overwritten.

        This will change the shape of the object.

        :param dimension: Dimension that has to be reflected.
        :type dimension: int

        :param parity: Fill the data assuming that the function is even (parity = 1),
                       or odd (parity = -1).
        :type parity: 1 or -1

        :returns: New :py:class:`UniformGridData` with values explicitly set for
                  reflected data.
        :rtype: :py:class:`~.UniformGridData`

        """

        # NOTE: this cannot be applied to HierarchicalGridData. In that case we would
        # need to add new components all together.

        return self._roto_reflection_symmetry_undone(
            dimension=dimension, parity=parity, second_reflect_dimension=None
        )

    def reflection_symmetry_undo(self, dimension, parity=1):
        """Undo reflection symmetry for the given dimension.

        This method works only if the data crosses the value 0 along the given dimension.

        This will change the shape of the object.

        :param dimension: Dimension that has to be reflected.
        :type dimension: int

        :param parity: Fill the data assuming that the function is even (parity = 1),
                       or odd (parity = -1).
        :type parity: 1 or -1
        """
        self._apply_to_self(
            self.reflection_symmetry_undone, dimension, parity=parity
        )

    def dx_changed(self, new_dx, piecewise_constant=False):
        """Return a new :py:class:`UniformGridData` with the same grid extent, but with
        a new spacing. This effectively up-samples or down-samples the grid.

        Missing data is obtained with splines.

        ``new_dx`` has to be an integer multiple of the current ``dx`` (or vice
        versa).

        If ``piecewise_constant=True``, the missing information is obtained with
        from the nearest neighbors.

        :param new_dx: Do not use splines, use the nearest neighbors.
        :type new_dx: 1d NumPy array
        :param piecewise_constant: Do not use splines, use the nearest neighbors.
        :type piecewise_constant: bool

        :returns: Data with new grid spacing ``new_dx``.
        :rtype: :py:class:`~.UniformGridData`

        """
        if not hasattr(new_dx, "__len__"):
            raise TypeError(
                f"dx has to be a list or an array. {new_dx} is not"
            )

        # First, we check that new_dx is and dx are compatible
        if len(new_dx) != self.num_dimensions:
            raise ValueError(
                "Provided dx has not the correct number of dimensions"
            )

        # If we don't have to change dx, just return a copy
        if np.allclose(new_dx, self.dx):
            return self.copy()

        for new, old in zip(new_dx, self.dx):
            if not ((new / old).is_integer() or (old / new).is_integer()):
                raise ValueError(
                    "Provided dx is not an integer multiple or factor of current dx"
                )

        # new_dx can have zero entries, for which a shape of 1 should correspond.
        # There can zero entries, we substitute them with -1, so that we
        # can identify them as negative numbers
        new_dx = np.array([dx if dx > 0 else -1 for dx in new_dx])
        new_shape = ((self.x1 - self.x0) / new_dx + 1.5).astype(int)
        new_shape = np.array([s if s > 0 else 1 for s in new_shape])

        new_grid = UniformGrid(
            new_shape,
            x0=self.x0,
            dx=new_dx,
            ref_level=self.ref_level,
            component=self.component,
            num_ghost=self.num_ghost,
            time=self.time,
            iteration=self.iteration,
        )

        return self.resampled(new_grid, piecewise_constant=piecewise_constant)

    def dx_change(self, new_dx, piecewise_constant=False):
        """Up-samples or down-samples the grid data.

        Missing data is obtained with splines.

        ``new_dx`` has to be an integer multiple of the current ``dx`` (or vice
        versa).

        If ``piecewise_constant=True``, the missing information is obtained with
        from the nearest neighbors.

        :param new_dx: Do not use splines, use the nearest neighbors.
        :type new_dx: 1d NumPy array
        :param piecewise_constant: Do not use splines, use the nearest neighbors.
        :type piecewise_constant: bool
        """

        self._apply_to_self(
            self.dx_changed, new_dx, piecewise_constant=piecewise_constant
        )

    def copy(self):
        """Return a deep of self."""
        return type(self)(self.grid, self.data)

    @property
    def num_dimensions(self):
        """Number of dimensions of the grid.

        :returns: Number of dimensions of the grid.
        :rtype:   float
        """
        return self.grid.num_dimensions

    @property
    def num_extended_dimensions(self):
        """Return the number of dimensions with size larger than one gridpoint.

        :returns: The number of extended dimensions (the ones with more than one cell).
        :rtype:   int
        """
        return self.grid.num_extended_dimensions

    @property
    def extended_dimensions(self):
        """Return an array of bools with whether a dimension has more than one
        point or not.

        :returns: Dimensions with more than one point.
        :rtype:   1d NumPy of bools
        """
        return self.grid.extended_dimensions

    def integral(self):
        """Compute the integral over the whole volume of the grid.

        :returns: The integral computed as volume-weighted sum.
        :rtype:   float (or complex if data is complex).
        """
        return self.data.sum() * self.grid.dv

    def mean(self):
        """Compute the mean of the data over the whole volume of the grid.

        :returns: Arithmetic mean of the data.
        :rtype:   float (or complex if data is complex).
        """
        return np.mean(self.data)

    average = mean

    def norm_p(self, order):
        r"""Compute the norm of order ``p`` over the whole volume of the grid.

        :math:`\|u\|_p = (\sum \|u\|^p dv)^1/p`

        :param order: Order of the norm.
        :type order: int

        :returns: The norm2 computed as volume-weighted sum.
        :rtype:   float (or complex if data is complex).
        """
        return linalg.norm(np.ravel(self.data), ord=order) * self.grid.dv ** (
            1 / order
        )

    def norm2(self):
        r"""Compute the norm over the whole volume of the grid.

        :math:`\|u\|_2 = (\sum \|u\|^2 dv)^1/2`

        :returns: The norm2 computed as volume-weighted sum.
        :rtype:   float (or complex if data is complex).
        """
        return self.norm_p(order=2)

    def norm1(self):
        r"""Compute the norm over the whole volume of the grid.

        :math:`\|u\|_1 = \sum \|u\| dv`

        :returns: The norm2 computed as volume-weighted sum.
        :rtype:   float (or complex if data is complex).
        """
        return self.norm_p(order=1)

    def histogram(
        self,
        weights=None,
        min_value=None,
        max_value=None,
        num_bins=400,
        **kwargs,
    ):
        """Return the 1D Histogram of the data.

        :param weights:    The weight for each cell. Default is one.
        :type weights:     :py:class:`~.UniformGridData` or NumPy array of same shape or None.
        :param min_value: Lower bound of data to consider. Default is data range.
        :type min_value: float or None
        :param max_value: Upper bound of data to consider. Default is data range.
        :type max_value: float or None
        :param num_bins: Number of bins to create.
        :type num_bins: int > 1

        :returns: The positions of the data bins and the distribution.
        :rtype:   tuple of two 1D NumPy arrays.
        """
        # Function from Wolfgang Kastaun's PostCactus

        if self.is_complex():
            raise ValueError("Histogram only works with real data")

        if min_value is None:
            min_value = self.min()
        if max_value is None:
            max_value = self.max()

        if isinstance(weights, UniformGridData):
            weights = weights.data

        # Check that we have a NumPy array or None
        if weights is not None and not isinstance(weights, np.ndarray):
            raise TypeError(
                "Weights has to be a UniformGrid, NumPy array or None"
            )

        return np.histogram(
            self.data,
            range=(min_value, max_value),
            bins=num_bins,
            weights=weights,
            **kwargs,
        )

    def percentiles(
        self,
        fractions,
        weights=None,
        relative=True,
        min_value=None,
        max_value=None,
        num_bins=400,
    ):
        """Find values for which a given fraction(s) of the data is smaller.

        Optionally, the cells can have an optional weight, and absolute counts
        can be used instead of fraction.

        :param fractions: List of fraction/absolute values.
        :type fractions:  list or array of floats
        :param weights:    The weight for each cell. Default is one.
        :type weights:     :py:class:`~.UniformGridData` or NumPy array of same shape or None.
        :param relative:   Whether fractions refer to relative or absolute count.
        :type relative:    bool
        :param min_value: Lower bound of data to consider. Default is data range.
        :type min_value: float or None
        :param max_value: Upper bound of data to consider. Default is data range.
        :type max_value: float or None
        :param num_bins:      Number of bins to create.
        :type num_bins:       integer > 1

        :returns: Data values corresponding to the given fractions.
        :rtype:   1D NumPy array
        """
        # Function from Wolfgang Kastaun's PostCactus

        hist_values, bin_edges = self.histogram(
            min_value=min_value,
            max_value=max_value,
            num_bins=num_bins,
            weights=weights,
        )

        hist_cumulative = np.cumsum(hist_values)

        # So that the last element is 1
        if relative:
            # We need to make sure that the everything is float here,
            # otherwise NumPy complains
            hist_cumulative = 1.0 * hist_cumulative
            hist_cumulative /= hist_cumulative[-1]

        # We remove the first point because all the data is larger than that.
        bin_edges = bin_edges[1:]

        # So that we can use it as an array
        fractions = np.array(fractions)

        # We must make sure that fractions is not larger than the amount of data
        # (or of 1, in the case of normalized histogram). If input fraction is
        # larger than 1, the output must be 100 % of the data anyways.
        #
        # We make sure that this is at least 1d so that we can loop over it
        capped_fractions = np.atleast_1d(
            np.minimum(hist_cumulative[-1], fractions)
        )
        # Here we return the first element of the array bin edges that is larger
        # than each element in capped_fractions
        percentiles = np.array(
            [bin_edges[hist_cumulative >= f][0] for f in capped_fractions]
        )

        if len(percentiles) == 1:
            return percentiles[0]
        return percentiles

    def mask_applied(self, mask, ignore_existing=False):
        """Return a new :py:class:`~.UniformGridData` with given mask applied to the
        data.

        If a previous mask already exists, the new mask will be added on top,
        unless ``ignore_existing`` is True.

        :param mask: Array of booleans that identify where the data is invalid.
                     This can be obtained with the method :py:meth:`~.mask`.
        :type mask: NumPy array

        :param ignore_existing: If True, overwrite any previously existing mask.
        :type ignore_existing: bool

        :returns: New grid data with mask applied.
        :rtype: :py:class:`~.UniformGridData`

        """
        if self.is_masked() and not ignore_existing:
            mask = np.ma.mask_or(mask, self.mask)

        return type(self)(self.grid, np.ma.MaskedArray(self.data, mask=mask))

    def mask_apply(self, mask, ignore_existing=False):
        """Apply the given mask.

        If a previous mask already exists, the new mask will be added on top,
        unless ``ignore_existing`` is True.

        :param mask: Array of booleans that identify where the data is invalid.
                     This can be obtained with the method :py:meth:`~.mask`.
        :type mask: NumPy array

        :param ignore_existing: If True, overwrite any previously existing mask.
        :type ignore_existing: bool

        """
        self._apply_to_self(
            self.mask_applied, mask, ignore_existing=ignore_existing
        )

    def partial_differentiated(self, direction, order=1, accuracy_order=2):
        """Return a :py:class:`~.UniformGridData` that is the numerical
        order-differentiation of the present grid_data along a given direction.
        (``order`` = number of derivatives, ie ``order=2`` is second derivative)

        The derivative is calculated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of ``self``.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int
        :param direction: Direction of the partial derivative.
        :type direction: int
        :param accuracy_order: Order of accuracy of the finite difference scheme.
        :type accuracy_order: int

        :returns:  New :py:class:`~.UniformGridData` with derivative.
        :rtype:    :py:class:`~.UniformGridData`

        """
        if self.is_masked():
            raise RuntimeError(
                "Differentiation with masked data is not supported."
            )

        if direction < 0 or direction >= self.num_dimensions:
            raise ValueError(
                f"Grid has {self.num_dimensions}, dimensions, "
                f"{direction} is not available"
            )

        if self.shape[direction] < accuracy_order + 1:
            raise ValueError(
                f"Need at least {accuracy_order + 1} points for finite difference."
            )

        ret_value = np.zeros_like(self.data)

        def slicer(index0: int, index1: int) -> slice:
            """Create a slicer objects between indices index0 and index1 along the direction
            where the derivative is taken. Everything else is left untouched

            """
            slice_ = [slice(None) for _ in self.shape]
            slice_[direction] = slice(index0, index1)
            return tuple(slice_)

        def slicer1(index0: int) -> slice:
            """Create a slicer objects that isolated the given ``index0`` along the
            direction of the derivative.

            """
            # If index0 is -1, we are considering the end of the list
            index1 = index0 + 1 if index0 != -1 else None
            return slicer(index0, index1)

        dx = self.dx[direction]

        # We will set data to _ret_value after we computed each oder
        data = self.data

        for _num_deriv in range(order):
            if accuracy_order == 2:
                # Bulk, equivalent to f[i] - f[i + i]) / 2 dx
                ret_value[slicer(1, -1)] = (
                    data[slicer(2, None)] - data[slicer(0, -2)]
                ) / (2 * dx)
                # Second order forward difference for first element
                # -(3*y[0] - 4*y[1] + y[2]) / (2*dx)
                ret_value[slicer1(0)] = -(
                    3 * data[slicer1(0)]
                    - 4 * data[slicer1(1)]
                    + data[slicer1(2)]
                ) / (2 * dx)
                # Backwards difference final element
                # (3*y[-1] - 4*y[-2] + y[-3]) / (2*dx)
                ret_value[slicer1(-1)] = (
                    3 * data[slicer1(-1)]
                    - 4 * data[slicer1(-2)]
                    + data[slicer1(-3)]
                ) / (2 * dx)
            elif accuracy_order == 4:
                # Coefficients computed with
                # https://web.media.mit.edu/~crtaylor/calculator.html

                # Bulk, equivalent to (f[:4] - 8*f[1:-3] + 8*f[3:-1] - f[4:])/12.0
                ret_value[slicer(2, -2)] = (
                    data[slicer(None, -4)]
                    - 8.0 * data[slicer(1, -3)]
                    + 8.0 * data[slicer(3, -1)]
                    - data[slicer(4, None)]
                ) / (12.0 * dx)
                # Fourth order partially forward difference for second element
                # (-3*f[0]-10*f[1]+18*f[2]-6*f[3]+1*f[4])/(12 dx)
                ret_value[slicer1(1)] = (
                    -3 * data[slicer1(0)]
                    - 10 * data[slicer1(1)]
                    + 18 * data[slicer1(2)]
                    - 6 * data[slicer1(3)]
                    + 1 * data[slicer1(4)]
                ) / (12 * dx)
                # Fourth order fully forward difference for first element
                # (-25*f[0]+48*f[1]-36*f[2]+16*f[3]-3*f[4])/(12dx)
                ret_value[slicer1(0)] = (
                    -25 * data[slicer1(0)]
                    + 48 * data[slicer1(1)]
                    - 36 * data[slicer1(2)]
                    + 16 * data[slicer1(3)]
                    - 3 * data[slicer1(4)]
                ) / (12 * dx)
                # Fourth order fully backward difference for last element
                # (3*f[-5]-16*f[-4]+36*f[-3]-48*f[-2]+25*f[-1])/(12*dx)
                ret_value[slicer1(-1)] = -(
                    -25 * data[slicer1(-1)]
                    + 48 * data[slicer1(-2)]
                    - 36 * data[slicer1(-3)]
                    + 16 * data[slicer1(-4)]
                    - 3 * data[slicer1(-5)]
                ) / (12 * dx)
                # Fourth order partially backward difference for the second to last
                # (-1*f[-4]+6*f[-3]-18*f[-3]+10*f[-2]+3*f[-1])/(12*dx)
                ret_value[slicer1(-2)] = -(
                    -3 * data[slicer1(-1)]
                    - 10 * data[slicer1(-2)]
                    + 18 * data[slicer1(-3)]
                    - 6 * data[slicer1(-4)]
                    + 1 * data[slicer1(-5)]
                ) / (12 * dx)
            else:
                raise NotImplementedError(
                    f"Accuracy order {accuracy_order} not implemented"
                )

            data = ret_value.copy()
        return type(self)(self.grid, ret_value)

    def gradient(self, order=1, accuracy_order=2):
        """Return a list :py:class:`~.UniformGridData` that are the numerical
        order-differentiation of the present grid_data along all the directions.
        (``order`` = number of derivatives, ie ``order=2`` is second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of ``self``.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int
        :param direction: Direction of the partial derivative.
        :type direction: int
        :param accuracy_order: Order of accuracy of the finite difference scheme.
        :type accuracy_order: int

        :returns:  list of :py:class:`~.UniformGridData` with partial derivative
                   along the directions.
        :rtype:    list of :py:class:`~.UniformGridData`

        """
        return [
            self.partial_differentiated(
                direction, order=order, accuracy_order=accuracy_order
            )
            for direction in range(self.num_dimensions)
        ]

    def partial_differentiate(self, dimension, order=1, accuracy_order=2):
        """Derive the data with numerical finite difference along a given direction
        (``order`` = number of derivatives, ie ``order=2`` is second derivative).

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of ``self``.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int
        :param direction: Direction of the partial derivative.
        :type direction: int
        :param accuracy_order: Order of accuracy of the finite difference scheme.
        :type accuracy_order: int

        """
        self._apply_to_self(
            self.partial_differentiated,
            dimension,
            order=order,
            accuracy_order=accuracy_order,
        )

    def _coordinates_at(self, where, absolute):
        """Return coordinates of a point in data as selected by the given function.

        :param where: Function that extract a location in the data. The function
                      has to return a tuple, identifying the point along each of
                      the dimensions.
        :type where: callable

        :param absolute: Whether to take the absolute value of the data.
        :type absolute: bool

        :returns: Coordinate of the point identified by ``where``.
        :rtype: 1D NumPy array

        """
        data = np.abs(self.data) if absolute else self.data
        index = np.unravel_index(where(data), data.shape)

        # coordinates is a list, with the linear coordinates along each
        # direction
        coordinates = self.coordinates_from_grid()

        # We loop over the coordinates and extract the element of position
        # "pos". We collect the results in a NumPy array.
        return np.array(
            [coordinates[dim][pos] for dim, pos in enumerate(index)]
        )

    def coordinates_at_maximum(self, absolute=True):
        """Return the point with maximum value.

        :returns:  Coordinate at where the value is maximum. If ``absolute``
                   is True, then the absolute value is first taken.
        :rtype:    1D NumPy array

        """
        return self._coordinates_at(np.argmax, absolute=absolute)

    def coordinates_at_minimum(self, absolute=True):
        """Return the point with minimum value.

        :returns:  Coordinate at where the value is minimum. If ``absolute``
                   is True, then the absolute value is first taken.
        :rtype:    1D NumPy array

        """
        return self._coordinates_at(np.argmin, absolute=absolute)

    def _apply_unary(self, function, *args, **kwargs):
        """Apply a unary function to the data.

        :param function: Unary function.
        :type function:  callable
        :returns: Function applied to the data.
        :rtype:    :py:class:`~.UniformGridData`

        """
        return type(self)(self.grid, function(self.data, *args, **kwargs))

    def _apply_reduction(self, reduction, *args, **kwargs):
        """Apply a reduction to the data.

        :param function: Function to apply to the data.
        :type function: callable

        :return: Reduction applied to the data.
        :rtype: float

        """
        return reduction(self.data, *args, **kwargs)

    def _apply_binary(self, other, function, *args, **kwargs):
        """This is an abstract function that is used to implement mathematical
        operations with other :py:class:`~.UniformGridData` (if they have the
        same grid) or scalars.

        _apply_binary takes another object that can be of the same type or a
        scalar, and applies function(self.data, other.data), performing type
        checking.

        :param other: Other object.
        :type other: :py:class:`~.UniformGridData` or scalar
        :param function: Dyadic function.
        :type function: callable

        :returns:  Return value of function when called with ``self`` and ``other``.
        :rtype:    :py:class:`~.UniformGridData`

        """
        # If the other object is of the same type
        if isinstance(other, type(self)):
            # Check the the coordinates are the same by checking shape, origin
            # and dx
            if not (
                all(self.grid.shape == other.grid.shape)
                and np.allclose(self.grid.x0, other.grid.x0, atol=1e-14)
                and np.allclose(self.grid.dx, other.grid.dx, atol=1e-14)
            ):
                raise ValueError("The objects do not have the same grid!")
            return type(self)(
                self.grid, function(self.data, other.data, *args, **kwargs)
            )

        # If it is a number
        if isinstance(other, (int, float, complex)):
            return type(self)(
                self.grid, function(self.data, other, *args, **kwargs)
            )

        # If it is a Tensor of type(self), we have to return a Tensor
        if isinstance(other, Tensor) and type(self) is other.type:
            # We keep this at the high level
            return type(other).from_shape_and_flat_data(
                other.shape,
                [
                    function(ot, self, *args, **kwargs)
                    for ot in other.flat_data
                ],
            )

        # If we are here, it is because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            np.ma.allclose(self.data, other.data, atol=1e-14)
            and self.grid == other.grid
        )

    # From Python's docs: In order to conform to the object model, classes that
    # define their own equality method should also define their own hash method,
    # or be unhashable.

    # We consider grid data unhashable, this object also has to be unhashable.
    __hash__ = None

    def fourier_transform(self):
        """Perform the multi-dimensional Fourier transform on the data.

        We follow NumPy's conventions, with the exception that we normalize the
        amplitude with ``dx``.

        If the signal is complex, we also shift the negative components to be in
        the negative part of the signal.

        :returns: Fourier transform.
        :rtype: :py:class:`~.UniformGridData`

        """
        if self.is_masked():
            raise RuntimeError(
                "Fourier transform with masked data is not supported."
            )

        fft_data = np.fft.fftshift(np.fft.fftn(self.data))
        # We extract the frequencies along each direction
        freqs = [
            np.fft.fftshift(np.fft.fftfreq(self.shape[dim], d=self.dx[dim]))
            for dim in range(self.num_dimensions)
        ]

        for dim in range(self.num_dimensions):
            fft_data[dim] *= self.dx[dim]

        lowest_freqs = [freqs[dim][0] for dim in range(self.num_dimensions)]
        delta_freqs = [
            freqs[dim][1] - freqs[dim][0] for dim in range(self.num_dimensions)
        ]

        grid = UniformGrid(
            fft_data.shape,
            x0=lowest_freqs,
            dx=delta_freqs,
            ref_level=self.ref_level,
            component=self.component,
            num_ghost=self.num_ghost,
            time=self.time,
            iteration=self.iteration,
        )

        return type(self)(grid, fft_data)

    def to_GridSeries(self):
        """Return a :py:class:`~.GridSeries` (if the data is one-dimensional).

        When the data is one dimensional, sometimes it is more convenient to
        treat it a series instead of grid data. This class is uses the same
        infrastructure as :py:class:`~.TimeSeries` and
        :py:class:`~.FrequencySeries` and has more or less the same features.

        :returns: Data as a Series.
        :rtype: :py:class:`~.GridSeries`
        """
        if self.num_dimensions != 1:
            raise ValueError("Only 1D data can be transformed into a Series")

        return GridSeries(self.coordinates_from_grid()[0], self.data)


class HierarchicalGridData(BaseNumerical):
    """Represents data defined on mesh-refined grids, consisting of one or more
    regular datasets with different grid spacings.

    All the arithmetic operations and binary operators are defined for this
    class, as well as interpolation and resampling.

    Upon initialization, we try to merge together all the components (output
    from different MPI processes), so there is one :py:class:`~.UniformGridData`
    per refinement level. In case of grids with more than one center of
    refinement, this is currently not possible, so we keep all the components
    around. In this, ghost zone information may be discarded.

    :ivar grid_data_dict: Mapping between refinement levels and components at
                          that refinement level.
    :type grid_data_dict: dict of :py:class:`~.UniformGridData`

    """

    def __init__(self, uniform_grid_data):
        """Constructor.

        Here we try to merge the different components, if we can.

        :param uniform_grid_data: List of regular datasets.
        :type uniform_grid_data:  list of :py:class:`~.UniformGridData`
        """
        if not hasattr(uniform_grid_data, "__len__"):
            raise TypeError(
                f"{type(self).__name__} is built with list "
                "of UniformGridData"
            )

        if len(uniform_grid_data) == 0:
            raise ValueError(f"Cannot create an empty {type(self).__name__}")

        if not all(isinstance(d, UniformGridData) for d in uniform_grid_data):
            raise TypeError(
                f"{type(self).__name__} requires a list of UniformGridData"
            )

        if len({d.num_dimensions for d in uniform_grid_data}) != 1:
            raise ValueError("Dimensionality mismatch")

        # Let's sort as increasing refinement level and component
        uniform_grid_data_sorted = sorted(
            uniform_grid_data, key=lambda x: (x.ref_level, x.component)
        )

        components = {}

        # Organize the components and create a copy. In creating a copy we
        # declare ownership of the UniformGridData
        for comp in uniform_grid_data_sorted:
            components.setdefault(comp.ref_level, []).append(comp.copy())

        self.grid_data_dict = {
            ref_level: self._try_merge_components(comps)
            for ref_level, comps in components.items()
        }

        # Map between coordinates and which component to use when computing
        # values. self._component_mapping is a function that takes a point and
        # returns the associated component. The reason this is an attribute is
        # to save it and avoid re-computing the mapping all the time
        self._component_mapping = None

    def _check_ref_factors(self):
        """Check if all the grids have dx that is an integer multiple of dx_finest.

        :returns: Whether dx/dx_finest if integer for all the components.
        :rtype: bool

        """
        if len(self.refinement_levels) == 1:
            return True

        def is_integer(x):
            return np.allclose(x - np.rint(comp_ref_factor), 0)

        # ref_factors is a dictionary that has as keys the number of refinement
        # level and as values a NumPy array with the refinement factor for that
        # level. We will loop over all the components and use this dictionary to
        # check if we have variable refinement factors across levels.
        ref_factors = {}

        # Next, we loop over all the components and check if their dx is an
        # integral multiple of dx_finest
        for comp in self.all_components:
            ref_level = comp.ref_level
            comp_ref_factor = comp.dx / self.finest_dx

            if not is_integer(comp_ref_factor):
                return False

            # On a given refinement level, the refinement factors have to be
            # the same because HierarchicalGridData works only with grids
            # with same dx at a given level. So, here can simply overwrite
            # the value that we have.
            ref_factors[ref_level] = comp_ref_factor

        # Now we loop over the refinement factors to check that they are always
        # the same. We compute what is the refinement factor that we expect
        # looking at the coarsest and finest levels, then we check that this
        # is indeed the one we observe on each level.
        #
        # Consider the case we have three levels.
        # Level 3 (finest) has dx = [2, 2]
        # Level 2 has dx = [6, 6]
        # Level 1 has dx = [18, 18]
        #
        # Here what we do is we take dx_coarse / dx_fine = [9, 9], and take the
        # power of 1/(3 - 1) = 1/2 = sqrt, so we obtain [3, 3]. This is how the
        # dx grows on each level.
        expected_base_ref_factor = (ref_factors[self.num_coarsest_level]) ** (
            1 / (self.num_finest_level - self.num_coarsest_level)
        )

        for ref_level, ref_factor in ref_factors.items():
            # Then, we find on each level what is the expected refinement factor
            # given the base one. For the previous example we would have:
            # Level 1: [3, 3] ** (3 - 1) = [9, 9]
            # Level 2: [3, 3] ** (3 - 2) = [3, 3]
            # Level 3: [3, 3] ** (3 - 3) = [1, 1]
            expected_ref_factor = expected_base_ref_factor ** (
                self.num_finest_level - ref_level
            )
            if not np.allclose(ref_factor, expected_ref_factor):
                return False

        # All integers and constant
        return True

    def _compute_component_mapping(self):
        """Scan the grid structure and prepare a map between points and components.

        :returns: Function that maps a point to the UniformGridData at highest
                  resolution that contains that point.
        :rtype: callable

        """

        # If refinement factors are not nice powers, then we cannot use this
        # method.
        if not self._check_ref_factors():
            return None

        # NOTE: (GB) This algorithm is from PostCactus (in
        # grid_data.BrickCoords). I don't understand it, but it works. All the
        # comments are mine, and maybe they don't make any sense.

        # We can do this operation only if the various grids have dx that is all
        # multiple of dx_finest. This is checked by_check_ref_factors.
        #
        # First, we define a dx that is half a minimum cell and we are going to
        # use the lowest component number among the highest refinement level as
        # our "coordinate system". We call this new coordinate system the
        # "tilde" coordinate system. In the tilde coordinate system, the all the
        # cells have integer coordinate.

        # Create local copies of some variables, so that we don't have to
        # compute them multiple times. We also sort in the opposite order, with
        # highest resolution first
        all_components = sorted(
            self.all_components,
            key=lambda comp: (-comp.ref_level, comp.component),
        )

        half_dx_finest = all_components[0].dx / 2.0
        origin = all_components[0].x0

        def to_tilde(x):
            """Convert a coordinate to tilde coordinate system (based on the first
            finest component).

            :param x: Coordinate.
            :type x: NumPy array
            :returns: Index in the coordinate system based on the first finest
                      component.
            :rtype: NumPy array

            """
            return np.rint((x - origin) / half_dx_finest)

        # Now we find where all the origins of the components fall in this new
        # coordinate system
        origins_tilde = np.array(
            [to_tilde(comp.grid.lowest_vertex) for comp in all_components]
        )
        corners_tilde = np.array(
            [to_tilde(comp.grid.highest_vertex) for comp in all_components]
        )

        # origins and corners tilde are NumPy arrays with N D-dimensional
        # elements. N is the number of components, D is the dimensionality of
        # the grid.

        # Next, we group together all the origins and consider only the unique
        # points along each direction. unique_origins_tilde
        # (unique_corners_tilde) is a D-dimensional list
        unique_origins_tilde = [
            set(orig) for orig in np.transpose(origins_tilde)
        ]
        unique_corners_tilde = [
            set(corn) for corn in np.transpose(corners_tilde)
        ]

        # Now we collect all the boundaries (origins and corners)
        boundaries_tilde = [
            sorted(orig.union(corn))
            for orig, corn in zip(unique_origins_tilde, unique_corners_tilde)
        ]

        def x_tilde_to_component(x_tilde):
            """Given a point in the tilde coordinates, return the indices of
            boundaries_tilde that contain that point

            :returns: Multi-index (one index for each direction) that identifies
                      the block in ``boundaries_tilde`` that contains the
                      point.
            :rtype: tuple
            """
            return tuple(
                bisect_right(boundary_dim, x_tilde_dim) - 1
                for boundary_dim, x_tilde_dim in zip(boundaries_tilde, x_tilde)
            )

        # Now we go back and find where all the origins and corners are
        boundary_origins_tilde = [
            x_tilde_to_component(orig) for orig in origins_tilde
        ]
        boundary_corners_tilde = [
            x_tilde_to_component(corn) for corn in corners_tilde
        ]

        # And the last step is to find what is the finest refinement level
        # available on each block

        finest_map = np.zeros(
            np.array([len(x) - 1 for x in boundaries_tilde]), dtype=int
        )

        for component_index in range(len(boundary_origins_tilde) - 1, -1, -1):
            slicer = [
                slice(orig, corn)
                for orig, corn in zip(
                    boundary_origins_tilde[component_index],
                    boundary_corners_tilde[component_index],
                )
            ]
            finest_map[tuple(slicer)] = component_index

        def get_component(coordinate):
            """Map a coordinate to the component that contains it."""

            # This is like to_tilde, but we floor instead of rounding.
            # God knows why.
            x_tilde = np.floor((coordinate - origin) / half_dx_finest)

            if any(
                ((point_dim < border_dim[0]) or (point_dim >= border_dim[-1]))
                for point_dim, border_dim in zip(x_tilde, boundaries_tilde)
            ):
                raise ValueError(f"{coordinate} outside the grid")

            # The list comprehension is over the various dimensions
            component_index = tuple(
                bisect_right(boundary_dim, x_tilde_dim) - 1
                for boundary_dim, x_tilde_dim in zip(boundaries_tilde, x_tilde)
            )

            return all_components[finest_map[component_index]]

        return get_component

    @staticmethod
    def _fill_grid_with_components(grid, components):
        """Given a grid, try to fill it with the components, return eturning a
        :py:class:`~.UniformGridData` and the indices that actually were used in
        filling the grid.

        This happens by iterating over the components and copying data to the
        output grid, recording what points were filled. We also return the indices
        of the points that were filled.

        :param grid: Grid to fill.
        :type grid: :py:class:`~.UniformGrid`
        :param components: Components to fill the grid.
        :type components: list of :py:class:`~.UniformGridData`

        :returns: Merged components and indices used to merge the components.
        :rtype: tuple of :py:class:`~.UniformGridData` and numpy array

        """

        # For filling the data, we prepare the array first, and we fill it with
        # the single components. We fill a second array which we use to keep
        # track of what indices have been filled with the input data.
        data = np.zeros(grid.shape, dtype=components[0].data.dtype)
        indices_used = np.zeros(grid.shape, dtype=components[0].data.dtype)

        if any(
            isinstance(comp.data, np.ma.MaskedArray) for comp in components
        ):
            data = np.ma.MaskedArray(data)

        for comp in components:
            # We find the index corresponding to x0 and x1 of the component
            index_x0 = ((comp.x0 - grid.x0) / grid.dx + 0.5).astype(int)
            index_x1 = index_x0 + comp.shape
            slicer = tuple(
                slice(index_j0, index_j1)
                for index_j0, index_j1 in zip(index_x0, index_x1)
            )
            data[slicer] = comp.data
            indices_used[slicer] = np.ones(comp.data.shape)

        return UniformGridData(grid, data), indices_used

    def _try_merge_components(self, components):
        """Try to merge a list of :py:class:`~.UniformGridData` instances into one,
        assuming they all have the same grid spacing and filling a regular grid
        completely.

        If the assumption is not verified, and some blank spaces are found, then
        it returns the input untouched. This happens in the case that there are
        multiple refinement centers, or if there are missing components.

        This function always returns a list, even when the components are merged.
        In that case, the return value is a ``[merged]``, where ``merged`` is a
        :py:class:`~.UniformGridData`.

        :param components: List of components.
        :type components: list of :py:class:`~.UniformGridData`

        :returns: List of components, or list with one single element, the merged
                  components.
        :rtype: list of :py:class:`~.UniformGridData`

        """

        if len(components) == 1:
            return [components[0].ghost_zones_removed()]

        # TODO: Instead of throwing away the ghost zones, we should check them

        # We remove all the ghost zones so that we can arrange all the grids
        # one next to the other without having to worry about the overlapping
        # regions
        components_no_ghosts = [
            comp.ghost_zones_removed() for comp in components
        ]

        # For convenience, we also order the components from the one with the
        # smallest x0 to the largest, so that we can easily find the
        # coordinates.
        #
        # We have to transform x.x0 in tuple because we cannot compare NumPy
        # arrays directly for sorting.
        components_no_ghosts.sort(key=lambda x: tuple(x.x0))

        # Next, we prepare the global grid
        grid = gdu.merge_uniform_grids(
            [comp.grid for comp in components_no_ghosts]
        )

        merged_grid_data, indices_used = self._fill_grid_with_components(
            grid, components_no_ghosts
        )

        if np.amin(indices_used) == 1:
            return [merged_grid_data]

        return components

    def __getitem__(self, key):
        """Return the list of components at the given refinement level.

        You can also consider using :py:meth:`~.get_level`, which returns a
        single :py:class:`~.UniformGridData` if there's only one component at
        that level (otherwise error).

        :param key: Refinement level.
        :type key: int

        :returns: List of components at a given refinement level.
        :rvalue: list of :py:class:`~.UniformGridData`

        """
        return self.grid_data_dict[key]

    def get_level(self, ref_level):
        """Return the data at a given refinement level.

        :param ref_level: Number of refinement level.
        :type ref_level: int

        :returns: Data at given refinement level.
        :rtype: :py:class:`~.UniformGridData`
        """
        if ref_level not in self.refinement_levels:
            raise ValueError(f"Level {ref_level} not available")
        if len(self[ref_level]) > 1:
            raise ValueError(
                f"Level {ref_level} has multiple patches"
                " get_level works only when there is one"
            )
        return self[ref_level][0]

    def iter_from_finest(self):
        """Iterator over the components, sorted by refinement level, from the finest to
        the coarsest.

        :returns: Refinement level number, component index, and data.
        :rtype: generator of tuples (int, int, :py:class:`~.UniformGridData`)
        """
        # TODO (FUTURE): Reverse dictionary in Python 3.8
        #
        # In Python 3.8 we can reverse without transforming into a list first
        for ref_level, data in reversed(list(self.grid_data_dict.items())):
            for comp_index, comp in enumerate(data):
                yield ref_level, comp_index, comp

    def __iter__(self):
        """Iterate across all the refinement levels and components from the coarsest
        to the finest.

        :returns: Refinement level number, component index, and data.
        :rtype: tuple (int, int, :py:class:`~.UniformGridData`)
        """
        for ref_level, data in self.grid_data_dict.items():
            for comp_index, comp in enumerate(data):
                yield ref_level, comp_index, comp

    def __len__(self):
        return len(self.refinement_levels)

    @property
    def refinement_levels(self):
        """Return a list with the refinement levels available.

        :returns: List of refinement levels available.
        :rtype: list of ints
        """
        return list(self.grid_data_dict.keys())

    @property
    def all_components(self):
        """Return a list with all the components.

        This is useful to create a new :py:class:`~.HierarchicalGridData`
        from ``self``.

        :returns: List of all the components.
        :rtype: list of :py:class:`~.UniformGridData`

        """
        # TODO: (PERFORMANCE) Optimize method called often
        #
        # This method is used every time we loop over all the components. Hence,
        # it is called by several other methods. It is important to optimize it
        # further to improve overall performance.

        all_components = []
        for comps in self.grid_data_dict.values():
            all_components.extend(comps)
        return all_components

    @property
    def num_finest_level(self):
        """Return the number of the finest refinement level.

        :returns: Index of the finest level.
        :rtype: int
        """
        return self.refinement_levels[-1]

    @property
    def finest_level(self):
        """Return the finest level, if it is a single grid.

        :returns: Finest level.
        :rtype: :py:class:`~UniformGridData`
        """
        return self.get_level(self.num_finest_level)

    @property
    def max_refinement_level(self):
        """Return the number of the finest refinement level.

        Alias for :py:meth:`~.num_finest_level`.

        :returns: Index of the finest level.
        :rtype: int
        """
        return self.num_finest_level

    @property
    def num_coarsest_level(self):
        """Return the number of the coarsest refinement level.

        :returns: Index of the coarsest level.
        :rtype: int
        """
        return self.refinement_levels[0]

    @property
    def coarsest_level(self):
        """Return the coarsest level, if it is a single grid.

        :returns: Coarsest level.
        :rtype: :py:class:`~UniformGridData`
        """
        return self.get_level(self.num_coarsest_level)

    @property
    def first_component(self):
        """Return the first component of the coarsest refinement level.

        :returns: First component of the coarsest level.
        :rtype: `:py:class:~UniformGridData`
        """
        return self[self.num_coarsest_level][0]

    @property
    def _a_component(self):
        """Return the a component of the hierarchy.

        It is useful to understand various properties that are shared
        by all the components, e.g. the dtype.

        Using this method is faster than :py:func:`~.first_component`.

        :returns: First component of the coarsest level.
        :rtype: `:py:class:~UniformGridData`

        """
        return next(iter(self.grid_data_dict.values()))[0]

    @property
    def dtype(self):
        return self._a_component.dtype

    @property
    def shape(self):
        """Return the number of components per each refinement level.

        For example, if data has three levels, with 1 component in the first, 2
        in the second, and three in the fifth, shape will be {1: 1, 2: 2, 5: 3}

        This method is useful for quick high level comparison between two
        :py:class:`~.HierachicalGridData`.

        :returns: Dictionary with keys the refinement level numbers and values the
                  number of components at that level.
        :rtype: dictionary

        """
        return {
            ref_level: len(comp)
            for ref_level, comp in self.grid_data_dict.items()
        }

    @property
    def x0(self):
        """Origin of the coarsest grid, if it is a single component.

        :returns: Origin of the coarsest grid, if it is a single component.
        :rtype: 1d NumPy array
        """
        # We have multiple patches
        if len(self[self.num_coarsest_level]) != 1:
            raise ValueError(
                "Data does not have a well defined x0"
                " (there are multiple patches)"
            )
        return self.first_component.x0

    @property
    def x1(self):
        """Corner of the coarsest grid, if it is a single component.

        :returns: Corner of the coarsest grid, if it is a single component.
        :rtype: 1d NumPy array
        """
        # We have multiple patches
        if len(self[self.num_coarsest_level]) != 1:
            raise ValueError(
                "Data does not have a well defined x1"
                " (there are multiple patches)"
            )
        return self.first_component.x1

    def dx_at_level(self, level):
        """Return the grid spacing at the specified refinement level.

        :param level: Refinement level number.
        :type level: int
        :returns: Spacing at the given refinement level.
        :rtype: 1d NumPy array
        """
        return self[level][0].dx

    @property
    def coarsest_dx(self):
        """Return the grid spacing of the coarsest level.

        :returns:  Grid spacing of the coarsest level.
        :rtype:   1d NumPy array
        """
        return self.dx_at_level(self.num_coarsest_level)

    @property
    def finest_dx(self):
        """Return the grid spacing of the finest level.

        :returns:  Grid spacing of the finest level.
        :rtype:   1d NumPy array
        """
        return self.dx_at_level(self.num_finest_level)

    @property
    def num_dimensions(self):
        """Return the number of dimensions.

        :returns:  Number of dimensions.
        :rtype:   int
        """
        return self._a_component.num_dimensions

    @property
    def num_extended_dimensions(self):
        """Return the number of dimensions with more than one cell.

        :returns:  Number of dimensions with more than one gridpoint.
        :rtype:   int
        """
        return self._a_component.num_extended_dimensions

    @property
    def time(self):
        """The time of the coarsest refinement level.

        :returns:  Time of the coarsest refinement level.
        :rtype:   float
        """
        return self.first_component.time

    @property
    def iteration(self):
        """The iteration of the coarsest refinement level.

        :returns:  Iteration number of the coarsest refinement level.
        :rtype:   int
        """
        return self.first_component.iteration

    def is_complex(self):
        """Return whether the data is complex.

        :returns:  True if the data is complex, false if it is not.
        :rtype:   bool

        """
        return any(comp.is_complex() for comp in self.all_components)

    @property
    def mask(self):
        """Return where the data is valid (according to the mask).

        :returns: List of arrays of True/False, one per component
                  in the same order as :py:meth:`~.all_components`.
        :rtype: list of arrays of bool
        """
        return [comp.mask for comp in self.all_components]

    def is_masked(self):
        """Return whether the data is masked.

        :returns:  True if the data is masked, false if it is not.
        :rtype:   bool

        """
        return any(comp.is_masked() for comp in self.all_components)

    def copy(self):
        """Return a deep copy.

        :returns:  Deep copy of the :py:class:`~.HierarchicalGridData`.
        :rtype:    :py:class:`~.HierarchicalGridData`
        """
        return type(self)(self.all_components)

    def mask_applied(self, mask, ignore_existing=False):
        """Return a new grid data with given mask applied to the data.

        If a previous mask already exists, the new mask will be added on top,
        unless ``ignore_existing`` is True.

        :param mask: List of arrays of booleans (one per component) that identify
                     where the data is invalid.
                     This can be obtained with the method :py:meth:`~.mask`.
        :type mask: list of NumPy array

        :param ignore_existing: If True, overwrite any previously existing mask.
        :type ignore_existing: bool

        :returns: New grid data with mask applied.
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        return type(self)(
            [
                comp.mask_applied(mask_comp, ignore_existing=ignore_existing)
                for comp, mask_comp in zip(self.all_components, mask)
            ]
        )

    def mask_apply(self, mask, ignore_existing=False):
        """Apply given mask.

        If a previous mask already exists, the new mask will be added on top,
        unless ``ignore_existing`` is True.

        :param mask: List of arrays of booleans (one per component) that identify
                     where the data is invalid.
                     This can be obtained with the method :py:meth:`~.mask`.
        :type mask: list of NumPy array

        :param ignore_existing: If True, overwrite any previously existing mask.
        :type ignore_existing: bool

        """
        self._apply_to_self(
            self.mask_applied, mask, ignore_existing=ignore_existing
        )

    def __eq__(self, other):
        """Check for equality."""
        if (
            not isinstance(other, HierarchicalGridData)
            or self.shape != other.shape
        ):
            return False

        return self.all_components == other.all_components

    # From Python's docs: In order to conform to the object model, classes that
    # define their own equality method should also define their own hash method,
    # or be unhashable.

    # We consider grid data unhashable, this object also has to be unhashable.
    __hash__ = None

    def _finest_component_at_point_mapping(self, coordinate):
        """Return the component of the most refined level that contains the given
        coordinate assuming a valid input coordinate using the component mapping.

        This routine works only with grids in which the spacings are
        commensurable.

        :param coordinate: Point.
        :type coordinate: tuple or NumPy array with the same dimension

        :returns: Component with highest resolution at point
        :rtype: :py:class:`~.UniformGridData`

        """
        return self._component_mapping(coordinate)

    def _finest_component_at_point_general(self, coordinate):
        """Return the component of the most refined level that contains the given
        coordinate assuming a valid input coordinate.

        This routine works with all the grids.

        :param coordinate: Point.
        :type coordinate: tuple or NumPy array with the same dimension

        :returns: Component with highest resolution at point
        :rtype: :py:class:`~.UniformGridData`

        """
        # We walk from the finest level to the coarsest. If we find the point,
        # re return it. If we find nothing, we raise error.
        for ref_level, comp, grid_data in self.iter_from_finest():
            if coordinate in grid_data.grid:
                return self[ref_level][comp]

        raise ValueError(f"{coordinate} outside the grid")

    def finest_component_at_point(self, coordinate, no_checks=False):
        """Return the number and the component index of the most
        refined level that contains the given coordinate.

        :param coordinate: Point.
        :type coordinate: tuple or NumPy array with the same dimension
        :param no_checks: Do not perform sanity checks on the input (for
                          speed).
        :type no_checks: bool

        :returns: Component with highest resolution at point
        :rtype: :py:class:`~.UniformGridData`
        """
        if not no_checks:
            if not hasattr(coordinate, "__len__"):
                raise TypeError(f"{coordinate} is not a valid point")

            if len(coordinate) != self.num_dimensions:
                raise ValueError(
                    f"The input point has dimension {len(coordinate)}"
                    f" but the data has dimension {self.num_dimensions}"
                )

        # If we don't have self._component_mapping, we should try to compute it.
        # If we get back None, it means that it cannot be computed for this
        # grid. Here it is the perfectly place where to use the walrus operator.
        if self._component_mapping is None:
            self._component_mapping = self._compute_component_mapping()

        if self._component_mapping is not None:
            finder = self._finest_component_at_point_mapping
        else:
            finder = self._finest_component_at_point_general

        # finder is the function that returns the component given the coordinate
        return finder(coordinate)

    def evaluate_with_spline(self, x, ext=2, piecewise_constant=False):
        """Evaluate the spline on the points ``x``.

        Values outside the interval are set to 0 if ext=1, or a ``ValueError``
        is raised if ``ext=2``.

        This method is meant to be used only if you want to use a different
        ``ext`` for a specific call, otherwise, just use __call__.

        :param x: Points where to evaluate the data.
        :type x: 1D NumPy array of float, or :py:class:`~.UniformGrid`

        :param ext: How to deal values outside the bounaries. Values outside
                    the interval are set to 0 if ``ext=1``,
                    or an error is raised if ``ext=2``.
        :type ext:  int

        :returns: Values of the data evaluated on the input ``x``.
        :rtype:   1D NumPy array or float

        """
        if not piecewise_constant and self.is_masked():
            raise RuntimeError("Splines with masked data are not supported.")

        if isinstance(x, UniformGrid):
            # The way we want the coordinates is like as an array with the same
            # shape of the grid and with values the coordinates (as arrays). This
            # is similar to as_same_shape, but the coordinates have to be the
            # value, and not the first index.
            x = np.moveaxis(x.coordinates(as_same_shape=True), 0, -1)

        # We flatten the array (up to the last dimension) and we save the
        # original shape, because we are going to reshape it at the end.
        points_arr = np.asarray(x)
        original_shape = points_arr.shape
        points_arr = points_arr.reshape(-1, points_arr.shape[-1])

        # NOTE: The following algorithm is not the fastest but it doesn't matter
        #       too much because UniformGridData.evaluate_with_spline dominates
        #       the execution time.
        #
        #       Note also that is it tested by testing finest_component_at_point
        #       (and not directly)

        # Next, we organize points depending on the component/refinement level
        # they belong.
        #
        # level_comps is a dictionary with keys the refinement levels and
        # components for which we have to compute points and for values a list
        # with the index of the points in points_arr. We need the indices because
        # we need to put back the values where they were, since we are going to
        # take bit and pieces of the array.
        #
        # Then, we have another mapping level_comps_data that has as keys the
        # same keys as level_comps, but values the actual component
        # (UniformGridData) that has to be used for the calculation
        level_comps = {}
        level_comps_data = {}
        for index, point in enumerate(points_arr):
            data = self.finest_component_at_point(point, no_checks=True)
            ref_level, comp = data.ref_level, data.component
            level_comps.setdefault((ref_level, comp), []).append(index)
            level_comps_data.setdefault((ref_level, comp), data)

        # Now, we can evaluate the points using the methods of UniformGridData.
        # We collect all results in a new array that is initially full of zeros
        ret = np.zeros(len(points_arr), dtype=self.dtype)
        for (ref_level, comp), points_indices in level_comps.items():
            points = points_arr[level_comps[ref_level, comp]]
            evaluated_points = level_comps_data[
                (ref_level, comp)
            ].evaluate_with_spline(
                points, ext=ext, piecewise_constant=piecewise_constant
            )
            ret[points_indices] = evaluated_points

        # Finally, we have to reshape the array to the correct form.
        ret = ret.reshape(original_shape[:-1])
        return ret

    def __call__(self, x):
        return self.evaluate_with_spline(x)

    def to_UniformGridData_from_grid(self, grid, resample=False):
        """Combine the refinement levels into a :py:class:`~.UniformGridData`
        on the specified :py:class:`~.UniformGrid`.

        If ``resample`` is True, the data is resampled with multilinear
        interpolation.

        :param grid: Grid onto which to resample the data.
        :type grid: :py:class:`~.UniformGrid`.
        :param resample: If True, resample the data with multilinear interpolation,
                         otherwise, use nearest neighbors.
        :type resample: bool

        """
        return UniformGridData(
            grid,
            self.evaluate_with_spline(grid, piecewise_constant=(not resample)),
        )

    def to_UniformGridData(
        self, shape, x0, x1=None, dx=None, resample=False, **kwargs
    ):
        """Combine the refinement levels into a :py:class:`~.UniformGridData` specified
        by the given ``shape``, ``x0``, and ``dx`` or ``x1``.

        Additional arguments are sent to the constructor of
        :py:class:`~.UniformGrid`.

        If ``resample`` is True, the data is resampled with multilinear
        interpolation.

        :param shape: Number of points across all the dimensions.
        :type shape: 1d NumPy array
        :param x0: Origin.
        :type x0: 1d NumPy array, or None
        :param x1: Grid corner. If None, it will be inferred.
        :type x1:  1d NumPy array, or None
        :param dx: Grid spacing. If None, it will be inferred.
        :type dx: 1d NumPy array, or None
        :param resample: If True, resample the data with multilinear interpolation,
                         otherwise, use nearest neighbors.
        :type resample: bool

        """
        grid = UniformGrid(shape, x0, x1, dx, **kwargs)

        return self.to_UniformGridData_from_grid(grid, resample=resample)

    def refinement_levels_merged(self, resample=False):
        """Return a new :py:class:`~.UniformGridData` with all the available
        data combined and resampled to a grid that encompasses all the
        components and has resolution of the finest refinement level.

        When ``resample`` is True, data from coarser refinement levels is
        resampled with multilinear interpolation, otherwise the nearest
        neighbors are used.

        .. warning::

            For most practical purposes, using this function is an overkill.
            This can be a very expensive operation and require a lot of memory.
            Prefer :py:meth:`to_UniformGridData` when possible. The only real
            reasonable application of this function is with small simluations or
            1D data.

        :param resample: If True, resample the data with multilinear interpolation,
                         otherwise, use nearest neighbors.
        :type resample: bool

        :returns: New :py:class:`~.UniformGridData` with the resolution of the
                  finest refinement level.
        :rtype: :py:class:`~.UniformGridData`

        """
        # If we have only one refinement level, with one component, we should
        # just return that.
        if len(self.all_components) == 1:
            return self.first_component.copy()

        # finest_dx can have zero entries, for which a shape of 1 should
        # correspond. There can zero entries, we substitute them with -1, so
        # that we can identify them as negative numbers
        new_dx = np.array([dx if dx > 0 else -1 for dx in self.finest_dx])
        new_shape = ((self.x1 - self.x0) / new_dx + 1.5).astype(int)
        new_shape = np.array([s if s > 0 else 1 for s in new_shape])

        return self.to_UniformGridData(
            new_shape,
            self.x0,
            x1=None,
            dx=new_dx,
            time=self.time,
            iteration=self.iteration,
            resample=resample,
        )

    def merge_refinement_levels(self, resample=False):
        """DEPRECATED."""
        warnings.warn(
            "merge_refinement_levels was renamed to refinement_levels_merged "
            "and it will be removed in kuibit 1.5.0",
            category=FutureWarning,
        )
        return self.refinement_levels_merged(resample=resample)

    def _apply_to_self(self, f, *args, **kwargs):
        """Apply the method ``f`` to ``self``, modifying ``self``.
        This is used to transform the commands from returning an object
        to modifying ``self``.
        The function has to return a new copy of the object (not a reference).

        :param f: Function to apply to ``self``.
        :type f:  callable
        """
        ret = f(*args, **kwargs)
        self.grid_data_dict = ret.grid_data_dict
        # We need to invalidate the _component_mapping (it may have changed)
        self._component_mapping = None

    def _apply_binary(self, other, function, *args, **kwargs):
        """Apply a binary function to the data of ``self`` and ``other``.

        :param function: Function to apply to all the data in the various
        refinement levels.
        :type function: callable

        :return: New :py:class:`~.HierarchicalGridData` with function applied to
        ``self.data`` and ``other.data``.
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        # We only know what how to combine HierarchicalGridData
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Grid structure incompatible")
            new_data = [
                function(data_self, data_other, *args, **kwargs)
                for data_self, data_other in zip(
                    self.all_components, other.all_components
                )
            ]
            return type(self)(new_data)

        if isinstance(other, (int, float, complex)):
            new_data = [
                function(data_self, other, *args, **kwargs)
                for data_self in self.all_components
            ]
            return type(self)(new_data)

        # If it is a Tensor of type(self), we have to return a Tensor
        if isinstance(other, Tensor) and type(self) is other.type:
            # We keep this at the high level
            return type(other).from_shape_and_flat_data(
                other.shape,
                [
                    function(ot, self, *args, **kwargs)
                    for ot in other.flat_data
                ],
            )

        # If we are here, it is because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")

    def _apply_reduction(self, reduction, *args, **kwargs):
        """Apply a reduction to the data.

        :param function: Reduction to apply to all the data in the various
        refinement levels
        :type function: callable

        :return: Output of the reduction on the data.
        :rtype: return type of ``reduction``

        """
        # Assume reduction is np.min, we want the real minimum, so we have to
        # take the reduction of the reduction
        return reduction(
            # Here we are accessing _apply_reduction, which is a protected
            # member, so we ignore potential complaints.
            np.array(
                [
                    # skipcq: PYL-W0212
                    data._apply_reduction(reduction, *args, **kwargs)
                    for data in self.all_components
                ]
            ),
            *args,
            **kwargs,
        )

    def _apply_unary(self, function, *args, **kwargs):
        """Apply a unary function to the data.

        :param function: Function to apply to all the data in the various
        refinement levels
        :type function: callable

        :return: New :py:class:`~.HierarchicalGridData` with function applied to
        the data.
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        # Here we are accessing _apply_unary, which is a protected member, so
        # we ignore potential complaints.
        new_data = [
            # skipcq: PYL-W0212
            data._apply_unary(function, *args, **kwargs)
            for data in self.all_components
        ]
        return type(self)(new_data)

    def _call_component_method(
        self, method_name, *args, method_returns_list=False, **kwargs
    ):
        """Call a method on each component and return the result as a
        :py:class:`~.HierarchicalGridData`.

        :param method_name: a string that identifies one of the methods in
        :py:class:`~.UniformGridData`.
        :type method_name: str

        :param method_returns_list: If True, the method is expected to return a
                                    list, one :py:class:`~.UniformGridData` per
                                    dimension (e.g,
                                    :py:meth:`HierarchicalGridData.gradient`,
                                    :py:meth:`HierarchicalGridData.coordinates`).
        :type method_returns_list: bool

        :return: New :py:class:`~.HierarchicalGridData` with function applied to the data
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        if not isinstance(method_name, str):
            raise TypeError(
                f"method_name has to be a string (but it is {method_name})"
            )

        if not hasattr(self._a_component, method_name):
            raise ValueError(
                f"UniformGridData does not have a method with name {method_name}"
            )

        # Here we get the method as a function with getattr(data, method_name),
        # then we apply this function with arguments *args and **kwargs
        new_data = [
            getattr(data, method_name)(*args, **kwargs)
            for data in self.all_components
        ]
        # There are two possibilities: new data is a list of UniformGridData
        # (when method_returns_list is False), alternatively it is a list of
        # lists of UniformGridData

        # First, the case in which the method returns a UniformGridData (and not
        # a list of UniformGridData)
        if not method_returns_list:
            return type(self)(new_data)

        # Second, we have a list of UniformGridData
        return [
            type(self)([data[dim] for data in new_data])
            for dim in range(self.num_dimensions)
        ]

    def partial_differentiated(self, direction, order=1, accuracy_order=2):
        """Return a :py:class:`~.HierarchicalGridData` that is the numerical
        order-differentiation of the present grid_data along a given direction.
        (order = number of derivatives, ie ``order=2`` is second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of ``self``.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int
        :param direction: Direction of the partial derivative.
        :type direction: int
        :param accuracy_order: Order of accuracy of the finite difference scheme.
        :type accuracy_order: int

        :returns:  New :py:class:`~.HierarchicalGridData` with derivative.
        :rtype:    :py:class:`~.HierarchicalGridData`

        """
        return self._call_component_method(
            "partial_differentiated",
            direction,
            order=order,
            accuracy_order=accuracy_order,
        )

    def gradient(self, order=1, accuracy_order=2):
        """Return a list :py:class:`~.HierarchicalGridData` that are the numerical
        order-differentiation of the present grid_data along all the directions.
        (order = number of derivatives, ie ``order=2`` is second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of self.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int
        :param accuracy_order: Order of accuracy of the finite difference scheme.
        :type accuracy_order: int
        :returns: list of :py:class:`~.HierarchicalGridData` with partial
                  derivative along all the directions.
        :rtype:  list of :py:class:`~.HierarchicalGridData`

        """
        return self._call_component_method(
            "gradient",
            method_returns_list=True,
            order=order,
            accuracy_order=accuracy_order,
        )

    def partial_differentiate(self, direction, order=1, accuracy_order=2):
        """Apply a numerical differentiatin along the specified direction.

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of ``self``.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int
        :param direction: Direction of the partial derivative.
        :type direction: int
        :param accuracy_order: Order of accuracy of the finite difference scheme.
        :type accuracy_order: int

        :returns: Derivative along the specified direction.
        :rtype: list of :py:class:`~.HierarchicalGridData`

        """
        return self._apply_to_self(
            self.partial_differentiated,
            direction,
            order=order,
            accuracy_order=accuracy_order,
        )

    def ghost_zones_removed(self):
        """Return a new :py:class:`~.HierarchicalGridData` with all the ghost zones removed.

        :returns: New :py:class:`~.HierarchicalGridData` without ghostzones.
        :rtype: :py:class:`HierarchicalGridData`
        """
        return self._call_component_method("ghost_zones_removed")

    def ghost_zones_remove(self):
        """Remove all the ghost zones."""
        self._apply_to_self(self.ghost_zones_removed)

    def sliced(self, cut, resample=False):
        """Return a new :py:class:`~.HierarchicalGridData` obtained slicing the current one.

        ``cut`` specifies how to slice the data. It has to be an array with the
        same number of dimensions of the data. In the entries where ``cut`` is
        None, that dimension is kept, where it is a number, the data is cut
        fixing that coordinate. For example, for a 2D array, if ``cut`` is
        ``[None, 2]``, the cut will be with ``y = 2``.

        If ``resample`` is True, you can cut at any point and we will compute
        the values with multilinear interpolation. If ``resample`` is False, we
        will use the data already available.

        In doing this, dimensions that are only one grid point are lost.

        :param cut: How to slice the array. None entries mean "keep that dimension".
        :type cut:  array or list with dimension
        :param resample: Whether to use multilinear interpolation to compute the
                         data or simply use the value of the closest point.
        :type resample: bool

        :returns: A sliced :py:class:`~.HierachicalGridData`.
        :rtype: :py:class:`~.HierachicalGridData`

        """
        # We can use _call_component_method here because we have to handle the
        # errors

        # A HierarchicalGridData can be formed by multiple components (e.g., one
        # for each MPI rank). When we slice it, some components do not
        # contribute at all to the result. For example, if we ask for the xy
        # plane and a component has zmin = 3, the component should be excluded.
        # The slice method raises an error when the cut is outside the grid, here
        # we capture those errors and ignore the components that raised them.
        new_data = []

        for data in self.all_components:
            try:
                new_data.append(data.sliced(cut, resample=resample))
            except ValueError as e:
                if str(e) != "Cut point is outside the grid":
                    raise
                # Otherwise, do nothing
                pass  # Ignore the component

        if len(new_data) == 0:
            raise ValueError("Cut point is outside the grid")

        return type(self)(new_data)

    def slice(self, cut, resample=False):
        """Slice the data along given direction.

        ``cut`` specifies how to slice the data. It has to be an array with the
        same number of dimensions of the data. In the entries where ``cut`` is
        None, that dimension is kept, where it is a number, the data is cut
        fixing that coordinate. For example, for a 2D array, if ``cut`` is
        ``[None, 2]``, the cut will be with ``y = 2``.

        If ``resample`` is True, you can cut at any point and we will compute
        the values with multilinear interpolation. If ``resample`` is False, we
        will use the data already available.

        In doing this, dimensions that are only one grid point are lost.

        :param cut: How to slice the array. None entries mean "keep that dimension".
        :type cut:  array or list with dimension
        :param resample: Whether to use multilinear interpolation to compute the
                         data or simply use the value of the closest point.
        :type resample: bool

        """
        self._apply_to_self(self.sliced, cut=cut, resample=resample)

    def coordinates(self):
        """Return coordinates as a list of :py:class:`~.HierarchicalGridData`.

        Useful for computations involving coordinates.

        :returns: Coordinates.
        :rtype: list of :py:class:`~.HierarchicalGridData`
        """
        return self._call_component_method(
            "coordinates", method_returns_list=True
        )

    def coordinates_at_maximum(self, absolute=True):
        """Return the point with maximum value.

        :returns:  Coordinate at where the value is maximum. If ``absolute``
                   is True, then the absolute value is first taken.
        :rtype:    1D NumPy array

        """
        comps = self.all_components

        # We extract the maximum on each component, and find the maximum of the
        # maxima
        maxima = [
            np.max(np.abs(comp.data) if absolute else np.max(comp.data))
            for comp in comps
        ]

        comp_max = np.argmax(maxima)
        return comps[comp_max].coordinates_at_maximum(absolute=absolute)

    def coordinates_at_minimum(self, absolute=True):
        """Return the point with minimum value.

        :returns:  Coordinate at where the value is minimum. If ``absolute``
                   is True, then the absolute value is first taken.
        :rtype:    1D NumPy array

        """
        comps = self.all_components

        # We extract the minimum on each component, and find the minimum of the
        # minima
        minima = [
            np.min(np.abs(comp.data) if absolute else np.min(comp.data))
            for comp in comps
        ]

        comp_min = np.argmin(minima)
        return comps[comp_min].coordinates_at_minimum(absolute=absolute)

    def __str__(self):
        ret = "Available refinement levels (components):\n"
        for ref_level in self.refinement_levels:
            ret += f"{ref_level} ({len(self[ref_level])})\n"
        ret += f"Spacing at coarsest level ({self.num_coarsest_level}): "
        ret += f"{self.coarsest_dx}\n"
        ret += f"Spacing at finest level ({self.num_finest_level}): {self.finest_dx}"
        return ret
