#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
#
# Inspired by code originally developed by Wolfgang Kastaun. See, GitHub,
# wokast/PyCactus/PostCactus/grid_data.py
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

"""The :py:mod:`~.grid_data` module provides representations of data on
uniform grids as well as for data on refined grid hierarchies. Standard
arithmetic operations are supported for those data grids, further methods
to interpolate and resample. The number of dimensions is arbitrary.

The important classes defined here are
- :py:class:`~.UniformGrid` represents the geometry of a uniform Cartesian
cell-centered grid.
- :py:class:`~.UniformGridData` represents data on a uniform grid.
- :py:class:`~.HierarchicalGridData` represents data on a refined grid
hierarchy (AMR).

A :py:class:`~.UniformGridData` object contains a :py:class:`~.UniformGrid` one.
Similarly, a :py:class:`~.HierarchicalGridData` contains multiple
:py:class:`~.UniformGridData`.

"""
from bisect import bisect_right
from os.path import splitext

import numpy as np
from scipy import interpolate, linalg

from kuibit import grid_data_utils as gdu
from kuibit.numerical import BaseNumerical


class UniformGrid:
    """Describes the geometry of a regular rectangular dataset, as well as
    information needed to identify the grid if part of refined grid hierarchy
    (namely component number and refinement level). In practice, this a fixed
    refinement level, or part of it (as output by an MPI process).

    This is a standard Cartesian grid that we will describe with the language
    of computer graphics. To make things clear, let's consider a 2D grid (see
    schematics below). We call the lower left corner "origin" or "x0". We call
    the top right corner "x1". The grid is cell-centered (see Fig 2).

    ..code-block::

        Fig 1

         o---------x1
         |          |
         |          |
         |          |
         |          |
        x0----------o

    ..code-block::

         Fig 2, the point sits in the center of a cell.

          --------
          |      |
          |  x0  |
          |      |
          --------

    The concept of ``shape`` is the same as NumPy shape: it's the number of points
    in each dimension. ``dx`` is the spacing (dx, dy, dz, ...). To fully
    describe a grid, one needs the ``origin``, the ``shape``, and ``x1`` or ``dx``.

    (This is the same convention that Carpet has.)

    This class is supposed to be immutable.

    :ivar ~.shape:     Number of points in each dimension.
    :type ~.shape:      1d NumPy arrary or list of int.
    :ivar ~.x0:    Position of cell center with lowest coordinate.
    :type ~.x0:     1d NumPy array or list of float.
    :ivar ~.dx:     If not None, specifies grid spacing, else grid
                      spacing is computed from x0, x1, and shape.
    :type ~.dx:      1d NumPy array or list of float.
    :ivar ~.x1:        If grid spacing is None, this specifies the
                      position of the cell center with largest
                      coordinates.
    :type ~.x1:         1d NumPy array or list of float.
    :ivar ~.ref_level:  Refinement level if this belongs to a hierarchy,
                      else -1.
    :type ~.ref_level:   int
    :ivar ~.component: Component number if this belongs to a hierarchy,
                      else -1.
    :type ~.component:  int
    :ivar ~.num_ghost:    Number of ghost zones (default=0)
    :type ~.num_ghost:     1d NumPy arrary or list of int.
    :ivar ~.time:      Time if that makes sense, else None.
    :type ~.time:       float or None
    :ivar ~.iteration: Iteration if that makes sense, else None.
    :type ~.iteration:  float or None

    """

    def _check_dims(self, var, name):
        """Check that the dimensions are consistent with the shape of the object."""
        if len(var.shape) != 1:
            raise ValueError(
                f"{name} must not be multi-dimensional {var.shape}."
            )
        if len(var) != len(self.shape):
            raise ValueError(
                f"The dimensions of this object are {self.shape.shape}, "
                f"not {var.shape} in {name}."
            )

    def __init__(
        self,
        shape,
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
        :param shape:     Number of points in each dimension.
        :type shape:      1d NumPy arrary or list of int.
        :param x0:    Position of cell center with lowest coordinate.
        :type x0:     1d NumPy array or list of float.
        :param dx:     If not None, specifies grid spacing, else grid
                          spacing is computed from x0, x1, and shape.
        :type dx:      1d NumPy array or list of float.
        :param x1:        If grid spacing is None, this specifies the
                          position of the cell center with largest
                          coordinates.
        :type x1:         1d NumPy array or list of float.
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

        self.__shape = np.atleast_1d(np.array(shape, dtype=int))
        self.__x0 = np.atleast_1d(np.array(x0, dtype=float))

        self._check_dims(self.shape, "shape")
        self._check_dims(self.x0, "x0")

        # Internally, what is important is shape, x0 and dx

        if dx is None:
            if x1 is None:
                raise ValueError("Must provide one between x1 and dx")

            # Here x1 is given, we compute dx. Consider the case
            # with three cells, x0 = (0,0) and x1 = (4,4).
            # Since we work with cell-centered, the x=0 line would
            # look like:
            #
            # --|------|------|------|------|--
            # (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
            #
            # If we want a dx of 1, we have need 5 points. Vice versa,
            # if we have n points, dx is (x1 - x0)/(n - 1).
            x1_arr = np.atleast_1d(x1)
            self._check_dims(x1_arr, "x1")

            if not all(self.x0 <= x1_arr):
                raise ValueError(
                    f"x1 {x1_arr} should be the upper corner (x0 = {x0})"
                )
            # We have to deal with grid with only one point along one direction.
            # When that happens, the only way to produce a grid that makes sense
            # is to have x0 and dx, because for those grids x0 = x1 along the
            # direction that has one single point.
            if np.any(self.shape == 1):
                raise ValueError(
                    "Cannot initialize a grid with a dimension"
                    " that has only one grid point by providing"
                    " x0 and x1. You must provide dx"
                )
            self.__dx = (x1_arr - self.x0) / (self.shape - 1)
        else:
            # Here we assume dx is given, but if also x1 is given, that
            # would may lead to problems if the paramters do not agree. So, we
            # first compute what x1 would be given x0 and dx, then if x1
            # is provided, we compare the result with the given x1. We raise an
            # error if they disagree.
            self.__dx = np.atleast_1d(np.array(dx, dtype=float))
            self._check_dims(self.dx, "dx")
            expected_x1 = self.x0 + (self.shape - 1) * self.dx
            if (x1 is not None) and (
                not np.allclose(expected_x1, x1, atol=1e-14)
            ):
                raise ValueError("Incompatible x1 and dx")

        if num_ghost is None:
            self.__num_ghost = np.zeros_like(self.shape)
        else:
            self.__num_ghost = np.atleast_1d(np.array(num_ghost, dtype=int))
            self._check_dims(self.num_ghost, "num_ghost")

        self.__ref_level = int(ref_level)
        self.__component = int(component)
        self.__time = None if time is None else float(time)
        self.__iteration = None if iteration is None else int(iteration)

        # The coordinates are a widely requested property, so the first time the
        # coordinates 1d are computed, we save them here.
        self.__coordinates_1d = None
        # Same with x1
        self.__x1 = None

        # The method __contains__ is called extremely often when dealing with
        # HierachicalGridData (because it is used to find which subgrid
        # contains a point). So, to make it faster, we save the values of the
        # bottom and upper cell faces (we account for the fact that the grid
        # is cell centered)
        self.__lowest_vertex = None
        self.__highest_vertex = None
        # Same considerations for num_dimensions
        self.__num_dimensions = len(self.shape)

    def __hash__(self):
        """UniformGrid is immutable, we can define an hash as the composition of
        the hashes of the members. This hash is quite slow to compute, so it
        is not useful for caching small computations. Having an hash function
        solidifies the idea that this class is immutable.
        """
        # We convert all the arrays in tuples (because they are hashable)
        hash_shape = hash(tuple(self.shape))
        hash_x0 = hash(tuple(self.x0))
        hash_dx = hash(tuple(self.dx))
        hash_num_ghost = hash(tuple(self.num_ghost))
        hash_ref_level = hash(self.ref_level)
        hash_component = hash(self.component)
        hash_time = hash(self.time)
        hash_iteration = hash(self.iteration)

        # ^ = bitwise xor
        return (
            hash_shape
            ^ hash_x0
            ^ hash_dx
            ^ hash_num_ghost
            ^ hash_ref_level
            ^ hash_component
            ^ hash_time
            ^ hash_iteration
        )

    @property
    def x0(self):
        """Lower corner.

        :returns: Center of lowest corner grid point.
        :rtype: 1d NumPy array
        """
        return self.__x0

    @property
    def shape(self):
        """Number of cells across each dimension.

        :returns: Number of cells across each dimension.
        :rtype: 1d NumPy array
        """
        return self.__shape

    @property
    def x1(self):
        """Upper corner.

        :returns: Center of top corner grid point.
        :rtype: 1d NumPy array
        """
        # We save x1 because it is computed a lot of times
        if self.__x1 is None:
            self.__x1 = self.x0 + (self.shape - 1) * self.dx
        return self.__x1

    @property
    def origin(self):
        """Lower corner.

        Alias for :py:meth:`~.x0`.

        :returns: Center of lowest corner grid point.
        :rtype: 1d NumPy array
        """
        return self.__x0

    @property
    def dx(self):
        """Grid spacing.

        :returns: Cell size across each dimension.
        :rtype: 1d NumPy array
        """
        return self.__dx

    @property
    def delta(self):
        """Grid spacing.

        Alias for :py:meth:`~.dx`.

        :returns: Cell size across each dimension.
        :rtype: 1d NumPy array
        """
        return self.__dx

    @property
    def num_ghost(self):
        """Number of ghost zones.

        :returns: Number of ghost zones across each dimension.
        :rtype: 1d NumPy array
        """
        return self.__num_ghost

    @property
    def ref_level(self):
        """Refinement level number.

        :returns: Refinement level number.
        :rtype: int
        """
        return self.__ref_level

    @property
    def component(self):
        """Component number.

        :returns: Component number.
        :rtype: int
        """
        return self.__component

    @property
    def time(self):
        """Time.

        :returns: Time.
        :rtype: float
        """
        return self.__time

    @property
    def iteration(self):
        """Iteration number

        :returns: Iteration number.
        :rtype: float
        """
        return self.__iteration

    @property
    def dv(self):
        """Volume of a grid cell.

        :returns: Volume of a grid cell.
        :rtype:   float
        """
        return self.dx.prod()

    @property
    def volume(self):
        """Volume of the whole grid.

        :returns: Volume of the whole grid.
        :rtype:   float
        """
        return self.shape.prod() * self.dv

    @property
    def num_dimensions(self):
        """Number of dimensions of the grid.

        :returns: Number of dimensions of the grid.
        :rtype:   float
        """
        return self.__num_dimensions

    @property
    def extended_dimensions(self):
        """Return an array of bools with whether a dimension has more than one
        point or not.

        :returns: Dimensions with more than one point.
        :rtype:   1d NumPy of bools
        """
        return self.shape > 1

    @property
    def num_extended_dimensions(self):
        """Return the number of dimensions with size larger than one gridpoint.

        :returns: The number of extended dimensions (the ones with more than one cell).
        :rtype:   int
        """
        return sum(self.extended_dimensions)

    @property
    def lowest_vertex(self):
        """Return the location of the lowest cell vertex (considering that
        the grid is cell centered).

        :returns: Lowest vertex of the lowest cell.
        :rtype:   1d NumPy array
        """
        if self.__lowest_vertex is None:
            self.__lowest_vertex = self.x0 - 0.5 * self.dx
        return self.__lowest_vertex

    @property
    def highest_vertex(self):
        """Return the location of the highest cell vertex (considering that
        the grid is cell centered).

        :returns: Highest vertex of the highest cell.
        :rtype:   1d NumPy array
        """
        if self.__highest_vertex is None:
            self.__highest_vertex = self.x1 + 0.5 * self.dx
        return self.__highest_vertex

    def indices_to_coordinates(self, indices):
        """Compute coordinate corresponding to one or more grid points.

        :param indices: Grid indices.
        :type indices:  1d array or list of int
        :returns: Corresponding coordinates of the grid points.
        :rtype:   1d NumPy array of float
        """
        # TODO (FEATURE): Add dimensionality checks
        return np.asarray(indices) * self.dx + self.x0

    def coordinates_to_indices(self, coordinates):
        """Find the indices corresponding to the point nearest to the given coordinates.

        :param coordinates: Coordinates.
        :type coordinates:  1d NumPy array or list of float
        :returns: Grid indices of nearest points.
        :rtype:   NumPy array

        """
        # TODO (FEATURE): Add dimensionality checks
        indices = (
            ((np.asarray(coordinates) - self.x0) / self.dx) + 0.5
        ).astype(np.int32)
        return indices

    def __getitem__(self, index):
        """Return the coordinates corresponding to a given (multi-dimensional)
        index.
        """
        index = np.array(index)
        self._check_dims(index, "index")
        coordinate = index * self.dx + self.x0
        if coordinate not in self:
            raise ValueError(f"{index} is not in on the grid")
        return coordinate

    def __contains__(self, point):
        """Test if a coordinate is contained in the grid. The size of the
        grid cells is taken into account, resulting in a cube larger by
        dx/2 on each side compared to the one given by x0, x1.

        :param point: Coordinate to test.
        :type point:  1d NumPy array or list of float.
        :returns:   If point is contained.
        :rtype:     bool
        """

        # A pythonic way to write this function is:
        # if np.any(point < (self.lowest_vertex)) or np.any(
        #     point > (self.highest_vertex)
        # ):
        #     return False
        # return True

        # However, this happens to be not the fastest. This method is called a
        # huge number of times when in HierarchicalGridData methods, because it
        # is the main method to find which grid contains a given point. (method
        # finest_component_at_point). So, it is better to have a less pythonic
        # method that fails as soon as possible.
        for dim in range(self.num_dimensions):
            if not (
                self.lowest_vertex[dim]
                <= point[dim]
                < self.highest_vertex[dim]
            ):
                return False
        return True

    def contains(self, point):
        """Test if a coordinate is contained in the grid. The size of the
        grid cells is taken into account, resulting in a cube larger by
        dx/2 on each side compared to the one given by x0, x1.

        :param point: Coordinate to test.
        :type point:  1d NumPy array or list of float.
        :returns:   If point is contained.
        :rtype:     bool
        """
        return point in self

    @property
    def coordinates_1d(self):
        """Return coordinates of the grid points.

        The return value is a list with the coordinates along each direction.

        :returns: Coordinates of the grid points on each direction.
        :rtype: list of 1d NumPy array

        """
        if self.__coordinates_1d is None:
            self.__coordinates_1d = [
                np.linspace(x0, x1, n)
                for n, x0, x1 in zip(self.shape, self.x0, self.x1)
            ]
        return self.__coordinates_1d

    def coordinates(self, as_meshgrid=False, as_same_shape=False):
        """Return coordinates of the grid points.

        If ``as_meshgrid`` is True, the coordinates are returned as NumPy
        meshgrid. Otherwise, return the coordinates of the grid points as 1D
        arrays (schematically, [array for x coordinates, array for y
        coordinates, ...]).

        If ``as_same_shape`` is True return the coordinates as an array with the
        same shape of self and with values the coordinates. This is useful for
        computations involving the coordinates. The output of ``as_same_shape``
        is the same as using ``np.mgrid``.

        :param as_meshgrid: If True, return the coordinates as meshgrid.
        :type as_meshgrid: bool
        :param as_same_shape: If True, return the coordinates as a list
                              or coordinates with the same shape of self
                              and with values of a given coordinate.
                              For instance, if ``self.num_dimension = 3`` there
                              will be three lists with ``shape = self.shape``.
                              This is equivalent to ``np.mgrid``.
        :type as_same_shape: bool
        :returns:  Grid coordinates.
        :rtype:   list of NumPy arrays with the same shape as grid

        """
        if as_meshgrid and as_same_shape:
            raise ValueError("Cannot ask for both meshgrid and shaped array.")

        if as_meshgrid:
            return np.meshgrid(*self.coordinates_1d)

        if as_same_shape:
            # np.indeces prepares an array in which each element has the value
            # of its index
            indices = np.indices(self.shape)
            # Here we transform the index into coordinate, along each dimension
            return [
                indices[dim] * self.dx[dim] + self.x0[dim]
                for dim in range(0, self.shape.size)
            ]

        return self.coordinates_1d

    def flat_dimensions_removed(self):
        """Return a new :py:class:`~.UniformGrid` with dimensions that have no
        flat dimensions (dimensions with only one grid point).

        :returns: Return a new grid without flat dimensions.
        :rtype: :py:class:`~.UniformGrid`
        """

        # We need this infrastructure to slice UniformGridData

        copied = self.copy()

        # We have to save this, otherwise it would be recomputed at every line,
        # affecting the result.
        extended_dims = copied.extended_dimensions

        copied.__shape = copied.__shape[extended_dims]
        copied.__x0 = copied.__x0[extended_dims]
        copied.__dx = copied.__dx[extended_dims]
        copied.__num_ghost = copied.__num_ghost[extended_dims]
        return copied

    def ghost_zones_removed(self):
        """Return a new :py:class:`~.UniformGrid` with ghostzones removed.

        :returns: Return a new grid without ghost zones.
        :rtype: :py:class:`~.UniformGrid`
        """
        copied = self.copy()

        # We remove twice the number of ghost zones because there are
        # lower and upper ghostzones
        copied.__shape = self.shape - 2 * self.num_ghost
        # We "push x0 inside the grid"
        copied.__x0 = copied.__x0 + self.num_ghost * self.dx
        copied.__num_ghost = np.zeros_like(copied.shape)
        return copied

    def shifted(self, shift):
        """Return a new UniformGrid with coordinates shifted by the given amount.

        ``x -> x + shift``.

        :param shift: Amount to shift coordinates.
        :type shift: 1d NumPy array
        :returns: New grid with coordinates shifted.
        :rtype: :py:class:`~.UniformGrid`

        """
        shift = np.asarray(shift)
        self._check_dims(shift, "shift")

        copied = self.copy()

        # We only need to shift x0 because x1 is computed from x0 using dx
        copied.__x0 = copied.__x0 + shift
        return copied

    def copy(self):
        """Return a deep copy.

        :returns:  Deep copy of the UniformGrid.
        :rtype:    :py:class:`~.UniformGrid`
        """
        return type(self)(
            self.shape,
            self.x0,
            dx=self.dx,
            x1=self.x1,
            ref_level=self.ref_level,
            component=self.component,
            num_ghost=self.num_ghost,
            time=self.time,
            iteration=self.iteration,
        )

    def __eq__(self, other):
        """Test if two :py:class:`~.UniformGrid` are the equal up to numerical
        precision.
        """
        if not isinstance(other, type(self)):
            return False
        # Time and iterations can be None, so we check them independently
        if self.time is not None and other.time is not None:
            time_bool = np.isclose(self.time, other.time, atol=1e-14)
        else:
            # In this case one of the two is None (or both)
            time_bool = self.time is other.time

        if self.iteration is not None and other.iteration is not None:
            iteration_bool = np.isclose(
                self.iteration, other.iteration, atol=1e-14
            )
        else:
            # In this case one of the two is None (or both)
            iteration_bool = self.iteration is other.iteration

        return (
            np.array_equal(self.shape, other.shape)
            and np.allclose(self.x0, other.x0, atol=1e-14)
            and np.allclose(self.dx, other.dx, atol=1e-14)
            and np.allclose(self.num_ghost, other.num_ghost, atol=1e-14)
            and np.allclose(self.ref_level, other.ref_level, atol=1e-14)
            and np.allclose(self.component, other.component, atol=1e-14)
            and time_bool
            and iteration_bool
        )

    def __str__(self):
        """:returns: a string describing the geometry."""
        return f"""Shape            = {self.shape}
Num ghost zones  = {self.num_ghost}
Ref. level       = {self.ref_level}
Component        = {self.component}
x0               = {self.x0}
x0/dx            = {self.x0/self.dx}
x1               = {self.x1}
x1/dx            = {self.x1/self.dx}
Volume           = {self.volume}
dx               = {self.dx}
Time             = {self.time}
Iteration        = {self.iteration}
"""


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

        coords = self.grid.coordinates()

        if k not in (0, 1):
            raise ValueError(
                "Order for splines for dimensions > 2 must be 0 or 1"
            )

        # Here k is 0 or 1
        method = "nearest" if k == 0 else "linear"

        # Our grid is cell-centered, so it is perfecly valid to evaluate a point
        # that it is outside coords, as long as it is within 0.5 * dx. For
        # example, if the grid is linear from 0 to 10 with dx = 1, the pint
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
        indices_arr[outside_indices] = np.zeros(
            self.num_dimensions, dtype=np.int
        )

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
            [
                slice(ghost_zones, -ghost_zones)
                for ghost_zones in self.num_ghost
            ]
        )
        new_data = self.data[slicer]
        return type(self)(new_grid, new_data)

    def ghost_zones_remove(self):
        """Remove all the ghost zones."""
        self._apply_to_self(self.ghost_zones_removed)

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
        new_shape = ((self.x1 - self.x0) / new_dx + 1.5).astype(np.int64)
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

    def partial_differentiated(self, direction, order=1):
        """Return a :py:class:`~.UniformGridData` that is the numerical
        order-differentiation of the present grid_data along a given direction.
        (``order`` = number of derivatives, ie ``order=2`` is second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of ``self``.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int
        :param direction: Direction of the partial derivative.
        :type direction: int

        :returns:  New :py:class:`~.UniformGridData` with derivative.
        :rtype:    :py:class:`~.UniformGridData`

        """
        if direction < 0 or direction >= self.num_dimensions:
            raise ValueError(
                f"Grid has {self.num_dimensions}, dimensions, "
                f"{direction} is not available"
            )

        ret_value = self.data.copy()
        for _num_deriv in range(order):

            ret_value = np.gradient(
                ret_value, self.dx[direction], axis=direction, edge_order=2
            )
        return type(self)(self.grid, ret_value)

    def gradient(self, order=1):
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
        :returns:  list of :py:class:`~.UniformGridData` with partial derivative
                   along the directions.
        :rtype:    list of :py:class:`~.UniformGridData`

        """
        return [
            self.partial_differentiated(direction, order=order)
            for direction in range(self.num_dimensions)
        ]

    def partial_differentiate(self, dimension, order=1):
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

        """
        self._apply_to_self(
            self.partial_differentiated, dimension, order=order
        )

    def _apply_unary(self, function):
        """Apply a unary function to the data.

        :param function: Unary function.
        :type function:  callable
        :returns: Function applied to the data.
        :rtype:    :py:class:`~.UniformGridData`

        """
        return type(self)(self.grid, function(self.data))

    def _apply_reduction(self, reduction):
        """Apply a reduction to the data.

        :param function: Function to apply to the data.
        :type function: callable

        :return: Reduction applied to the data.
        :rtype: float

        """
        return reduction(self.data)

    def _apply_binary(self, other, function):
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
            return type(self)(self.grid, function(self.data, other.data))

        # If it is a number
        if isinstance(other, (int, float, complex)):
            return type(self)(self.grid, function(self.data, other))

        # If we are here, it is because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            np.allclose(self.data, other.data, atol=1e-14)
            and self.grid == other.grid
        )

    def fourier_transform(self):
        """Perform the multi-dimensional Fourier transform on the data.

        We follow NumPy's conventions, with the exception that we normalize the
        amplitude with ``dx``.

        If the signal is complex, we also shift the negative components to be in
        the negative part of the signal.

        :returns: Fourier transform.
        :rtype: :py:class:`~.UniformGridData`

        """
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

        for comp in uniform_grid_data_sorted:
            components.setdefault(comp.ref_level, []).append(comp)

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
            np.array([len(x) - 1 for x in boundaries_tilde]), dtype=np.int32
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

        for comp in components:
            # We find the index corresponding to x0 and x1 of the component
            index_x0 = ((comp.x0 - grid.x0) / grid.dx + 0.5).astype(np.int32)
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
                "Data does not have a well defined x0 "
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

    def copy(self):
        """Return a deep copy.

        :returns:  Deep copy of the :py:class:`~.HierarchicalGridData`.
        :rtype:    :py:class:`~.HierarchicalGridData`
        """
        return type(self)(self.all_components)

    def __eq__(self, other):
        """Check for equality."""
        if (
            not isinstance(other, HierarchicalGridData)
            or self.shape != other.shape
        ):
            return False

        return self.all_components == other.all_components

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

    def merge_refinement_levels(self, resample=False):
        """Combine all the available data and resample it grid that encompasses all the
        components and has resolution of the finest refinement level.

        When ``resample`` is True, data from coarser refinement levels is
        resampled with multilinear interpolation, otherwise the nearest
        neighbors are used.

        .. warning::

            For most practical purposes, using this function is an overkill.
            This can be a very expensive operation and require a lot of memory.
            Prefer :py:meth:`to_UniformGridData` when possible.

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
        new_shape = ((self.x1 - self.x0) / new_dx + 1.5).astype(np.int64)
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

    def _apply_binary(self, other, function):
        """Apply a binary function to the data of ``self`` and ``other``.

        :param function: Function to apply to all the data in the various
        refinement levels.
        :type function: callable

        :return: New :py:class:`~.HierarchicalGridData` with function applied to
        ``self.data`` and ``other.data``.
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        # We only know what how to combine HierarhicalGridData
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Grid structure incompatible")
            new_data = [
                function(data_self, data_other)
                for data_self, data_other in zip(
                    self.all_components, other.all_components
                )
            ]
            return type(self)(new_data)

        if isinstance(other, (int, float, complex)):
            new_data = [
                function(data_self, other) for data_self in self.all_components
            ]
            return type(self)(new_data)

        # If we are here, it is because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")

    def _apply_reduction(self, reduction):
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
                    data._apply_reduction(reduction)
                    for data in self.all_components
                ]
            )
        )

    def _apply_unary(self, function):
        """Apply a unary function to the data.

        :param function: Function to apply to all the data in the various
        refinement levels
        :type function: callable

        :return: New :py:class:`~.HierarchicalGridData` with function applied to
        the data.
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        new_data = [function(data) for data in self.all_components]
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

    def partial_differentiated(self, direction, order=1):
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

        :returns:  New :py:class:`~.HierarchicalGridData` with derivative.
        :rtype:    :py:class:`~.HierarchicalGridData`

        """
        return self._call_component_method(
            "partial_differentiated", direction, order=order
        )

    def gradient(self, order=1):
        """Return a list :py:class:`~.HierarchicalGridData` that are the numerical
        order-differentiation of the present grid_data along all the directions.
        (order = number of derivatives, ie ``order=2`` is second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of self.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int
        :returns: list of :py:class:`~.HierarchicalGridData` with partial
                  derivative along all the directions.
        :rtype:  list of :py:class:`~.HierarchicalGridData`

        """
        return self._call_component_method(
            "gradient", method_returns_list=True, order=order
        )

    def partial_differentiate(self, direction, order=1):
        """Apply a numerical differentiatin along the specified direction.

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of ``self``.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int
        :param direction: Direction of the partial derivative.
        :type direction: int

        :returns: Derivative along the specified direction.
        :rtype: list of :py:class:`~.HierarchicalGridData`

        """
        return self._apply_to_self(
            self.partial_differentiated, direction, order=order
        )

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

    def __str__(self):
        ret = "Available refinement levels (components):\n"
        for ref_level in self.refinement_levels:
            ret += f"{ref_level} ({len(self[ref_level])})\n"
        ret += f"Spacing at coarsest level ({self.num_coarsest_level}): "
        ret += f"{self.coarsest_dx}\n"
        ret += f"Spacing at finest level ({self.num_finest_level}): {self.finest_dx}"
        return ret
