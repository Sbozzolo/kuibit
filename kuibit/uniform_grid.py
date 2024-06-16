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

"""The :py:mod:`~.uniform_grid` module provides representations of uniform
grids.The object defined is :py:class:`~.UniformGrid`, which represents the
geometry of a uniform Cartesian cell-centered grid.

This is a basic building block of :py:class:`~.UniformGridData`.

"""
import numpy as np


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
        # We combine information by taking the hash of the tuple, where we
        # convert all the arrays in tuples (because they are hashable)
        return hash(
            (
                tuple(self.shape),
                tuple(self.x0),
                tuple(self.dx),
                self.ref_level,
                self.component,
                self.time,
                self.iteration,
            )
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
        for dim in range(self.num_dimensions):  # skipcq: PY-W0075
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
