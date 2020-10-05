#!/usr/bin/env python3

"""The :py:mod:`~.grid_data` module provides representations of data on
uniform grids as well as for data on refined grid hirachies. Standard
arithmetic operations are supported for those data grids, further methods
to interpolate and resample. The number of dimensions is arbitrary.
Rudimentary vector and matrix oprations are also supported, using
Vectors of data grids (instead of grids of vectors).

The important classes defined here are
 * :py:class:`~.UniformGrid`  represents the geometry of a uniform grid.
 * :py:class:`~.RegData`    represents data on a uniform grid and the
   geometry.
 * :py:class:`~.CompData`   represents data on a refined grid hirachy.
 * :py:class:`~.Vec`        represents a generic vector
 * :py:class:`~.Mat`        represents a generic matrix
"""

import numpy as np


class UniformGrid:
    """Describes the geometry of a regular rectangular dataset, as well as
    information needed if part of refined grid hierachy, namely component
    number and refinement level. In practice, this a fixed refinement level.

    Also stores the number of ghost zones, which is however not used anywhere
    in this class.

    This is a standard Cartesian grid that we will describe with the language
    of computer graphics. To make things clear, let's consider a 2D grid (see
    schematics below). We call the lower left corner "origin" or `x0`. We call
    the top right corner "x1". The grid is cell-centered (see Fig 2).

    Fig 1

     o---------x1
     |          |
     |          |
     |          |
     |          |
    x0----------o


    Fig 2, the point sits in the center of a cell.

     --------
     |      |
     |  x0  |
     |      |
     --------

    The concept of shape is the same as NumPy shape: it's the number of points
    in each dimention. Delta is the spacing (dx, dy, dz, ...). To fully
    describe a grid, one needs the origin, the shape, and x1 or delta.


    This class is supposed to be immutable.

    """

    def _check_dims(self, var, name):
        """Check that the dimensions are consistent with the shape of the object."""
        if len(var.shape) != 1:
            raise ValueError(f"{name} must not be multi-dimensional.")
        if len(var) != len(self.shape):
            raise ValueError(
                f"The dimensions of this object are {self.shape}, not{name}."
            )

    def __init__(
        self,
        shape,
        origin,
        delta=None,
        x1=None,
        ref_level=-1,
        component=-1,
        num_ghost=None,
        time=None,
        iteration=None,
    ):
        """
        :param shape:     Number of points in each dimension.
        :type shape:      1d numpy arrary or list of int.
        :param origin:    Position of cell center with lowest coordinate.
        :type origin:     1d numpy array or list of float.
        :param delta:     If not None, specifies grid spacing, else grid
                          spacing is computed from x0, x1, and shape.
        :type delta:      1d numpy array or list of float.
        :param x1:        If grid spacing is None, this specifies the
                          position of the cell center with largest
                          coordinates.
        :type x1:         1d numpy array or list of float.
        :param ref_level:  Refinement level if this belongs to a hierachy,
                          else -1.
        :type ref_level:   int
        :param component: Component number if this belongs to a hierachy,
                          else -1.
        :type component:  int
        :param num_ghost:    Number of ghost zones (default=0)
        :type num_ghost:     1d numpy arrary or list of int.
        :param time:      Time if that makes sense, else None.
        :type time:       float or None
        :param iteration: Iteration if that makes sense, else None.
        :type iteration:  float or None

        """
        self.__shape = np.array(shape, dtype=int)
        self.__origin = np.array(origin, dtype=float)

        self._check_dims(self.__shape, "shape")
        self._check_dims(self.origin, "origin")

        if delta is None:
            if x1 is None:
                raise ValueError("Must provide one between x1 and delta")

            # Here x1 is given, we compute delta. Consider the case
            # with three cells, origin = (0,0) and x1 = (4,4).
            # Since we work with cell-centered, the x=0 line would
            # look like:
            #
            # --|------|------|------|------|--
            # (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
            #
            # If we want a delta of 1, we have need 5 points. Vice versa,
            # if we have n points, delta is (x1 - x0)/(n - 1).
            x1_arr = np.array(x1, dtype=float)
            self._check_dims(x1_arr, "x1")
            self.__delta = (x1_arr - self.origin) / (self.shape - 1)
        else:
            # Here we assume delta is given, but if also x1 is given, that
            # would may lead to problems if the paramters do not agree. So, we
            # first compute what x1 would be given origin and delta, then if x1
            # is provided, we compare the result with the given x1. We raise an
            # error if they disagree.
            self.__delta = np.array(delta, dtype=float)
            self._check_dims(self.delta, "delta")
            expected_x1 = self.origin + (self.shape - 1) * self.delta
            if x1 is not None:
                if not np.allclose(expected_x1, x1, atol=1e-14):
                    raise ValueError("Incompatible x1 and delta")

        if num_ghost is None:
            self.__num_ghost = np.zeros_like(self.shape)
        else:
            self.__num_ghost = np.array(num_ghost, dtype=int)
            self._check_dims(self.num_ghost, "num_ghost")

        self.__ref_level = int(ref_level)
        self.__component = int(component)
        self.__time = None if time is None else float(time)
        self.__iteration = None if iteration is None else int(iteration)

    @property
    def x0(self):
        return self.__origin

    @property
    def shape(self):
        return self.__shape

    @property
    def x1(self):
        return self.origin + (self.shape - 1) * self.delta

    @property
    def origin(self):
        return self.__origin

    @property
    def delta(self):
        return self.__delta

    @property
    def dx(self):
        return self.__delta

    @property
    def num_ghost(self):
        return self.__num_ghost

    @property
    def ref_level(self):
        return self.__ref_level

    @property
    def component(self):
        return self.__component

    @property
    def time(self):
        return self.__time

    @property
    def iteration(self):
        return self.__iteration

    @property
    def dv(self):
        """
        :returns: Volume of a grid cell.
        :rtype:   float
        """
        return self.delta.prod()

    @property
    def volume(self):
        """
        :returns: Volume of the whole grid.
        :rtype:   float
        """
        return self.shape.prod() * self.dv

    @property
    def num_dimensions(self):
        return len(self.shape)

    @property
    def extended_dimensions(self):
        """Return an array of bools with whether a dimension has more than one
        point or not.
        """
        return self.shape > 1

    @property
    def num_extended_dimensions(self):
        """
        :returns: The number of dimensions with size larger than one gridpoint.
        :rtype:   int
        """
        return sum(self.extended_dimensions)

    def __getitem__(self, index):
        """Return the coordinates corresponding to a given (multi-dimensional)
        index.
        """
        index = np.array(index)
        self._check_dims(index, "index")
        return index * self.delta + self.origin

    def __contains__(self, point):
        """Test if a coordinate is contained in the grid. The size of the
        grid cells is taken into account, resulting in a cube larger by
        dx/2 on each side compared to the one given by x0, x1.

        :param point: Coordinate to test.
        :type point:  1d numpy array or list of float.
        :returns:   If point is contained.
        :rtype:     bool
        """
        point = np.array(point)
        if not np.alltrue(point > (self.x0 - 0.5 * self.dx)):
            return False
        if not np.alltrue(point < (self.x1 + 0.5 * self.dx)):
            return False
        return True

    def contains(self, point):
        """Test if a coordinate is contained in the grid. The size of the
        grid cells is taken into account, resulting in a cube larger by
        dx/2 on each side compared to the one given by x0, x1.

        :param point: Coordinate to test.
        :type point:  1d numpy array or list of float.
        :returns:   If point is contained.
        :rtype:     bool
        """
        return point in self

    def coordinates(self, as_1d_arrays=False, as_meshgrid=False):
        """Return coordinates of the grid points.

        If as_1d_arrays is True, return the coordinates of the grid points as
        1D arrays (schematically, [array for x coordinates, array for y
        coordinates, ...])

        If as_meshgrid is True, the coordinates are returned as NumPy meshgrid.

        If neither is True, return the coordinates a list of as
        multidimensional arrays with the same shape as the grid. Useful for
        arithmetic computations involving both data and coordinates.

        For example, for a 2D grid, this would be

        X, Y = self.coordinates()

        X, Y are arrays with have the same shape as the grid. X has the x
        coordinates of all the points, and similarly does Y.

        :param as_1d_arrays: If True, return a list of 1d arrays of coordinates
        along the different axes.
        :type as_1d_arrays: bool

        :returns: The coordinate array of each dimension.
        :rtype:   list of numpy arrays with the same shape as grid

        """
        if as_meshgrid and as_1d_arrays:
            raise ValueError(
                "Cannot request two different type of returns in coordinates"
            )

        if as_1d_arrays:
            return [
                np.linspace(x0, x1, n)
                for n, x0, x1 in zip(self.shape, self.x0, self.x1)
            ]

        if as_meshgrid:
            return np.meshgrid(
                *[
                    np.linspace(x0, x1, n)
                    for n, x0, x1 in zip(self.shape, self.x0, self.x1)
                ]
            )

        # np.indeces prepares a multimensional array given a shape with content
        # the corresponding index.

        return [
            np.indices(self.shape)[d] * self.dx[d] + self.x0[d]
            for d in range(0, self.shape.size)
        ]

    def flat_dimensions_remove(self):
        """Remove dimensions which are only one gridpoint across"""
        # We have to save this, otherwise it would be recomputed at every line,
        # affecting the result.
        extended_dims = self.extended_dimensions

        self.__shape = self.__shape[extended_dims]
        self.__origin = self.__origin[extended_dims]
        self.__delta = self.__delta[extended_dims]
        self.__num_ghost = self.__num_ghost[extended_dims]

    def copy(self):
        """Return a deep copy.

        :returns:  Deep copy of the UniformGrid
        :rtype:    :py:class:`~.UniformGrid`
        """
        return type(self)(
            self.shape,
            self.origin,
            delta=self.delta,
            x1=self.x1,
            ref_level=self.ref_level,
            component=self.component,
            num_ghost=self.num_ghost,
            time=self.time,
            iteration=self.iteration,
        )

    def __eq__(self, other):
        return (
            np.allclose(self.shape, other.shape, atol=1e-14)
            and np.allclose(self.origin, other.origin, atol=1e-14)
            and np.allclose(self.delta, other.delta, atol=1e-14)
            and np.allclose(self.num_ghost, other.num_ghost, atol=1e-14)
            and np.allclose(self.ref_level, other.ref_level, atol=1e-14)
            and np.allclose(self.component, other.component, atol=1e-14)
            and np.allclose(self.time, other.time, atol=1e-14)
            and np.allclose(self.iteration, other.iteration, atol=1e-14)
        )

    def __str__(self):
        """:returns: a string describing the geometry."""
        return f"""Shape            = {self.shape}
Num ghost zones  = {self.num_ghost}
Ref. level       = {self.ref_level}
Component        = {self.component}
x0               = {self.x0}
x0/delta         = {self.x0/self.dx}
x1               = {self.x1}
x1/delta         = {self.x0/self.dx}
Volume           = {self.volume}
Delta            = {self.dx}
Time             = {self.time}
Iteration        = {self.iteration}
"""


def common_bounding_box(grids):
    """Return corners of smallest common bounding box of regular grids.

    :param geoms: list of grid geometries.
    :type geoms:  list of :py:class:`~.UniformGrid`
    :returns: the common bounding box of a list of geometries
    :rtype: tuple of coordinates (x0 and x1)
    """
    # We have to check that the number of dimensions is the same
    try:
        num_dim = grids[0].num_dimensions
    except AttributeError:
        raise TypeError("bounding_box takes a list of UniformGrids")

    for g in grids:
        if g.num_dimensions != num_dim:
            raise ValueError(
                "All the UniformGrids must have the same number of dimensions"
            )

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
    # In this way we put all the same coordiantes in a single
    # row. We can take the minimum and maximum along these rows
    # to find the common bounding box.
    x0 = np.array([min(b) for b in np.transpose(x0s)])
    x1 = np.array([max(b) for b in np.transpose(x1s)])
    return (x0, x1)
