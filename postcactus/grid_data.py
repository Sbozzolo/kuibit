#!/usr/bin/env python3

"""The :py:mod:`~.grid_data` module provides representations of data on
uniform grids as well as for data on refined grid hirachies. Standard
arithmetic operations are supported for those data grids, further methods
to interpolate and resample. The number of dimensions is arbitrary.

The important classes defined here are
 * :py:class:`~.UniformGrid`  represents the geometry of a uniform grid.
 * :py:class:`~.UniformGridData`  represents data on a uniform grid.
 * :py:class:`~.HierarchicalGridData` represents data on a refined grid
hierachy.
"""

import itertools

import numpy as np
from scipy import interpolate
from scipy import linalg

from postcactus.numerical import BaseNumerical


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
    in each dimention. dx is the spacing (dx, dy, dz, ...). To fully
    describe a grid, one needs the origin, the shape, and x1 or dx.

    This is the same convention that Carpet has.

    This class is supposed to be immutable.

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
        :type shape:      1d numpy arrary or list of int.
        :param x0:    Position of cell center with lowest coordinate.
        :type x0:     1d numpy array or list of float.
        :param dx:     If not None, specifies grid spacing, else grid
                          spacing is computed from x0, x1, and shape.
        :type dx:      1d numpy array or list of float.
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
            x1_arr = np.atleast_1d(np.array(x1, dtype=float))
            self._check_dims(x1_arr, "x1")

            if not all(self.x0 <= x1_arr):
                raise ValueError(
                    f"x1 {x1_arr} should be the upper corner (x0 = {x0})"
                )

            # If shape has ones, then dx does not make sense, so we create a
            # temporary temp_shape object where we substitute the ones with
            # zeros, so dx ends up being negative where shape is 1. Then, we
            # force the negative values to zero.
            temp_shape = self.shape.copy()
            temp_shape[temp_shape == 1] = 0
            self.__dx = (x1_arr - self.x0) / (temp_shape - 1)
            self.__dx[self.__dx < 0] = 0
        else:
            # Here we assume dx is given, but if also x1 is given, that
            # would may lead to problems if the paramters do not agree. So, we
            # first compute what x1 would be given x0 and dx, then if x1
            # is provided, we compare the result with the given x1. We raise an
            # error if they disagree.
            self.__dx = np.atleast_1d(np.array(dx, dtype=float))
            self._check_dims(self.dx, "dx")
            expected_x1 = self.x0 + (self.shape - 1) * self.dx
            if x1 is not None:
                if not np.allclose(expected_x1, x1, atol=1e-14):
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

    def __hash__(self):
        """UniformGrid is immutable, we can define an hash as the composition of
        the hases of the members. This hash is quite slow to compute, so it
        is not useful for caching small computations.
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
        return self.__x0

    @property
    def shape(self):
        return self.__shape

    @property
    def x1(self):
        # We save x1 because it is computed a lot of times
        if self.__x1 is None:
            self.__x1 = self.x0 + (self.shape - 1) * self.dx
        return self.__x1

    @property
    def origin(self):
        return self.__x0

    @property
    def dx(self):
        return self.__dx

    @property
    def delta(self):
        return self.__dx

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
        return self.dx.prod()

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

    @property
    def lowest_vertex(self):
        """Return the location of the lowest cell vertex (considering that
        the grid is cell centered).
        """
        if self.__lowest_vertex is None:
            self.__lowest_vertex = self.x0 - 0.5 * self.dx
        return self.__lowest_vertex

    @property
    def highest_vertex(self):
        """Return the location of the highest cell vertex (considering that
        the grid is cell centered).
        """
        if self.__highest_vertex is None:
            self.__highest_vertex = self.x1 + 0.5 * self.dx
        return self.__highest_vertex

    def indices_to_coordinates(self, indices):
        """Compute coordinate corresponding to one or more grid points.

        :param indices: Grid indices
        :type indices:  1d array or list of int.
        :returns: The coordinate of grid points
        :rtype:   1d numpy array of float
        """
        # TODO: Add dimensionality checks
        return np.array(indices) * self.dx + self.x0

    def coordinates_to_indices(self, coordinates):
        """Find the indices corresponding to the
        point nearest to the given coordinates.

        :param coordinates: Coordinates.
        :type coordinates:  1d numpy array or list of float
        :returns: grid indidces of nearest point.
        :rtype:   array of int
        """
        # TODO: Add dimensionality checks
        return (((np.array(coordinates) - self.x0) / self.dx) + 0.5).astype(
            np.int32
        )

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
        :type point:  1d numpy array or list of float.
        :returns:   If point is contained.
        :rtype:     bool
        """

        # A pythonic way to write this function is:
        # if np.any(point < (self.__lower_edge)) or np.any(
        #     point > (self.__upper_edge)
        # ):
        #     return False
        # return True
        #
        # However, this happens to be not the fastest. This method is called a
        # huge number of times when in HierarchicalGridData methods, because it
        # is the main method to find which grid contains a given point.
        # (method finest_level_component_at_point). So, it is better to have
        # a less pythonic method that fails as soon as possible.
        for dim in range(self.num_dimensions):
            if not (
                self.lowest_vertex[dim]
                <= point[dim]
                <= self.highest_vertex[dim]
            ):
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

    @property
    def coordinates_1d(self):
        """Return coordinates of the grid points.

        The return value is a list with the coordinates along each direction

        """
        if self.__coordinates_1d is None:
            self.__coordinates_1d = [
                np.linspace(x0, x1, n)
                for n, x0, x1 in zip(self.shape, self.x0, self.x1)
            ]
        return self.__coordinates_1d

    def coordinates(self, as_meshgrid=False, as_same_shape=False):
        """Return coordinates of the grid points.

        If as_meshgrid is True, the coordinates are returned as NumPy meshgrid.
        Otherwise, return the coordinates of the grid points as
        1D arrays (schematically, [array for x coordinates, array for y
        coordinates, ...]).

        If True as_same_shape is True return the coordinates as an array
        with the same shape of self and with values the coordinates.

        :param as_meshgrid: If True, return the coordinates as meshgrid.
        :type as_meshgrid: bool

        :param as_same_shape: If True, return the coordinates as a list
        or coordinates with the same shape of self and with values of a given
        coordinate. For instance, if the self.num_dimension there will be
        three lists with shape = self.shape.
        :type as_same_shape: bool

        :returns:  A list of 1d arrays of coordinates
        along the different axes.
        :rtype:   list of numpy arrays with the same shape as grid

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
        """Return a new UniformGrid with dimensions which are only one gridpoint across
        removed."""
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
        """Return a new UniformGrid with ghostzones removed"""
        copied = self.copy()

        # We remove twice the number of ghost zones because there are
        # lower and upper ghostzones
        copied.__shape = self.shape - 2 * self.num_ghost
        # We "push x0 inside the grid"
        copied.__x0 = copied.__x0 + self.num_ghost * self.dx
        copied.__num_ghost = np.zeros_like(copied.shape)
        return copied

    def shifted(self, shift):
        """Return a new UniformGrid with coordinates shifted by some amount

        x -> x + shift."""
        shift = np.array(shift)
        self._check_dims(shift, "shift")

        copied = self.copy()

        # We only need to shift x0 because x1 is computed from x0 using dx
        copied.__x0 = copied.__x0 + shift
        return copied

    def copy(self):
        """Return a deep copy.

        :returns:  Deep copy of the UniformGrid
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
x1/dx            = {self.x0/self.dx}
Volume           = {self.volume}
dx               = {self.dx}
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
    # Let's check that grids is a list like objects
    if not hasattr(grids, "__len__"):
        raise TypeError("common_bounding_box takes a list")

    # Check that they are all UniformGrids
    if not all(isinstance(g, UniformGrid) for g in grids):
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
    # In this way we put all the same coordiantes in a single
    # row. We can take the minimum and maximum along these rows
    # to find the common bounding box.
    x0 = np.array([min(b) for b in np.transpose(x0s)])
    x1 = np.array([max(b) for b in np.transpose(x1s)])
    return (x0, x1)


def merge_uniform_grids(grids, component=-1):
    """Compute a regular grid covering the bounding box of a list of grid
    geometries, with the same grid spacing. All geometries must belong to the
    same refinement level and have the same dx. In practice, we return a new
    grid that covers all the grids in the list.

    dx is kept constant, but the number of points will change.

    :param geoms: list of grid geometries.
    :type geoms:  list of :py:class:`~.UniformGrid`
    :returns: Grid geometry covering all input grids.
    :rtype: :py:class:`~.UniformGrid`

    """
    # Let's check that grids is a list like objects
    if not hasattr(grids, "__len__"):
        raise TypeError("merge_uniform_grids takes a list")

    if not all(isinstance(g, UniformGrid) for g in grids):
        raise TypeError("merge_uniform_grids works only UniformGrid")

    # Check that all the grids have the same refinement levels
    ref_levels = {g.ref_level for g in grids}

    if len(ref_levels) != 1:
        raise ValueError("Can only merge grids on same refinement level.")

    # Extract the only element from the set
    ref_level = next(iter(ref_levels))

    dx = [g.dx for g in grids]

    if not np.allclose(dx, dx[0]):
        raise ValueError("Can only merge grids on with same dx.")

    # Find the bounding box
    x0, x1 = common_bounding_box(grids)

    # The additional 1.5 and 0.5 factors are because the points are
    # cell-centered, so the cells have size

    # dx here is a list of all the dx, we just want one (they are all the same)
    shape = ((x1 - x0) / dx[0] + 1.5).astype(np.int64)

    return UniformGrid(
        shape, x0=x0, dx=dx[0], ref_level=ref_level, component=component
    )


class UniformGridData(BaseNumerical):
    """Represents a rectangular data grid with coordinates, supporting
    common arithmetic operations.

    :ivar grid: Uniform grid over which the data is defined.
    :type grid: :py:class:`~.UniformGrid`

    :ivar data: The actual data (numpy array).

    """

    # We are deriving this from BaseNumerical. This will give all the
    # mathematical operators for free, as long as we defined _apply_unary
    # and _apply_binary.

    def __init__(self, grid, data):
        """
        :param grid: Uniform grid over which the data is defined
        :type grid: :py:class:`~.UniformGrid`
        :param data: The data.
        :type data: A numpy array.
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
        # In case of dimensions larger than 2, this will be an object
        # RegularGridInterpolator.
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
        :type x0:     1d numpy array or list of float.
        :param dx:     If not None, specifies grid spacing, else grid
                          spacing is computed from x0, x1, and shape.
        :type dx:      1d numpy array or list of float.
        :param data:      The data.
        :type data:       A numpy array.
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
        """Return coordinates of the grid points as list of UniformGridData.

        This can be used for computations."""
        return [
            type(self)(self.grid, coord)
            for coord in self.coordinates_from_grid(as_same_shape=True)
        ]

    def coordinates_from_grid(self, as_meshgrid=False, as_same_shape=False):
        """Return coordinates of the grid points.

        If as_meshgrid is True, the coordinates are returned as NumPy meshgrid.
        Otherwise, return the coordinates of the grid points as
        1D arrays (schematically, [array for x coordinates, array for y
        coordinates, ...]).

        If True as_same_shape is True return the coordinates as an array
        with the same shape of self and with values the coordinates.

        :param as_meshgrid: If True, return the coordinates as meshgrid.
        :type as_meshgrid: bool

        :param as_same_shape: If True, return the coordinates as an array
        with the same shape of self and with values the coordinates.
        :type as_same_shape: bool

        :returns:  A list of 1d arrays of coordinates
        along the different axes.
        :rtype:   list of numpy arrays with the same shape as grid

        """
        return self.grid.coordinates(
            as_meshgrid=as_meshgrid, as_same_shape=as_same_shape
        )

    def coordinates_meshgrid(self):
        """Return coordinates of the grid points as numpy meshgrid.
        This is useful for plotting"""
        return self.coordinates_from_grid(as_meshgrid=True)

    # # TODO: Implement load and save methods
    # def save(self, file_name, *args, **kwargs):
    #     pass

    # @classmethod
    # def load(cls, file_name, *args, **kwargs):
    #     """Load UniformGridData from saved instance
    #     """
    #     pass

    @property
    def x0(self):
        return self.grid.x0

    @property
    def shape(self):
        return self.grid.shape

    @property
    def x1(self):
        return self.grid.x1

    @property
    def origin(self):
        return self.x0

    @property
    def dx(self):
        return self.grid.dx

    @property
    def delta(self):
        return self.dx

    @property
    def num_ghost(self):
        return self.grid.num_ghost

    @property
    def ref_level(self):
        return self.grid.ref_level

    @property
    def component(self):
        return self.grid.component

    @property
    def time(self):
        return self.grid.time

    @property
    def iteration(self):
        return self.grid.iteration

    def __getitem__(self, key):
        return self.data[key]

    def _make_spline(self, *args, k=1, **kwargs):
        """Private function to make spline representation of the data using
        scipy.interpolate.RegularGridInterpolator.

        Only nearest neighbor or multilinear interpolations are available.

        This function is not meant to be called directly.

        :param k: Order of the interpolation (k = 0 or 1)
        :type k:  int

        """

        coords = self.grid.coordinates()

        if k not in (0, 1):
            raise ValueError(
                "Order for splines for dimensions > 2 must be 0 or 1"
            )

        # Here k is 0 or 1
        method = "nearest" if k == 0 else "linear"

        self.spline_real = interpolate.RegularGridInterpolator(
            coords,
            self.data.real,
            method=method,
            fill_value=0,
            bounds_error=True,
        )

        if self.is_complex():
            self.spline_imag = interpolate.RegularGridInterpolator(
                coords,
                self.data.imag,
                method=method,
                fill_value=0,
                bounds_error=True,
            )

        self.invalid_spline = False

    def evaluate_with_spline(self, x, ext=2, piecewise_constant=False):
        """Evaluate the spline on the points x.

        Values outside the interval are set to 0 if ext=1, or a ValueError is
        raised if ext=2.

        This method is meant to be used only if you want to use a different ext
        for a specific call, otherwise, just use __call__.

        :param x: Array of x where to evaluate the series or single x
        :type x: 1D numpy array of float, or UniformGrid

        :param ext: How to deal values outside the bounaries. Values outside
                    the interval are set to 0 if ext=1,
                    or an error is raised if ext=2.
        :type ext:  bool

        :returns: Values of the series evaluated on the input x
        :rtype:   1D numpy array or float

        """
        # ext = 0 is extrapolation and ext = 3 is setting the boundary
        # value. We cannot do this with RegularGridInterpolator

        # Check if we have the point on the grid, in that case return the
        # value. At the moment, we do this only if x is one single point.
        #
        # Not a UniformGrid and only a single point
        if not isinstance(x, UniformGrid) and np.array(x).shape == (
            self.num_dimensions,
        ):
            # Either we have the point exactly on the grid, or we
            # are called with piecewise_constant=True
            if piecewise_constant or all(  # On each dimension, we
                [  # have it on the grid.
                    x[dim] in self.grid.coordinates_1d[dim]
                    for dim in range(self.num_dimensions)
                ]
            ):
                return self.data[tuple(self.grid.coordinates_to_indices(x))]

        # TODO: We can do better than this. If piecewise_constant is True
        #       we can avoid using splines. We only have to handle the ext.

        if ext not in (1, 2):
            raise ValueError("Only ext=1 or ext=2 are available")

        if self.invalid_spline:
            self._make_spline()

        # To make sure we give what the user asks we temporarly change the
        # method in the RegularGridInterpolator to 'nearest' (see documentation
        # in SciPy), if piecewise_constant is True, or 'linear', if it is
        # False.

        old_method = self.spline_real.method
        new_method = "nearest" if piecewise_constant else "linear"

        self.spline_real.method = new_method
        if self.is_complex():
            self.spline_imag.method = new_method

        # ext = 1 is setting to 0. We set fill_value to 0, so this is the
        # default behavior. We change the bounds_error attribute in
        # RegularGridInterpolator that controls this. By default, we set it
        # to raise an error. We reset it to True when we are done.
        if ext == 1:
            self.spline_real.bounds_error = False
            if self.is_complex():
                self.spline_imag.bounds_error = False

        if isinstance(x, UniformGrid):
            # The way we want the coordinates is like as an array with the same
            # shape of the grid and with values the coordines (as arrays). This
            # is similar to as_same_shape, but the coordinates have to be the
            # value, and not the first index.
            x = np.moveaxis(x.coordinates(as_same_shape=True), 0, -1)

        x = np.atleast_1d(np.array(x))

        y_real = self.spline_real(x)
        if self.is_complex():
            y_imag = self.spline_imag(x)
            ret = y_real + 1j * y_imag
        else:
            ret = y_real

        if ext == 1:
            self.spline_real.bounds_error = True
            if self.is_complex():
                self.spline_imag.bounds_error = True

        ret = np.atleast_1d(ret)

        # Restore the old method
        self.spline_real.method = old_method
        if self.is_complex():
            self.spline_imag.method = old_method

        # Was the input a single point? If yes we return a single value
        input_one_point = x.shape == (self.num_dimensions,)
        return ret[0] if input_one_point else ret

    def __call__(self, x):
        # TODO: Avoid splines when the data is already available
        return self.evaluate_with_spline(x)

    def resampled(self, new_grid, ext=2, piecewise_constant=False):
        """Return a new UniformGridData resampled from this to new_grid.

        You can specify the details of the spline with the method make_spline.

        If you want to resample without using the spline, and you want a nearest
        neighbor resampling, pass the keyword piecewise_constant=True.
        This may be a good choice for data with large discontinuities, where the
        splines are ineffective.

        :param new_grid: New independent variable
        :type new_grid:  1D numpy array or list of float
        :param ext: How to handle points outside the data interval
        :type ext: 1 for returning zero, 2 for ValueError,
        :param piecewise_constant: Do not use splines, use the nearest neighbors.
        :type piecewise_constant: bool
        :returns: Resampled series.
        :rtype:   :py:class:`~.UniformGridData` or derived class

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
        """Apply the method f to self, modifying self.
        This is used to transform the commands from returning an object
        to modifying self.
        The function has to return a new copy of the object (not a reference).
        """
        ret = f(*args, **kwargs)
        self.grid, self.data = ret.grid, ret.data
        # We have to recompute the splines
        self.invalid_spline = True

    def flat_dimensions_removed(self):
        """Return a new UniformGridData with dimensions of one grid
        point removed.

        :returns: New UniformGridData without flat dimensions.
        :rtype: :py:class:`UniformGridData`
        """
        new_grid = self.grid.flat_dimensions_removed()
        new_data = self.data.reshape(new_grid.shape)
        return type(self)(new_grid, new_data)

    def flat_dimensions_remove(self):
        """Remove dimensions which are only one gridpoint large."""
        self._apply_to_self(self.flat_dimensions_removed)

    def ghost_zones_removed(self):
        """Return a new UniformGridData witho all the ghost zones removed.

        :returns: New UniformGridData without ghostzones.
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
        """Remove ghost zones"""
        self._apply_to_self(self.ghost_zones_removed)

    def dx_changed(self, new_dx, piecewise_constant=False):
        """Return a new UniformGridData with the same grid extent, but with a new
        spacing. This effectively up-samples or down-samples the grid.

        Missing data is obtained with splines.

        new_dx has to be an integer multiple of the current dx (or vice versa).

        If piecewise_constant=True, the missing information is obtained with
        from the nearest neighbors.
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
            x1=self.x1,
            ref_level=self.ref_level,
            component=self.component,
            num_ghost=self.num_ghost,
            time=self.time,
            iteration=self.iteration,
        )

        return self.resampled(new_grid, piecewise_constant=piecewise_constant)

    def dx_change(self, new_dx, piecewise_constant=False):
        """Return a new UniformGridData with the same grid extent, but with a new
        spacing. This effectively up-samples or down-samples the grid.

        Missing data is obtained with splines.

        new_dx has to be an integer multiple of the current dx (or vice versa).

        If piecewise_constant=True, the missing information is obtained with
        from the nearest neighbors.
        """

        self._apply_to_self(
            self.dx_changed, new_dx, piecewise_constant=piecewise_constant
        )

    def copy(self):
        """Return a deep of self"""
        return type(self)(self.grid, self.data)

    @property
    def num_dimensions(self):
        """Return the number of dimensions."""
        return self.grid.num_dimensions

    @property
    def num_extended_dimensions(self):
        """Return the number of dimensions with more than one grid point."""
        return self.grid.num_extended_dimensions

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
        r"""Compute the norm over the whole volume of the grid.

        \|u\|_p = (\sum \|u\|^p dv)^1/p

        :returns: The norm2 computed as volume-weighted sum.
        :rtype:   float (or complex if data is complex).
        """
        return linalg.norm(np.ravel(self.data), ord=order) * self.grid.dv ** (
            1 / order
        )

    def norm2(self):
        r"""Compute the norm over the whole volume of the grid.

        \|u\|_2 = (\sum \|u\|^2 dv)^1/2

        :returns: The norm2 computed as volume-weighted sum.
        :rtype:   float (or complex if data is complex).
        """
        return self.norm_p(order=2)

    def norm1(self):
        r"""Compute the norm over the whole volume of the grid.

        \|u\|_1 = \sum \|u\| dv

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
        """1D Histogram of the data.
        :param weights:    the weight for each cell. Default is one.
        :type weights:     RegData or numpy array of same shape or None.
        :param min_value: Lower bound of data to consider. Default is data range.
        :type min_value: float or None
        :param max_value: Upper bound of data to consider. Default is data range.
        :type max_value: float or None
        :param num_bins:      Number of bins to create.
        :type num_bins:       integer > 1

        :returns: the positions of the data bins and the distribution.
        :rtype:   tuple of two 1D numpy arrays.
        """
        if self.is_complex():
            raise ValueError("Histogram only works with real data")

        if min_value is None:
            min_value = self.min()
        if max_value is None:
            max_value = self.max()

        if isinstance(weights, UniformGridData):
            weights = weights.data

        # Check that we have a numpy array or None
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
        can be used insted of fraction.

        :param fractions: list of fraction/absolute values
        :type fractions:  list or array of floats
        :param weights:    the weight for each cell. Default is one.
        :type weights:     UniformGridData or numpy array of same shape or None.
        :param relative:   whether fractions refer to relative or absolute count.
        :type relative:    bool
        :param min_value: Lower bound of data to consider. Default is data range.
        :type min_value: float or None
        :param max_value: Upper bound of data to consider. Default is data range.
        :type max_value: float or None
        :param num_bins:      Number of bins to create.
        :type num_bins:       integer > 1

        :returns: data values corresponding to the given fractions.
        :rtype:   1D numpy array
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
            # otherwise numpy complains
            hist_cumulative = 1.0 * hist_cumulative
            hist_cumulative /= hist_cumulative[-1]

        # TODO: Finish this

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

    def partial_derived(self, direction, order=1):
        """Return a UniformGriDatad that is the numerical order-differentiation of the
        present grid_data along a given direction. (order = number of
        derivatives, ie order=2 is second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of self.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int
        :param direction: Direction of the partial derivative
        :type direction: int

        :returns:  New UniformGridData with derivative
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
        """Return a list UniformGriDatad that are the numerical
        order-differentiation of the present grid_data along all the
        directions. (order = number of derivatives, ie order=2 is second
        derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of self.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int
        :param direction: Direction of the partial derivative
        :type direction: int

        :returns:  list of UniformGridData with partial derivative along the
        directions
        :rtype:    list of :py:class:`~.UniformGridData`

        """
        return [
            self.partial_derived(direction, order=order)
            for direction in range(self.num_dimensions)
        ]

    def partial_derive(self, dimension, order=1):
        """Return a UniformGriDatad that is the numerical order-differentiation of the
        present grid_data along a given direction. (order = number of
        derivatives, ie order=2 is second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of self.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int
        :param direction: Direction of the partial derivative
        :type direction: int

        :returns:  New UniformGridData with derivative
        :rtype:    :py:class:`~.UniformGridData`

        """
        self._apply_to_self(self.partial_derived, dimension, order=order)

    def _apply_unary(self, function):
        """Apply a unary function to the data.

        :param op: unary function.
        :type op:  function operating on a numpy array
        :returns:  result.
        :rtype:    :py:class:`~.UniformGridData`.

        """
        return type(self)(self.grid, function(self.data))

    def _apply_reduction(self, reduction):
        """Apply a reduction to the data.

        :param function: Function to apply to the series
        :type function: callable

        :return: Reduction applied to the data
        :rtype: float

        """
        # TODO: Turn this into a decorator

        return reduction(self.data)

    def _apply_binary(self, other, function):
        """This is an abstract function that is used to implement mathematical
        operations with other series (if they have the same grid) or
        scalars.

        _apply_binary takes another object that can be of the same type or a
        scalar, and applies function(self.data, other.data), performing type
        checking.

        :param other: Other object
        :type other: :py:class:`~.UniformGridData` or scalar
        :param function: Dyadic function
        :type function: callable

        :returns:  Return value of function when called with self and ohter
        :rtype:    :py:class:`~.UniformGridData`

        """
        # TODO: Turn this into a decorator

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


def sample_function_from_uniformgrid(function, grid):
    """Create a regular dataset by sampling a scalar function of the form
    f(x, y, z, ...) on a grid.

    :param function:  The function to sample.
    :type function:   A callable that takes as many arguments as the number
                      of dimensions (in shape).
    :param grid:   Grid over which to sample the function.
    :type grid:    :py:class:`~.UniformGrid`
    :returns:     Sampled data.
    :rtype:       :py:class:`~.UniformGridData`

    """
    if not isinstance(grid, UniformGrid):
        raise TypeError("grid has to be a UniformGrid")

    # The try except block checks that the function supplied has the correct
    # signature for the grid provided. If you try to pass a function that takes
    # too many of too few arguments, you will get a TypeError

    try:
        ret = UniformGridData(
            grid, np.vectorize(function)(*grid.coordinates(as_same_shape=True))
        )
    except TypeError as type_err:
        # Too few arguments, type_err = missing N required positional arguments: ....
        # Too many arguments, type_err = takes N positional arguments but M were given
        ret = str(type_err)

    # TODO: This is fragile way to do error parsing
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
    f(x, y, z, ...) on a grid.

    :param function:  The function to sample.
    :type function:   A callable that takes as many arguments as the number
                      of dimensions (in shape).
    :param shape: Number of sample points in each dimension.
    :type shape:  1d numpy array or list of int
    :param x0:    Minimum corner of regular sample grid.
    :type x0:     1d numpy array or list of float
    :param x0:    Maximum corner of regular sample grid.
    :type x0:     1d numpy array or list of float
    :returns:     Sampled data.
    :rtype:       :py:class:`~.UniformGridData`

    """
    grid = UniformGrid(shape, x0=x0, x1=x1, *args, **kwargs)
    return sample_function_from_uniformgrid(function, grid)


class HierarchicalGridData(BaseNumerical):
    """Data defined on mesh-refined grids, consisting of one or more regular
    datasets with different grid spacings, i.e. a mesh refinement hierachy. The
    grid spacings should differ by powers of two. Origins of the components
    have to shifted relative to each other only by multiples of the finest
    spacing. All the components are merged together, so there is one
    UniformGridData per refinement level.

    Important: ghost zone information may be discarded!
    TODO: Do not throw away ghost zones at the outer boundary

    Basic arithmetic operations are defined for this class, as well as
    interpolation and resampling. This class can be iterated over to get all
    the regular datasets, ordered by refinement level.

    """

    def __init__(self, uniform_grid_data):
        """
        :param data: list of regular datasets
        :type adat:  list of :py:class:`~.UniformGridData` instances.
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

    @staticmethod
    def _fill_grid_with_components(grid, components):
        """Given a grid, try to fill it with the components Return a UniformGridData
        and the indices that actually were used in filling the grid

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
        """Try to merge a list of UniformGridData instances into one, assuming they all
        have the same grid spacing and filling a regular grid completely.

        If the assumption is not verified, and some blank spaces are found, then
        it returns the input untouched. This is because there are real cases in
        which multiple components cannot be merged (if there are multiple
        refinement levels).

        This function always returns a list, even when the components are merged.
        In that case, the return value is a [UniformGridData].

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
        # We have to transform x.x0 in tuple because we cannot compare numpy
        # arrays directly for sorting.
        components_no_ghosts.sort(key=lambda x: tuple(x.x0))

        # Next, we prepare the global grid
        grid = merge_uniform_grids(
            [comp.grid for comp in components_no_ghosts]
        )

        merged_grid_data, indices_used = self._fill_grid_with_components(
            grid, components_no_ghosts
        )

        if np.amin(indices_used) == 1:
            return [merged_grid_data]

        return components

    def __getitem__(self, key):
        return self.grid_data_dict[key]

    def get_level(self, ref_level):
        if ref_level not in self.refinement_levels:
            raise ValueError(f"Level {ref_level} not avilable")
        if len(self[ref_level]) > 1:
            raise ValueError(
                f"Level {ref_level} has multiple patches"
                " get_level works only when there is one"
            )
        return self[ref_level][0]

    def iter_from_finest(self):
        """Supports iterating over the regular elements, sorted by refinement level.
        This can yield a UniformGridData or a list of UniformGridData when it
        is not possible to merge the grids.

        From the finest to the coarsest.

        Use this when you know that the data you are working with have single
        grids or grids that can be merged.

        """
        for ref_level, data in reversed(list(self.grid_data_dict.items())):
            for comp_index, comp in enumerate(data):
                yield ref_level, comp_index, comp

    def __iter__(self):
        """Iterate across all the refinement levels and components from the coarsest
        to the finest."""
        for ref_level, data in reversed(list(self.grid_data_dict.items())):
            for comp_index, comp in enumerate(data):
                yield ref_level, comp_index, comp

    def __len__(self):
        return len(self.refinement_levels)

    @property
    def refinement_levels(self):
        return list(self.grid_data_dict.keys())

    @property
    def all_components(self):
        all_components = []
        for comps in self.grid_data_dict.values():
            all_components.extend(comps)
        return all_components

    @property
    def num_finest_level(self):
        """Return the number of the finest refinement level.

        :returns: index of the finest level
        :rtype: int
        """
        return self.refinement_levels[-1]

    @property
    def finest_level(self):
        """Return the finest level, if it is a single grid.

        :returns: finest level
        :rtype: `:py:class:~UniformGridData`
        """
        return self.get_level(self.num_finest_level)

    @property
    def max_refinement_level(self):
        return self.num_finest_level

    @property
    def num_coarsest_level(self):
        """Return the number of the coarsest refinement level.

        :returns: index of the coarsest level
        :rtype: int
        """
        return self.refinement_levels[0]

    @property
    def coarsest_level(self):
        """Return the coarsest level, if it is a single grid.

        :returns: Coarsest level
        :rtype: `:py:class:~UniformGridData`
        """
        return self.get_level(self.num_coarsest_level)

    @property
    def first_component(self):
        """Return the first component of the coarsest refinement level

        :returns: First component of coarsest level
        :rtype: `:py:class:~UniformGridData`
        """
        return self[self.num_coarsest_level][0]

    @property
    def dtype(self):
        return self.first_component.dtype

    @property
    def shape(self):
        """Num components per each level.

        Eg, if data has three levels, with 1 component in the first, 2 in the second,
        and three in the fifth, shape will be {1: 1, 2: 2, 5: 3}

        shape is useful for quick high level comparison between two HierachicalGridData

        :rtype: dictionary
        """
        return {
            ref_level: len(comp)
            for ref_level, comp in self.grid_data_dict.items()
        }

    @property
    def x0(self):
        # We have multiple patches
        if len(self[self.num_coarsest_level]) != 1:
            raise ValueError(
                "Data does not have a well defined x0 "
                " (there are multiple patches)"
            )
        return self.first_component.x0

    @property
    def x1(self):
        # We have multiple patches
        if len(self[self.num_coarsest_level]) != 1:
            raise ValueError(
                "Data does not have a well defined x1"
                " (there are multiple patches)"
            )
        return self.first_component.x1

    def dx_at_level(self, level):
        """Return the grid spacing at the specified refinement level"""
        return self[level][0].dx

    @property
    def coarsest_dx(self):
        """Return the coarsest dx"""
        return self.dx_at_level(self.num_coarsest_level)

    @property
    def finest_dx(self):
        """Return the finest dx"""
        return self.dx_at_level(self.num_finest_level)

    @property
    def num_dimensions(self):
        return self.first_component.num_dimensions

    @property
    def num_extended_dimensions(self):
        return self.first_component.num_extended_dimensions

    @property
    def time(self):
        """The time of the coarsest refinement level"""
        return self.first_component.time

    @property
    def iteration(self):
        """The iteration of the coarsest refinement level"""
        return self.first_component.iteration

    def copy(self):
        """Return a deep copy.

        :returns:  Deep copy of the HierarchicalGridData
        :rtype:    :py:class:`~.HierarchicalGridData`
        """
        return type(self)(self.all_components)

    def __eq__(self, other):
        """Return a deep copy.

        :returns:  Deep copy of the HierarchicalGridData
        :rtype:    :py:class:`~.HierarchicalGridData`
        """
        if not isinstance(other, HierarchicalGridData):
            return False
        if self.shape != other.shape:
            return False

        return self.all_components == other.all_components

    def _finest_level_component_at_point_core(self, coordinate):
        """Return the number and the component index of the most
        refined level that contains the given coordinate assuming
        a valid input coordinate.
        """
        # We walk from the finest level to the coarsest. If we find the point,
        # re return it. If we find nothing, we raise error.
        for ref_level, comp, grid_data in self.iter_from_finest():
            if coordinate in grid_data.grid:
                return ref_level, comp

        raise ValueError(f"{coordinate} outside the grid")

    def finest_level_component_at_point(self, coordinate):
        """Return the number and the component index of the most
        refined level that contains the given coordinate.

        If the grid has multiple patches, the component index is
        returned, otherwise, only the finest level.

        :param coordiante: point
        :type coordinate: tuple or numpy array with the same dimension

        :returns: Most refined level (and component) that contains the coordinate.
        :rtype: int if there's only one component, or tuple of ints if there are
        multiple components.

        """
        if not hasattr(coordinate, "__len__"):
            raise TypeError(f"{coordinate} is not a valid point")

        if len(coordinate) != self.num_dimensions:
            raise ValueError(
                f"The input point has dimension {len(coordinate)}"
                f" but the data has dimension {self.num_dimensions}"
            )

        return self._finest_level_component_at_point_core(coordinate)

    def _evaluate_at_one_point(self, point, ext=2, piecewise_constant=False):

        level, comp = self.finest_level_component_at_point(np.array(point))

        return self[level][comp].evaluate_with_spline(
            point, ext=ext, piecewise_constant=piecewise_constant
        )

    def evaluate_with_spline(self, x, ext=2, piecewise_constant=False):
        """Evaluate the spline on the points x.

        Values outside the interval are set to 0 if ext=1, or a ValueError is
        raised if ext=2.

        This method is meant to be used only if you want to use a different ext
        for a specific call, otherwise, just use __call__.

        :param x: Array of x where to evaluate the series or single x
        :type x: 1D numpy array of float, or UniformGrid

        :param ext: How to deal values outside the bounaries. Values outside
                    the interval are set to 0 if ext=1,
                    or an error is raised if ext=2.
        :type ext:  bool

        :returns: Values of the series evaluated on the input x
        :rtype:   1D numpy array or float

        """

        if isinstance(x, UniformGrid):
            # The way we want the coordinates is like as an array with the same
            # shape of the grid and with values the coordines (as arrays). This
            # is similar to as_same_shape, but the coordinates have to be the
            # value, and not the first index.
            x = np.moveaxis(x.coordinates(as_same_shape=True), 0, -1)

        # If we consider the case that points is a single point
        points_arr = np.array(x)

        # We check that points has exactly the dimensions of the
        # data
        if points_arr.shape == (self.num_dimensions,):
            return self._evaluate_at_one_point(
                points_arr, ext=ext, piecewise_constant=piecewise_constant
            )

        # TODO: Vectorize this

        # To make sure that the return shape is the same as the input one
        # (important for uniformgriddata), we prepare a return array filled
        # with zeros and we overwrite each element calling
        # self._evaluate_at_point. We use itertools.product to loop over
        # aribtary number of dimensions to get the multi-index. This is not an
        # elegant solution.
        ret = np.zeros(points_arr.shape[:-1], dtype=self.dtype)
        for multi_index in itertools.product(
            *(range(num_points) for num_points in ret.shape)
        ):
            ret[multi_index] = self._evaluate_at_one_point(
                points_arr[multi_index],
                ext=ext,
                piecewise_constant=piecewise_constant,
            )

        # Was the input a single point? If yes we return a single value
        input_one_point = points_arr.shape == (self.num_dimensions,)
        return ret[0] if input_one_point else ret

    def __call__(self, x):
        return self.evaluate_with_spline(x)

    def to_UniformGridData(self, grid, resample=False):
        """Combine the refinement levels into a UniformGridData.

        Optionally resample the data with a multilinear resampling.
        """
        return UniformGridData(
            grid,
            self.evaluate_with_spline(grid, piecewise_constant=(not resample)),
        )

    def merge_refinement_levels(self, resample=False):
        """Combine all the available data and resample it on a provided
        UniformGrid with resolution of the finest refinement level.

        Optionally data from coarser refinement levels is resampled too (with a
        multilinear resampling)

        This can be a very expensive operation!

        """
        # finest_dx can have zero entries, for which a shape of 1 should
        # correspond. There can zero entries, we substitute them with -1, so
        # that we can identify them as negative numbers
        new_dx = np.array([dx if dx > 0 else -1 for dx in self.finest_dx])
        new_shape = ((self.x1 - self.x0) / new_dx + 1.5).astype(np.int64)
        new_shape = np.array([s if s > 0 else 1 for s in new_shape])

        new_grid = UniformGrid(
            new_shape,
            x0=self.x0,
            x1=self.x1,
            time=self.time,
            iteration=self.iteration,
        )
        return self.to_UniformGridData(new_grid, resample=resample)

    def _apply_to_self(self, f, *args, **kwargs):
        """Apply the method f to self, modifying self.
        This is used to transform the commands from returning an object
        to modifying self.
        The function has to return a new copy of the object (not a reference).
        """
        ret = f(*args, **kwargs)
        self.grid_data_dict = ret.grid_data_dict

    def _apply_binary(self, other, function):
        """Apply a binary function to the data.

        :param function: Function to apply to all the data in the various
        refinement levels
        :type function: callable

        :return: New HierarchicalGridData with function applied to the data
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        # We only know what how to h
        if isinstance(other, type(self)):
            if self.refinement_levels != other.refinement_levels:
                raise ValueError("Refinement levels incompatible")
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
        # Assume reduction is np.min, we want the real minimum, so we have to
        # take the reduction of the reduction
        return reduction(
            # Here we are accessing _apply_reduction, which is a protected
            # member, so we ignore potential complaints.
            # skipcq: PYL-W0212
            np.array(
                [
                    data._apply_reduction(reduction)
                    for data in self.all_components
                ]
            )
        )

    def _apply_unary(self, function):
        """Apply a unary function to the data.

        :param function: Function to apply to all the data in the various
        refinement levels :type function: callable

        :return: New HierarchicalGridData with function applied to the data
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        new_data = [function(data) for data in self.all_components]
        return type(self)(new_data)

    def _call_component_method(
        self, method_name, *args, method_returns_list=False, **kwargs
    ):
        """Call a method on each UniformGridData component and return
        the result as a HierarchicalGridDatax

        :param method_name: a string that identifies one of the methods in
        :py:class:`~.UniformGridData`

        :param method_returns_list: If True, the method is expected to return a
        list, one UniformGridData per dimension (e.g, gradient, coordiantes)
        :type method_returns_list: bool

        :return: New HierarchicalGridData with function applied to the data
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        if not isinstance(method_name, str):
            raise TypeError(
                f"method_name has to be a string (but it is {method_name})"
            )

        if not hasattr(self.first_component, method_name):
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

    def partial_derivated(self, direction, order=1):
        """Return a HierarchicalGridData that is the numerical order-differentiation of
        the present grid_data along a given direction. (order = number of
        derivatives, ie order=2 is second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of self.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int
        :param direction: Direction of the partial derivative
        :type direction: int

        :returns:  New HierarchicalGridData with derivative
        :rtype:    :py:class:`~.HierarchicalGridData`

        """
        return self._call_component_method(
            "partial_derived", direction, order=order
        )

    def gradient(self, order=1):
        """Return a list HierarchicalGridData that are the numerical
        order-differentiation of the present grid_data along all the
        directions. (order = number of derivatives, ie order=2 is second
        derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of self.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int
        :param direction: Direction of the partial derivative
        :type direction: int

        :returns:  list of HierarchicalGridData with partial derivative along
        the directions
        :rtype:    list of :py:class:`~.HierarchicalGridData`

        """
        return self._call_component_method(
            "gradient", method_returns_list=True, order=order
        )

    def partial_derive(self, direction, order=1):
        """Apply a numerical differentiatin along the specified direction.

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        The output has the same shape of self.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int
        :param direction: Direction of the partial derivative
        :type direction: int

        """
        return self._apply_to_self(
            self.partial_derivated, direction, order=order
        )

    def coordinates(self):
        """Return coordiantes a list of HierarchicalGridData.

        Useful for computations involving coordinates.

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
