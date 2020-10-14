#!/usr/bin/env python3

"""The :py:mod:`~.grid_data` module provides representations of data on
uniform grids as well as for data on refined grid hirachies. Standard
arithmetic operations are supported for those data grids, further methods
to interpolate and resample. The number of dimensions is arbitrary.
Rudimentary vector and matrix oprations are also supported, using
Vectors of data grids (instead of grids of vectors).

The important classes defined here are
 * :py:class:`~.UniformGrid`  represents the geometry of a uniform grid.
 * :py:class:`~.UniformGridData`  represents data on a uniform grid.
 * :py:class:`~.CompData`   represents data on a refined grid hirachy.
 * :py:class:`~.Vec`        represents a generic vector
 * :py:class:`~.Mat`        represents a generic matrix
"""

import numpy as np
from scipy import interpolate

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
                f"The dimensions of this object are {self.shape}, not {var.shape} in {name}."
            )

    def __init__(
        self,
        shape,
        x0,
        dx=None,
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
        self.__origin = np.atleast_1d(np.array(x0, dtype=float))

        self._check_dims(self.shape, "shape")
        self._check_dims(self.origin, "origin")

        if dx is None:
            if x1 is None:
                raise ValueError("Must provide one between x1 and dx")

            # Here x1 is given, we compute dx. Consider the case
            # with three cells, origin = (0,0) and x1 = (4,4).
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
            self.__dx = (x1_arr - self.origin) / (temp_shape - 1)
            self.__dx[self.__dx < 0] = 0
        else:
            # Here we assume dx is given, but if also x1 is given, that
            # would may lead to problems if the paramters do not agree. So, we
            # first compute what x1 would be given origin and dx, then if x1
            # is provided, we compare the result with the given x1. We raise an
            # error if they disagree.
            self.__dx = np.atleast_1d(np.array(dx, dtype=float))
            self._check_dims(self.dx, "dx")
            expected_x1 = self.origin + (self.shape - 1) * self.dx
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

    @property
    def x0(self):
        return self.__origin

    @property
    def shape(self):
        return self.__shape

    @property
    def x1(self):
        return self.origin + (self.shape - 1) * self.dx

    @property
    def origin(self):
        return self.__origin

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

    def __getitem__(self, index):
        """Return the coordinates corresponding to a given (multi-dimensional)
        index.
        """
        index = np.array(index)
        self._check_dims(index, "index")
        return index * self.dx + self.origin

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

    def coordinates(self, as_meshgrid=False):
        """Return coordinates of the grid points.

        If as_meshgrid is True, the coordinates are returned as NumPy meshgrid.
        Otherwise, return the coordinates of the grid points as
        1D arrays (schematically, [array for x coordinates, array for y
        coordinates, ...])

        :param as_meshgrid: If True, return the coordinates as meshgrid.
        :type as_mesgrid: bool

        :returns:  A list of 1d arrays of coordinates
        along the different axes.
        :rtype:   list of numpy arrays with the same shape as grid

        """

        coordinates_1d = [
            np.linspace(x0, x1, n)
            for n, x0, x1 in zip(self.shape, self.x0, self.x1)
        ]

        if as_meshgrid:
            return np.meshgrid(*coordinates_1d)

        return coordinates_1d

    def flat_dimensions_remove(self):
        """Remove dimensions which are only one gridpoint across"""
        # We have to save this, otherwise it would be recomputed at every line,
        # affecting the result.
        extended_dims = self.extended_dimensions

        self.__shape = self.__shape[extended_dims]
        self.__origin = self.__origin[extended_dims]
        self.__dx = self.__dx[extended_dims]
        self.__num_ghost = self.__num_ghost[extended_dims]

    def copy(self):
        """Return a deep copy.

        :returns:  Deep copy of the UniformGrid
        :rtype:    :py:class:`~.UniformGrid`
        """
        return type(self)(
            self.shape,
            self.origin,
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
            and np.allclose(self.origin, other.origin, atol=1e-14)
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
x0/dx         = {self.x0/self.dx}
x1               = {self.x1}
x1/dx         = {self.x0/self.dx}
Volume           = {self.volume}
Dx            = {self.dx}
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
    if not hasattr(grids, '__len__'):
        raise TypeError("common_bounding_box takes a list")

    # Check that they are all UniformGrids
    if not all(isinstance(g, UniformGrid) for g in grids):
        raise TypeError(
            "common_bounding_boxes takes a list of UniformGrid"
        )

    # We have to check that the number of dimensions is the same
    num_dims = set([g.num_dimensions for g in grids])

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
    if not hasattr(grids, '__len__'):
        raise TypeError("common_bounding_box takes a list")

    if not all(isinstance(g, UniformGrid) for g in grids):
        raise TypeError("merge_uniform_grids works only UniformGrid")

    # Check that all the grids have the same refinement levels
    ref_levels = set([g.ref_level for g in grids])

    if len(ref_levels) != 1:
        raise ValueError("Can only merge grids on same refinement level.")

    # Extract the only element from the set
    ref_level = next(iter(ref_levels))

    dx = [g.dx for g in grids]

    if not np.allclose(dx, dx[0]):
        raise ValueError("Can only merge grids on with same dx.")

    # Find the bounding box
    x0, x1 = common_bounding_box(grids)

    # We add a little bit more than 1 to ensure that we don't accidentally lose
    # one point by rounding off (conversion to int always rounds down)

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
            raise ValueError("grid and data shapes differ")

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
        dx=None,
        x1=None,
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
            dx,
            x1,
            ref_level=ref_level,
            component=component,
            num_ghost=num_ghost,
            time=time,
            iteration=iteration,
        )
        return cls(geom, data)

    def _make_spline(self, *args, k=1, **kwargs):
        """Private function to make spline representation of the data using
        scipy.interpolate.RegularGridInterpolator.

        Only nearest neighbor or multilinear interpolations are available.

        This function is not meant to be called directly.

        :param k: Order of the interpolation (k = 0 or 1)
        :type k:  int

        """

        coords = self.grid.coordinates()

        if k != 0 and k != 1:
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

    def evaluate_with_spline(self, x, ext=2):
        """Evaluate the spline on the points x.

        Values outside the interval are set to 0 if ext=1, or a ValueError is
        raised if ext=2.

        This method is meant to be used only if you want to use a different ext
        for a specific call, otherwise, just use __call__.

        :param x: Array of x where to evaluate the series or single x
        :type x: 1D numpy array of float

        :param ext: How to deal values outside the bounaries. Values outside
                    the interval are set to 0 if ext=1,
                    or an error is raised if ext=2.
        :type ext:  bool

        :returns: Values of the series evaluated on the input x
        :rtype:   1D numpy array or float

        """
        # ext = 0 is extrapolation and ext = 3 is setting the boundary
        # value. We cannot do this with RegularGridInterpolator

        if not (ext == 1 or ext == 2):
            raise ValueError("Only ext=1 or ext=2 are available")

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

        # ext = 2 is raising an error.
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

        # When this method is called with a scalar input, at this point, ret
        # would be a 0d numpy scalar array. What's that? - you may ask. I have
        # no idea, but the user is expecting a scalar as output. Hence, we cast
        # the 0d array into at "at_least_1d" array, then we can see its length
        # and act consequently.
        ret = np.atleast_1d(ret)
        return ret if len(ret) > 1 else ret[0]

    def __call__(self, x):
        # TODO: Avoid splines when the data is already available
        return self.evaluate_with_spline(x)

    def is_complex(self):
        """Return whether the data is complex.

        :returns:  True if the data is complex, false if it is not.
        :rtype:   bool

        """
        return issubclass(self.data.dtype.type, complex)

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
        new_grid = self.grid.copy()
        new_grid.flat_dimensions_remove()
        new_data = self.data.reshape(new_grid.shape)
        return type(self)(new_grid, new_data)

    def flat_dimensions_remove(self):
        """Remove dimensions which are only one gridpoint large."""
        self._apply_to_self(self.flat_dimensions_removed)

    @property
    def num_dimensions(self):
        """Return the number of dimensions."""
        return self.grid.num_dimensions

    @property
    def num_extended_dimensions(self):
        """Return the number of dimensions with more than one grid point."""
        return self.grid.num_extended_dimensions

    # def histogram(
    #     self, weights=None, min_value=None, max_value=None, num_bins=400
    # ):
    #     """1D Histogram of the data.
    #     :param weights:    the weight for each cell. Default is one.
    #     :type weights:     RegData or numpy array of same shape or None.
    #     :param min_value: Lower bound of data to consider. Default is data range.
    #     :type min_value: float or None
    #     :param max_value: Upper bound of data to consider. Default is data range.
    #     :type max_value: float or None
    #     :param num_bins:      Number of bins to create.
    #     :type num_bins:       integer > 1

    #     :returns: the positions of the data bins and the distribution.
    #     :rtype:   tuple of two 1D numpy arrays.
    #     """
    #     if min_value is None:
    #         min_value = self.min()

    #     if max_value is None:
    #         max_value = self.max()

    #     if isinstance(weights, UniformGridData):
    #         weights = weights.data

    #     return np.histogram(
    #         self.data,
    #         range=(min_value, max_value),
    #         bins=num_bins,
    #         weights=weights,
    #     )

    # def percentiles(
    #     self,
    #     fractions,
    #     weights=None,
    #     relative=True,
    #     min_value=None,
    #     max_value=None,
    #     num_bins=400,
    # ):
    #     """Find values for which a given fraction(s) of the data is smaller.

    #     Optionally, the cells can have an optional weight, and absolute counts
    #     can be used insted of fraction.

    #     :param fractions: list of fraction/absolute values
    #     :type fractions:  list or array of floats
    #     :param weights:    the weight for each cell. Default is one.
    #     :type weights:     RegData or numpy array of same shape or None.
    #     :param relative:   whether fractions refer to relative or absolute count.
    #     :type relative:    bool
    #     :param min_value: Lower bound of data to consider. Default is data range.
    #     :type min_value: float or None
    #     :param max_value: Upper bound of data to consider. Default is data range.
    #     :type max_value: float or None
    #     :param num_bins:      Number of bins to create.
    #     :type num_bins:       integer > 1

    #     :returns: data values corresponding to the given fractions.
    #     :rtype:   1D numpy array
    #     """
    #     hist_values, bin_edges = self.histogram(
    #         min_value=min_value,
    #         max_value=max_value,
    #         num_bins=num_bins,
    #         weights=weights,
    #     )

    #     hist_cumulative = np.cumsum(hist_values)

    #     if relative:
    #         hist_cumulative /= hist_cumulative[-1]

    #     # TODO: FINISH HERE
    #     bin_edges = bin_edges[1:]
    #     fr = np.minimum(hc[-1], np.array(fractions))
    #     return np.array([hb[hc >= f][0] for f in fr])

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
                and np.allclose(
                    self.grid.origin, other.grid.origin, atol=1e-14
                )
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

    # TODO: Check the number of arguments taken by the function

    return UniformGridData(
        grid, np.vectorize(function)(*grid.coordinates(as_meshgrid=True))
    )


def sample_function(function, x0, x1, shape):
    """Create a regular dataset by sampling a scalar function of the form
    f(x, y, z, ...) on a grid.

    :param function:  The function to sample.
    :type function:   A callable that takes as many arguments as the number
                      of dimensions (in shape).
    :param x0:    Minimum corner of regular sample grid.
    :type x0:     1d numpy array or list of float
    :param x0:    Maximum corner of regular sample grid.
    :type x0:     1d numpy array or list of float
    :param shape: Number of sample points in each dimension.
    :type shape:  1d numpy array or list of int
    :returns:     Sampled data.
    :rtype:       :py:class:`~.UniformGridData`

    """
    grid = UniformGrid(shape, x0=x0, x1=x1)
    return sample_function_from_uniformgrid(function, grid)
