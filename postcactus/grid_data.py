#!/usr/bin/env python3

"""The :py:mod:`~.grid_data` module provides representations of data on
uniform grids as well as for data on refined grid hirachies. Standard
arithmetic operations are supported for those data grids, further methods
to interpolate and resample. The number of dimensions is arbitrary.
Rudimentary vector and matrix oprations are also supported, using
Vectors of data grids (instead of grids of vectors).

The important classes defined here are
 * :py:class:`~.RegularGeometry`  represents the geometry of a uniform grid.
 * :py:class:`~.RegData`    represents data on a uniform grid and the
   geometry.
 * :py:class:`~.CompData`   represents data on a refined grid hirachy.
 * :py:class:`~.Vec`        represents a generic vector
 * :py:class:`~.Mat`        represents a generic matrix
"""

import numpy as np


class RegularGeometry:
    """Describes the geometry of a regular rectangular dataset, as well as
    information needed if part of refined grid hierachy, namely component
    number and refinement level. In practice, this a fixed refinement level.

    Also stores the number of ghost zones, which is however not used anywhere in
    this class.

    This is a standard Cartesian grid that we will describe with the language of
    computer graphics. To make things clear, let's consider a 2D grid (see
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

    """

    def __init__(
        self,
        shape,
        origin,
        delta=None,
        x1=None,
        reflevel=-1,
        component=-1,
        nghost=None,
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
        :param reflevel:  Refinement level if this belongs to a hierachy,
                          else -1.
        :type reflevel:   int
        :param component: Component number if this belongs to a hierachy,
                          else -1.
        :type component:  int
        :param nghost:    Number of ghost zones (default=0)
        :type nghost:     1d numpy arrary or list of int.
        :param time:      Time if that makes sense, else None.
        :type time:       float or None
        :param iteration: Iteration if that makes sense, else None.
        :type iteration:  float or None

        """
        self.shape = np.array(shape, dtype=int)
        self.origin = np.array(origin, dtype=float)

        self._check_dims(self.__shape, "shape")
        self._check_dims(self.__origin, "origin")

        if delta is None:
            cx1 = np.array(x1, dtype=float)
            self._check_dims(cx1, "x1")
            self.delta = (cx1 - self.__origin) / (self.__shape - 1)
        else:
            if x1 is not None:
                raise ValueError("RegGeom: specified both x1 and delta")
            #
            self.delta = np.array(delta, dtype=float)
            self.check_dims(self.__delta, "delta")
        #
        if nghost is None:
            self.nghost = np.zeros_like(self.__shape)
        else:
            self.nghost = np.array(nghost, dtype=int)
            self._check_dims(self.nghost, "nghost")
        #
        self.reflevel = int(reflevel)
        self.component = int(component)
        self.time = None if time is None else float(time)
        self.iteration = None if iteration is None else int(iteration)

    #
    def __check_dims(self, var, name):
        if len(var.shape) != 1:
            raise ValueError(
                "RegGeom: %s must not be multi-dimensional." % name
            )
        #
        if len(var) != len(self.__shape):
            raise ValueError("RegGeom: %s and shape dimensions differ." % name)
