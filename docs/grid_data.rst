Grid functions
==============================

Other than series (time and frequency), grid functions are probably the other
most important quantity that we extract from simulations with the Einstein
Toolkit. In this page, we describe how to use ``PostCactus`` to handle grids.
(:ref:`frequencyseries_ref:Reference on postcactus.grid_data`).

UniformGrid
---------------

The most basic concept that we need to work with grid function is the concept of
:py:class:`~.UniformGrid`, which represents an uniform cell-centered Cartesian
grid in arbitrary number of dimensions. An object of type
:py:class:`~.UniformGrid` is immutable and is defined by the location of the
origin of the grid, by the number of points along each dimension, and by either
the opposite corner or the spacing. :py:class:`~.UniformGrid` are important
because they are are the building blocks of grids with refinement levels and
because they are the most natural grid to plot.

Let's see how to define a :py:class:`~.UniformGrid`:

.. code-block:: python

    import postcactus.grid_data as gd

    box = gd.UniformGrid(
                [101, 201],  # shape: 101 (201) points along the x (y) direction,
                [0, 0],  # origin, at 0, 0 (cell-centered)
                x1=[10, 20]  # other corner, at (10, 20)
                )

This is a two dimensional grid where the bottom left corner is (0, 0), and the
top right one is (10, 20). There are 101 points on the x direction and 201 on
the y. The grid is cell-centered, so the coordinates of the points will be the
integers. Instead of specifying the other corner with respect to the origin, you
can specify the size of each cell by providing the ``delta`` parameter.

.. code-block:: python

    box = gd.UniformGrid(
                [101, 201],  # shape: 101 (201) points along the x (y) direction,
                [0, 0],  # origin, at 0, 0 (cell-centered)
                delta=[1, 1]  # cell size
                )

:py:class:`~.UniformGrid` are used as part of grids with refinement levels, so
they can house additional information, like ``time``, ``num_ghost``,
``ref_level``. In most cases, it is not necessary to work directly with these
quantities.

Some useful attributes to know about :py:class:`~.UniformGrid` are:
- ``x0`` and ``x1`` return the two corners of the grid,
- ``delta`` or ``dx`` return the cell size,
- ``dv`` returns the volume of a cell, and ``volume`` returns the
  total volume,
- ``num_dimensions`` returns how many dimensions are in the grid,
``num_extended_dimensions`` returns how many dimensions are in the grid with
more than one grid point.

You can use the ``in`` operator to see if a coordinate is inside the grid.
The operation considers the size of the cell, for example

.. code-block:: python

    box = gd.UniformGrid([101, 201], [0, 0], delta=[1, 1])

    [5, 5] in box  # True
    [-1, 2] in box  # False

The :py:meth:`~.contains` is syntactic sugar for the same operation.

To obtain all the coordinates in the grid, you can use the
:py:meth:`~.coordinates` method. This can be used in three different ways. When
called with no arguments, it returns a list of multidimensional numpy arrays,
one for each dimension. Each of these arrays are the coordinates for a fixed
dimension of all the points on the grid (so the array has the same shape as the
grid). For example, for a two dimensional grid, :py:meth:`~.coordinates` will
return a list with two arrays, the first contains the x coordinates of all the
points, the second all the y coordinates. The second way to use the method is to
call it with ``as_1d_arrays=True``. In this case the output is a list of 1D
arrays. Each of these arrays contains the coordinates along a fixed axis. For
example, for the 2D grid, the first array will be the x coordinates, the second
the y. Finally, with ``as_meshgrid=True``, the return value will be a NumPy
meshgrid. This is useful for plotting.

To obtain a coordinate from a multidimensional index, just use the bracket
operator (``box[i, j]``).

:py:class:`~.UniformGrid` may have dimensions that are only one point (e.g.,
when simulating a plane). We call ``extended_dimensions`` those that have more
than one grid point. You can remove all the dimensions that are not extended
using the method ``flat_dimensions_remove``.

You can also print a :py:class:`~.UniformGrid` object to have a full overview
of the properties of the grid.
