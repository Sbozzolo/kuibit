Grid functions
==============================

Other than series (time and frequency), grid functions are probably the other
most important quantity that we extract from simulations with the Einstein
Toolkit. In this page, we describe how to use ``kuibit`` to handle grids.
(:ref:`grid_data_ref:Reference on kuibit.grid_data`). The main object
that we use to represent grid function is the
:py:class:`~.HierarchicalGridData`. This represents data defined on a grid with
multiple refinement levels. On each level, data is represented as
:py:class:`~.UniformGridData`. While you will likely never initialize these
objects directly, it is useful to be aware of what they are and what they can
do. If you want to know how to read data into ``kuibit``, jump to the second
half of this page.

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

    import kuibit.grid_data as gd

    box = gd.UniformGrid(
                [101, 201],  # shape: 101 (201) points along the x (y) direction,
                [0, 0],  # origin, at 0, 0 (cell-centered)
                x1=[10, 20]  # other corner, at (10, 20)
                )

This is a two dimensional grid where the bottom left corner is (0, 0), and the
top right one is (10, 20). There are 101 points on the x direction and 201 on
the y. The grid is cell-centered, so the coordinates of the points will be the
integers. Instead of specifying the other corner with respect to the origin, you
can specify the size of each cell by providing the ``dx`` parameter.

.. code-block:: python

    box = gd.UniformGrid(
                [101, 201],  # shape: 101 (201) points along the x (y) direction,
                [0, 0],  # origin, at 0, 0 (cell-centered)
                dx=[1, 1]  # cell size
                )

:py:class:`~.UniformGrid` are used as part of grids with refinement levels, so
they can house additional information, like ``time``, ``num_ghost``,
``ref_level``. In most cases, it is not necessary to work directly with these
quantities.

Some useful attributes to know about :py:class:`~.UniformGrid` are:
- ``x0`` and ``x1`` return the two corners of the grid,
- ``dx`` or ``delta`` return the cell size,
- ``dv`` returns the volume of a cell, and ``volume`` returns the
total volume,
- ``num_dimensions`` returns how many dimensions are in the grid,
``num_extended_dimensions`` returns how many dimensions are in the grid with
more than one grid point.

When you initialize a grid with a flat dimension, you must specify ``x0`` and ``dx``
(you cannot do it by specifying ``x0`` and ``x1``, because there is no ``x1``!).
In general, prefer providing ``x0`` and ``dx`` instead of ``x0`` and ``x1``.

You can use the ``in`` operator to see if a coordinate is inside the grid.
The operation considers the size of the cell, for example

.. code-block:: python

    box = gd.UniformGrid([101, 201], [0, 0], delta=[1, 1])

    [5, 5] in box  # True
    [-1, 2] in box  # False

The :py:meth:`~.contains` is syntactic sugar for the same operation.

To obtain all the coordinates in the grid, you can use the
:py:meth:`~.grid_data.UnfiromGrid.coordinates` method. This can be used in three
different ways. When called with no arguments, the output is a list of 1D
arrays. Each of these arrays contains the coordinates along a fixed axis. For
example, for the 2D grid, the first array will be the x coordinates, the second
the y. Finally, with ``as_meshgrid=True``, the return value will be a NumPy
meshgrid. This is useful for plotting. When ``as_same_shape=True`` the return
value is a list of coordinates with the same shape of the grid itself, each
element of this list is the value of that coordinate over the grid. This last
one is the most useful way to do computations that involve the coordinates. You
can obtained the coordinate as a list of coordinates along each direction also
with the method :py:meth:`~.coordinates_1d`.

To obtain a coordinate from a multidimensional index, just use the bracket
operator (``box[i, j]``).

:py:class:`~.UniformGrid` may have dimensions that are only one point (e.g.,
when simulating a plane). We call ``extended_dimensions`` those that have more
than one grid point. You can return a new :py:class:`~.UniformGrid` with removed
all the dimensions that are not extended using the method
``flat_dimensions_removed``.

You return a new :py:class:`~.UniformGrid` with coordinates shifted with
:py:meth:`~.shifted`. You can also remove the ghost zones with
:py:meth:`~.grid_data.UnfiromGrid.ghost_zones_removed`. This will return a new
:py:class:`~.UniformGrid` with no ghost zones.

You can also print a :py:class:`~.UniformGrid` object to have a full overview
of the properties of the grid.

The functions :py:meth:`~.coordinates_to_indices` and
:py:meth:`~.indices_to_coordiantes` can be used to convert from indices to
coordinates for the considered grid. You can pass single points, or collection
of points. If you provide coordinates, the returned indices will be those of the
closest grid points.

UniformGridData
---------------

Once we have a grid, we can define data on it. :py:class:`~.UniformGridData`
packs together a :py:class:`~.UniformGrid` and data defined on it. This is the
most basic form of a grid function. There are two ways to define
:py:class:`~.UniformGridData`, first from a :py:class:`~.UniformGrid` and a
NumPy array with matching shape, or from the details of the grid along with
the data (again, as a NumPy array with matching shape):

.. code-block:: python

    box = gd.UniformGrid([101, 201], x0=[0, 0], delta=[1, 1])

    data = np.array([i * np.linspace(1, 5, 201) for i in range(101)])

    # First way
    ug_data1 = gd.UniformGridData(box, data)

    # Second way
    ug_data2 = gd.from_grid_structure(data, x0=[0, 0], delta=[1, 1])

:py:class:`~.UniformGridData` shares the same basic infrastructure as the
classes :py:class:`~.TimeSeries` and :py:class:`~.FrequencySeries` (they are
derived from the same abstract class :py:class:`~.BaseNumerical`). This means
that all the mathematical operations are defined, such as, adding two
:py:class:`~.UniformGridData`, or taking the exponential with ``np.exp``.

.. code-block:: python

    ug_data3 = np.exp(ug_data1) / ug_data2

Mathematical operations are performed only if the two
:py:class:`~.UniformGridData` have the same underlying grid structure.
:py:class:`~.UniformGridData` also support N-dimensional Fourier transforms with
the :py:meth:`~.fourier_transform` method.

:py:class:`~.UniformGridData` can be sliced to lower dimensional
:py:class:`~.UniformGridData`. To do this, use the meth:`~.slice` method. This
function takes a ``cut`` paramter which is a list of the same lenght as the
dimension of the data. The elements of ``cut`` are ``None`` for the dimensions you
want to keep and are the coordinate of where you want to slice. For example, if you
have 3D data and you want to only look at the line with ``x=1`` and ``y=2``, then,
``cut`` has to be ``[1, 2, None]``. You can cut in arbitrary places and optionally
enable the ``resample`` option to obtain the values with a multilinear interpolation
instead of approximating the point with the closest available.

As :py:class:`~.TimeSeries`, :py:class:`~.UniformGridData` can be represented as
splines (constant or linear). This means that the objects can be resampled or
can be called as normal functions. Computing splines is an expensive operation
that can take several seconds if the grid have thousands of points.

Splines allow you to use the :py:class:`~.UniformGridData` as a normal function.
Suppose ``rho`` is a grid function. You can either use the bracket operator to
find the value of ``rho`` corresponding to specific indices (``rho[i, j]``), or
you can call ``rho`` with the coordinate where you want to evalue it
(``rho(x)``). When there are flat dimensions, the only possible splines are with
nearest neighbors. You can use a multilinear interpolation on the extended by
removing the flat dimensions with :py:meth:`~flat_dimensions_remove`.

Some basic useful functions are :py:meth:`~.mean`, :py:meth:`~.integral`,
:py:meth:`~.norm1`, or :py:meth:`~.norm2`. In general, there's a
:py:meth:`~.norm_p`, computed as

.. :math:

   \| u \|_p = \left( \Delta v  \sum \|u \| \right)^{(1/p)}

with :math:`\Delta v` being the volume of a cell.

:py:class:`~.UniformGridData` can be derived along a direction with
:py:meth:`~.grid_data.UnfiromGridData.partial_derived`, or the gradient can be
:py:calculated with meth:`~.grid_data.UnfiromGridData.gradient`. In both cases,
:py:the order of the derivative can be specified. The derivative are numerical
:py:with finite difference. Derivative are second order accurate everywhere.

A convenient function is :py:meth:`~.sample_function`. This takes a multivariate
function (e.g., :math:`sin(x + y)`) and returns a :py:class:`~.UniformGridData`
sampling that function. If you already have the grid structure, you can use
:py:meth:`~.sample_function_from_uniformgrid`.

Another useful function is :py:meth:`~.histogram`, which can be used to compute
histograms of :py:class:`~.UniformGridData` with weights or without. Similarly,
one can compute percentiles with :py:meth:`~.percentiles`. The input of this
function can either be relative (percentuals, as 0.01, 0.5, or so, if you enable
``relative=True``), or the actual number of points.

You can resample the data to a new grid using the function
:py:meth:`~.grid_data.UniformGridData.resampled`, which takes as input a
:py:class:`~.UniformGrid` and returns a new :py:class:`~.UniformGridData`
resampled on the new grid. If the new grid is outside the old one, you can
either raise an error, of fill the points outside with zeros. This behavior is
controlled by the flag ``ext``. When ``ext=1``, zeros are returned, when it is
2, ``ValueError`` is raised. By default,
:py:meth:`~.grid_data.UniformGridData.resampled` uses a multilinear
interpolation, but you can force to use a piecewise constant interpolation with
the nearest neighbors by setting ``piecewise_constant=True``.

Another useful feature is to :py:meth:`~.dx_changed`, which can be used to
return a new :py:class:`~.UniformGridData` with different grid spacing. The new
grid spacing has to be an integer multiple or an integer factor of the old one.
With this function you can upsample or downsample data. This is especially
useful when dealing with refinement levels, which typically have spacing related
by factors of 2. :py:meth:`~.dx_changed` takes an optional argument
``piecewise_constant`` to prescribe how the resampling should be done.

Often, it is useful to save a :py:class:`~.UniformGridData` and read it later.
:py:class:`~.UniformGridData` can be saved as ASCII files with the
:py:meth:`save` method, which takes a path and writes an ASCII file to that
destination. The file contains a header that specifies the grid information. The
data is always saved as as 1D array (due to the limitations of the backend).
These files can be read with the :py:meth:`~.load_UniformGridData` function. For
large datasets, it is convinent to compress the file. To do this, just provide a
file extension that is compressed (e.g., ``.dat.gz``).

To access the data (ie, for plotting), you can simply use ``.data``. This is a
standard numpy array. Alternatively, you can use the ``.data_xyz`` attribute,
which swaps rows and columns (``.data_xyz`` is coordinates-indexed, ``.data`` is
matrix-indexed).

.. warning::

   Arrays are stored row-first, so if you want to use ``.data``, to have a
   natural mapping between coordinates and indices you have to transpose the
   data! (See, `this blog post
   <https://eli.thegreenplace.net/2014/meshgrids-and-disambiguating-rows-and-columns-from-cartesian-coordinates/>`_
   for an explanation.)


HierarchicalGridData
--------------------

A :py:class:`~.HierarchicalGridData` represents data defined on a mesh-refined
grid. In practice, this is a collection of :py:class:`~.UniformGridData`,
roughly one per level. You can work directly with the
:py:class:`~.UniformGridData` on the different levels using the brackets
operator. As for :py:class:`~.UniformGridData` supports all the mathematical
operations.

In many cases, one works with a nested series of refinement levels, with a
domain that is split in multiple patches. Hence, the output data will also be in
multiple chunks. When initializing an :py:class:`~.HierarchicalGridData`, kuibit
will make an effort to put all the different patches back together. If the
provided components cover an entire grid, kuibit will merge them. In doing this,
all the ghost zone information is discarded. If kuibit finds that the provided
components do not cover a regular grid, then it will leave them untouched. This
is the case when one has multiple refinement centers (for example in binary
simulations). :py:class:`~.HierarchicalGridData` is essentially a dictionary
that maps refinement level to lists of :py:class:`~.UniformGridData` that
represent the different patches. In case kuibit manages to combine all the
patches, then the list will have only one element. You can print a
:py:class:`~.HierarchicalGridData` to see what the structure looks like:

.. code-block:: python

    print(rho)

    # The output will look like
    #
    # Available refinement levels (components):
    # 0 (1)
    # 1 (3)
    # 2 (2)
    # 3 (2)
    # Spacing at coarsest level (0): [640. 640.]
    # Spacing at finest level (3): [0.01 0.01]

You can access the relative level using the bracket operator (e.g. ``rho[0][0]``
is ``rho`` on the coarsest level on the 0th patch, which could be the only one).
The two level of brackets are (in order): refinement level, then component. In
many cases, the grid structure is simple and there are no multiple refinement
centers, so one can access the level with `:py:meth:~.get_ref_level`. This
method will work only if there's a single component.

As for :py:class:`~.UniformGridData`, :py:class:`~.HierarchicalGridData` are
callable and splines are used to interpolate to the requested points. This
operation can be expensive, especially for 3D grids with many points.
The way calling works is the following: we find the finest
refinement level that contains the requested point, and we use the multilinear
interpolation on that level (and component, if there are multiple components).

Using splines, we can also combine the various refinement levels to obtain a
:py:class:`~.UniformGridData`. This is often handy when plotting. The method
:py:meth:`~.merge_refinement_levels` does exactly that. By default,
:py:meth:`~.merge_refinement_levels` does not resample the data, but simply uses
the values on the grid. If the argument ``resample`` is set to ``True``, the
data is resampled with a multilinear interpolation. One can also specify what
grid (as :py:class:`~.UniformGridData`) to merge the data on by calling the
method :py:meth:`~.to_UniformGridData` or
:py:meth:`~.to_UniformGridData_from_grid`. This is especially useful when
resampling on smaller grids, because it drastically reduces the computation
time.

.. warning::

   Operations that involve resampling can be very expensive and require a lot
   of memory!

Another useful method is the
:py:meth:`~.grid_data.HierarchicalGridData.coordinates`, which returns a list of
:py:class:`~.HierarchicalGridData` with the same structure as the one in
consideration but with values the various coordinates at the points. This is
useful for computations that involve the coordinates.

As it is the case for :py:class:`~.UniformGridData`, also
:py:class:`~.HierarchicalGridData` can be derived along a direction with
:py:meth:`~.grid_data.HierarchicalGridData.partial_derived`, or the gradient can
be calculated with :py:meth:`~.grid_data.HierarchicalGridData.gradient`. In both
cases, the order of the derivative can be specified. The derivative are
numerical with finite difference. The result is a
:py:class:`~.HierarchicalGridData` or a list of
:py:class:`~.HierarchicalGridData` (for each direction).

Reading data
------------

So far, we have discussed how grid functions are represented in ``kuibit``.
In this section, we discuss how to read the output data from simulations as
:py:class:`~.HierarchicalGridData` or :py:class:`~.UniformGridData`.

At the moment, ``kuibit`` fully support reading HDF5 files of any dimension
(1D, 2D, and 3D). ``kuibit`` can also read ASCII files, but the interface is
less robus and not as well-tested.

.. warning::

   ``kuibit`` works better with HDF5 data. In general, reading and parsing
   HDF5 is orders of magnitude faster than ASCII data. ``kuibit`` can read
   one iteration at the time in HDF5 data, but has to read the entire content of
   all the files when the data is ASCII. This can take a long time. HDF5 are
   also much more storage-efficient and contain metadata that can be used to
   better interpret the data (e.g., the number of ghost zones). For these
   reasons, we strongly recommend using HDF5 files.

.. warning::

   The ASCII reader should be considered experimental. If reads the files line
   by line and will likely not fail if the data is not exactly in the format
   that the reader expect. You may find unexpected results. If you use the ASCII
   reader, make sure to test it!

.. warning::

   The ASCII reader works by scanning all the files line by line. This can take an
   extremely long time if you have many files with a lot of iterations. If you want
   to speed up the process, consider isolating the files you are interested in
   working with in a separate directory, and run ``SimDir`` in that folder.

From SimDir
^^^^^^^^^^^

The easiest way to access grid data is from :py:class:`~.SimDir`.
:py:class:`~.SimDir` objects contain an overview of the entire data content of a
directory. For more information about :py:class:`~.SimDir`, read
:ref:`simdir:Getting started with SimDir`.

Assuming ``sim`` is a :py:class:`~.SimDir`, the access point to grid functions is
in `sim.gf` or ``sim.grid_functions``. You can find all the available variables just
by printing this object

.. code-block:: python

    print(sim.gf)

    # The output will look like
    #
    # Available grid data of dimension 1D (x):
    # ['P', 'rho', 'rho_star', 'vz', 'Bz', 'By', 'vx', 'rho_b', 'vy', 'Bx']
    #
    # ... and so on ...

`sim.gf` is an object of type :py:class:`~.GridFunctionsDir`. The main role of
this class is to organize the available files depending on their dimensions. So,
from :py:class:`~.GridFunctionsDir` you can specify what dimensions you are
interested in. You can do this in two ways, as a dictionary call, or via an
attribute. For example, if you are interested in 2D data on the xy plane:

.. code-block:: python

    # All these methods are equivalent
    data2d = sim.gf.xy
    data2d = sim.gf['xy']
    data2d = sim.gf[(0, 1)]

In case you want a lower dimensional cut (say, you want only the y axis and you
have the xy data), you can always look at higher-dimensional data and slice it
to your liking, as described in the above sections.

Once you selected the data you are interested in, you will be working with a
:py:class:`~.AllGridFunctions` object. This is a dictionary-like object that
organizes all the variables available for the requested dimensions. You can
access the variables using the bracket operator of looking in the ``fields``
attribute. In case a variable is available as HDF5 file and as ASCII file, the
HDF5 representation is preferred.

.. code-block:: python

    # These methods are equivalent
    rho = sim.gf.xy['rho']
    rho = sim.gf.xy.fields.rho

In case you are reading an ASCII file, you have to set the correct number of
ghost zones. The simplest way to do this is to set the :py:meth:`~.num_ghost`
attribute. If the output does not contain ghost zones, set them to zero.

.. code-block:: python

    # If rho_star is from an ASCII file, we want to set num_ghost before
    # reading it
    ASCII_reader = sim.gf.xy
    ASCII_reader.num_ghost = (3, 3)
    rho_star = ASCII_reader.rho_star

:py:meth:`~.num_ghost` has to be a tuple or a list with the same number of entries
as the dimensionality of the grid: each entry is the number of ghost zones along
a direction.

.. warning::

   ASCII files do not have information about how many ghost zones are in the
   data, so we will assume that there are none. This can lead to imperfect
   results in the regions of overlap between two grid patches. In the future, we
   will try to read this value from the parameter file.


Finally, once you selected the variable, you will have a
:py:class:`~.OneGridFunctionH5` or :py:class:`~.OneGridFunctionASCII` object.
These are derived from the same base class :py:class:`~.OneGridFunctionBase` and
share the interface. The main difference is how files are read (which justifies
why we need to different classes). These objects are certainly the most
interesting ones and the ones you will deal with most of the time.

At first level, :py:class:`~.OneGridFunctionH5` (we will consider this for
definiteness, but the most of what said here holds true for
:py:class:`~.OneGridFunctionASCII`) is another dictionary-like object. The keys
of this class are the various iterations available in the files. Hence, to read
some data at a given iteration ``iteration``, you can simply use the bracket
operator. Alternatively, you can use the :py:meth:`~.get_iteration` method:

.. code-block:: python

    # These methods are equivalent
    rho0 = sim.gf.xy.rho[0]
    rho0 = sim.gf.xy.rho.get_iteration(0)

You can find what iterations are available with the
:py:meth:`~.available_iterations` attribute. Similarly, you can find what times
are available with :py:meth:`~.available_times`:

.. code-block:: python

    print(sim.gf.xy.rho.available_iterations)
    print(sim.gf.xy.rho.available_times)

You can read a time instead of a iteration with the method
:py:meth:`~.get_time`. You can convert between time and iteration with the
methods :py:meth:`~.time_at_iteration` and :py:meth:`~.iteration_at_time`.

These methods return a :py:class:`~.HierarchicalGridData` object with all the
available data for the requested iteration. If HDF5 files are being read, the
correct ghost zone information is being used. In case you want to work with a
specific subgrid with uniform spacing, you can use the :py:meth:`~.read_on_grid`
method. This will return a :py:class:`~.UniformGridData` object instead, with
grid the grid you specify. The grid is specified by passing a
:py:class:`~UniformGrid` object. For example

.. code-block:: python

    from kuibit.grid_data import UniformGrid

    grid = UniformGrid([100, 100], x0=[0, 0], x1=[2,2])
    rho0_center = sim.gf.xy.rho.read_on_grid(0, # iteration
                                             grid)

This method works by reading the entire grid structure and resampling onto the
requested :py:class:`~.UniformGridData`, so it may be slow for large 3D data.

Similarly, you can read a chunk of evolution from ``min_iteration`` to
``max_iteration`` on a specified grid with the method
:py:meth:`~.read_evolution_on_grid`. This returns a
:py:class:`~.UniformGridData` that has as first dimension the time, and as other
dimensions the specified grid. So, this is a "spacetime"
:py:class:`~.UniformGridData`. With this function you can evaluate grid data on
specific spacetime points with multilinear interpolation in space and time. This
can also be used to generate additional time frames between two outputs.

:py:class:`~.OneGridFunctionH5` objects are iterable: you can loop over all
the available iterations by iterating over the object.
