Working with horizons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module :py:mod:`~.cactus_horizon` (:ref:`cactus_horizons_ref:Reference on
kuibit.cactus_horizons`) can read output data produced by ``AHFinderDirect``
and ``QuasiLocalMeasures``.

The simplest way to access horizon data is via :py:class:`~.SimDir` with the
attribute :py:meth:`~.horizons`, which returns an object of type
:py:class:`~.HorizonsDir`. :py:class:`~.HorizonsDir` collects all the
information available from the ``AHFinderDirect`` and ``QuasiLocalMeasures``
thorns, including the shape of the horizons.

.. note::

   ``AHFinderDirect`` and ``QuasiLocalMeasures`` use different indexing systems.
   At the moment, ``kuibit`` cannot connect the two automatically, so an
   horizon is identified by both the numbers.

To access horizon information, use the bracket notation, for example, if ``sim``
is a :py:class:`~.SimDir`, ``sim.horizons[0, 1]`` will return the horizon with
QLM index 0 and AH index 1. The result of this operation is a
:py:class:`~.OneHorizon` object. This contains all the variables from both
``QuasiLocalMeasures`` and ``AHFinderDirect`` (the ``BHdiagnostics`` files) as
:py:class:`~.TimeSeries`. To access the QLM variables, you canuse the bracket
notation (e.g., ``hor['mass']``), to access the AH ones you can access them via
the ``ah`` attribute (e.g., ``hor.ah.area``, or ``hor.ah['area']``).

If you only need (or have) one of the two, you can access the relevant
information with :py:meth:`~.get_apparent_horizon` or
:py:meth:`~.get_qlm_horizon`.

You can access the shape of an horizon from :py:class:`~.OneHorizon` with the
the method :py:meth:`~.shape_at_iteration`. This returns three lists of
with the various 3D patches that form each horizon. In case you are only
interested in a project of the shape on a 2D plane or 1D axis, you
can use :py:meth:`~.shape_outline_at_iteration` and specify the ``cut``.
For example, if you want to look at the equatiorial plane, you would
set ``cut=(None, None, 0)``.

.. warning::

   No interpolation is performed, so results are not accurate when the cut
   is not along one of the major directions centered in the origin of the
   horizon.

VT data
-----------------------

``QuasiLocalMeasures`` can optionally output ``.vtk`` files which contain the
horizon mesh and some internal variables defined on such meshes (when the option
``QuasiLocalMeasures::output_vtk_every`` is positive). ``kuibit`` can parse
these files and represent variables defined on the horizons. The attribute
:py:meth:`~.vtk_available_iterations` returns a list with the iterations at
which VTK data is available and the method
:py:meth:`~.available_vtk_variables_at_iteration` returns a list with which
variables are available at the given iteration.

.. note::

   As a design choice, ``.vtk`` parsing in ``kuibit`` was developed with
   flexibility as opposed to speed. ``kuibit`` will always scan all the various
   files without assuming much about them. ``.vtk`` can be large, so if this turns
   out to be a performance bottleneck, please open an issue on the bug tracker.

You can access the variables with the methods :py:meth:`~.vtk_at_iteration`, which
returns a dictionary-like object with all the variables at the given iteration, or
with :py:meth:`~.vtk_variable_at_iteration`, which returns a specific variable.

There are two special variables, ``coordinates``, which is a list of the 3D
coordinates of each vertex that form the horizon mesh, and ``connectivity``,
which describes the faces of the horizon. The ``connectivity`` list is formed by
NumPy arrays of the form ``4 i1 i2 i3 i4``. The first number (4) indicates that
the mesh is formed by polygons with four sides. The other four numbers mean that
the vertices identified by those four numbers are joined together. For example,
``4 0 1 2 3`` means that the ``coordinates[0]``, ``coordinates[1]``,
``coordinates[2]``, and ``coordinates[3]`` form a face of the mesh. Then, each
variable is defined as 1D NumPy array with one value for each vertex.
