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
