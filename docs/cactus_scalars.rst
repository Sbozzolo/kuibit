Scalar data
==============================

Scalar output are common in several simulations, with the most notable example
being reductions (max, min, average, ...). The module :py:mod:`~.cactus_scalars`
(:ref:`cactus_scalars_ref:Reference on kuibit.cactus_scalars`) handles these
quantities. Data is loaded lazily.

What data can be read?
-----------------------

:py:class:`~.OneScalar` reads files produced by ``CarpetASCII``. It
recognizes transparently ``gz`` and ``bz2`` compressed files and it works with
multiple variables in one file, or different files for each variable. In the
former case, :py:class:`~.OneScalar` reads the ``column format`` line in
the file and deduces the content. :py:class:`~.OneScalar` can return
a :py:class:`~.TimeSeries` with the time evolution of the various scalars.

Accessing data
--------------

One typically does not use directly :py:class:`~.OneScalar`, but
:py:class:`~.ScalarsDir`. This class takes as input a :py:class:`~.SimDir` and
organizes the various type of scalar data available. :py:class:`~.SimDir`
internally organizes its scalar data as :py:class:`~.ScalarsDir`, so this
documentation is of interest to the scalars in :py:class:`~.SimDir`.

You can also print the content of a :py:class:`~.ScalarsDir`:

.. code-block:: python

    import kuibit.cactus_scalars as cs
    import kuibit.simdir as sd

    sim = sd.SimDir("simulation")

    # The following three are equivalent
    timeseries = sim.ts
    timeseries = sim.timeseries
    timeseries = cs.ScalarsDir(sim)

    print(timeseries)

    # Extract of possible output:

    # Available norm1 timeseries:
    # ['H', 'M1', 'M2', 'M3', 'kxx', 'kxy', 'kxz', 'kyy', 'kyz', 'kzz', 'press', 'alp', 'eps', 'vel[0]', 'vel[1]', 'vel[2]', 'rho', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']

    # Available norm2 timeseries:
    # ['rho', 'M1', 'M2', 'M3', 'press', 'H', 'vel[0]', 'vel[1]', 'vel[2]', 'alp', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz', 'kxx', 'kxy', 'kxz', 'kyy', 'kyz', 'kzz', 'eps']

    # .....


The easieast way to access data is using the brackets operator, or using the
``get`` function. (You can also access reductions in the same way.)

.. code-block:: python

    rho_max = timeseries.maximum['rho']
    # or
    rho_max = timeseries.maximum.get('rho')

Yet another way is to use the ``.fields`` attribute:

.. code-block:: python

    rho_max = timeseries.maximum.fields.rho

Clearly, instead of ``maximum``, you can use any reduction you want. Use
``scalars`` for scalar values.

The return values of all these calls are :py:class:`~.TimeSeries`. The page
:ref:`series:Time and frequency series` has abundant information about these
objects.
