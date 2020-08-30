SimDir
==============================

The :py:class:`~.SimDir` class provide easy access to simulation data. Most data
analysis start by using this object. `SimDir`` takes as input the top level
directory containing simulation data, and read and organizes the content.
``SimDir`` contains the index of all the information that is possible to extract
from the ASCII and hdf5 files. If there are restarts, ``SimDir`` will handle
them transparently.

Defining a SimDir object
------------------------

Assuming ``gw150914`` is the folder where a simulation was run. ``gw150914`` can
possibly contain multiple checkpoints and restarts.

.. code-block:: python

    import postcactus.simdir as sd

    sim = sd.SimDir("gw150914")

In case the directory structure is very deep (more than 8 levels), you can
specify the option ``max_depth`` to increase the default.

If you want to ignore specific folders (by default ``SIMFACTORY``, ``report``,
``movies``, ``tmp``, ``temp``), you can provide the ``ignore`` argument.

Using SimDir objects
--------------------

:py:class:`~.SimDir` classes are used to read and organize data. You can easily
access simulation data from the attributes of :py:class:`~.SimDir`.

For all the :py:class:`~.TimeSeries` (scalars and reductions, like maximum), you
can use

.. code-block:: python

    timeseries = sd.ts
    # or timeseries = sd.timeseries

The resulting object is a :py:class:`~.ScalarDirs`. The page
:ref:`cactus_scalars:Scalar data` contains a lot of information on how to use
these.

For the multipoles (documentation: :ref:`cactus_scalars:Working with multipolar
decompositions`):

.. code-block:: python

    multipoles = sd.multipoles
