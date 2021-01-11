Getting started with SimDir
==============================

The :py:class:`~.SimDir` class provide easy access to simulation data. Most data
analysis start by using this object. ``SimDir`` takes as input the top level
directory containing simulation data, and read and organizes the content.
``SimDir`` contains the index of all the information that is possible to extract
from the ASCII and HDF5 files. If there are restarts, ``SimDir`` will handle
them transparently.

Defining a SimDir object
------------------------

Assuming ``gw150914`` is the folder where a simulation was run. ``gw150914`` can
possibly contain multiple checkpoints and restarts.

.. code-block:: python

    import kuibit.simdir as sd

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

The resulting object is a :py:class:`~.ScalarsDir`. The page
:ref:`cactus_scalars:Scalar data` contains a lot of information on how to use
these.

For the multipoles (documentation: :ref:`cactus_multipoles:Working with multipolar
decompositions`):

.. code-block:: python

    multipoles = sd.multipoles

Some useful conventions
------------------------

It is useful to be aware of some conventions employed by ``kuibit``. If you
are reading this for the first time, you may skip this section, but we reccomend
you come back here once you gain familiarity with the code. This section will be
useful to you also in the case that you are extending ``kuibit``.

Class hierarchy
________________

``kuibit`` defines a large number of custum object types to represent in a
convinent way the simulation data. Some of these classes are not designed to be
initialized directly, but are created by other more general objects. In general,
we advise users to only directly define ``SimDir`` objects, and access
everything else from there. However, it is useful to know what is the hierarchy
because in case you are not sure of what an object is supposed to do you
can evaluate ``type(object)``.

The most abstract object is the :py:class:`~.SimDir` which takes a path and has
as attributes a collection ``*Dir`` objects.

``*Dir`` classes do the first high-level organization of the content of the
``SimDir`` with respect to a specific area. For example, we have
:py:class:`~ScalarsDir`, or :py:class:`~MultipolesDir`, or
:py:class:`~GravitationalWavesDir`. To organize means to create a dictionary for
easier access to the quantities. For example, in the case of
:py:class:`~MultipolesDir`, we create a dictionary where the keys are the
available variables.

At the step below, we have ``*All*`` classes, for example
:py:class:`~AllScalars`, here, there's a second round of organizing the
available data in dictionaries. The keys of these new dictionaries are a second
quantity that is logically varying. Continuing the example of the
:py:class:`~MultipolesDir`, the second level is :py:class:`~MultipolesAllDets`
that organizes the available multipolar decompositions for different radii for a
given variable (where the variable was the higher level key in
:py:class:`~MultipolesDir`).

Finally, we have the ``*One*`` objects, which are responsible of returning the
actual data requested. In the case of :py:class:`~MultipolesDir`, that would be
:py:class:`~MultipolesOneDet`, which returns the timeseries of a specific choice
of :math:`l, m` for a given variable at a given radius.

To see more clearly this hierarchy, consider the following code

.. code-block:: python

    # This contains all the available information on the simulation
    sim = sd.SimDir("gw150914")

    # This contains all the available information on multipoles
    sim.multipoles  # type -> MultipolesDir

    # This contains all the available information on the multipolar
    # decomposition for 'Psi4'
    sim.multipoles['Psi4']  # type -> MultipolesAllDets

    # This contains all the available information on the multipolar
    # decomposition for 'Psi4' at the radius r
    sim.multipoles['Psi4'][r] # type -> MultipolesOneDet

    # This is the timeseries of the (2, 2) mode of Psi4 at radius r
    # at all the available times
    sim.multipoles['Psi4'][r][(2, 2)]  # type -> TimeSeries

    # This is the timeseries of the (2, 2) mode of Psi4 at radius r
    # at time t
    sim.multipoles['Psi4'][r][(2, 2)](t)  # type -> float

Accessing data
______________

There are up to four ways to access data stored in an object. Let us assume that
``data`` is one of these classes, and the relevant physical quantity for which you
want to find the value is ``x`` (e.g., ``data`` is a time series and you are asking
what is the value at time ``x``, or ``data`` is a gravitational wave signal and
you are asking what is the associated timeseries as extracted by radius ``x``).

1. Using the brackets notation: ``y = data[x]``
2. Using the parentheses notation: ``y = data(x)``
3. Using the ``get`` method: ``y = data.get(x)``
4. Accessing the ``fields`` attribute: ``y = data.fields.x`` (``x`` is
   labelling different grid functions or variables)

Not all the objects implement all the different methods, and others implement
additional ones, so you should refer to the documentation to find what is
available.

Printing objects
________________

When in doubt, you can always try to ``print`` an object. Most classes will tell
you what they are storing.

