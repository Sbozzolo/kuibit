Working with timers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module :py:mod:`~.cactus_timers` (:ref:`cactus_timers_ref:Reference on
kuibit.cactus_timers`) can be used to read timing information from the output of
``Carpet``. Timers are useful to profile the code and individuate bottlenecks.

At the moment, only XML timers are supported. These files are output when the
option ``Carpet::output_xml_timer_tree`` is set to ``yes``.

Trees
-----------------------

The way timers are represented in ``kuibit`` is with the :py:class:`~.Tree`
structure. The tree structure is ideal because XML timers contains a report on
the call-stack of the simulation, which is naturally hierarchical: it is the set
of functions called, and which functions each function called, and so on. The
only functions that are profiled are the ones that are scheduled. In ``kuibit``,
a :py:class:`~.Tree` is a collection of three elements: a ``name``, a ``value``,
and possibly a collection of ``children``. For timers, the name is the name of
the function, the value is the total time that was spent inside this function,
and ``children`` is the set of functions that were called inside this one.

For example,

.. code-block:: python

   # Assuming tim is a Tree
   print(tim.name)  # main
   print(tim.value)  # 1.4 (seconds)

   # The name of the first child
   print(tim[0].name)  # evolve
   # Can also be called with
   print(tim["evolve"].name)  # evolve

   # This prints the cumulative value of all the leaves of the tree
   print(tim["evolve"].tot_value_leaves)

   # This can be used to transform the tree into percentual instead of seconds
   tim_in_perc = tim / tim.tot_value_leaves

   # Trees can be exported to dictionaries of JSON
   tim_dict = tim.to_dict()
   print(tim.to_json())

Timer trees
-----------------------

The easiest way to access timing information is starting from
:py:class:`~.SimDir`:

.. code-block:: python

   # Assuming sim is a SimDir

   tim = sim.timers


``tim`` is a :py:class:`~.TimersDir` object: a dictionary-like object that has
as keys the process numbers (the various MPI ranks), and as values a
:py:class:`~.Tree` with all the timing information (see above). ``kuibit``
automatically detects restarts and sums them up in a single tree. More often
than not, however, we are not interest in timers for a specific process, but we
want to have a general idea. In that case, we can use the :py:meth:`~.average`,
or the :py:meth:`~.median` methods to obtain the average (or median) timers
across all the processes.


.. note::

   Check out the examples to see a neat and useful application of this module.
