Various utilities
==============================

The :py:mod:`~.utils` module contains a series of useful functions that are used
around the code. For example, the module contains the basics of the logging
facility used. For a full list, see :ref:`gw_utils_ref:Reference on
kuibit.utils`.

Loggers
-------

It is important to be able to print out info and debug messages to monitor the
execution of the various modules. The :py:mod:`~.utils` module contains useful
function to do so based on the ``logging`` module that comes with Python. The
main functions are:

- :py:func:`~.get_logger`, which returns ``kuibit``'s' ``logger`` object that
  should be used for all the internal logging,
- :py:func:`~.set_verbosity`, which takes as input one of ``INFO`` or ``DEBUG``
  and set the corresponding level of output for the logger.

In practice, the way these functions should be used is as follows:

.. code-block:: python

   logger = get_logger()
   # ...
   logger.info("This is a message with general information")
   # ...
   logger.debug("This is a message with debug information")

Then, applications that use ``kuibit`` can choose what level of reporting they
want with :py:func:`~.set_verbosity` (importing the function from the module).

In ``kuibit``, we only use ``INFO`` or ``DEBUG``. For errors and warnings, we
raise an exception.
