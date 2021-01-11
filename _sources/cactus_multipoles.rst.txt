Working with multipolar decompositions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gravitational waves are typically studied in terms of their multipolar
decompositions :math:`(l, m)`. In Einstein Toolkit, ``Multipoles`` is
responsable for computing these quantities, which can be read and analyzed by
:py:mod:`~.cactus_multipoles` (:ref:`cactus_multipoles_ref:Reference on
kuibit.cactus_multipoles`). Since the main application is gravitational waves, we
will use the word "detector" to mean "radius of the sphere where the multipoles
are computed".

Accessing multipole data
------------------------

:py:mod:`~.cactus_multipoles` has three main classes to work with multipoles.
The most basic one is :py:class:`~.MultipoleOneDet`. All the three different
classes can be printed to see what is the content.

.. note::

   The classes :py:class:`~.MultipoleOneDet` and :py:class:`~.MultipoleAllDets` are
   not designed to be initialized directly. They should be obtained using
   :py:class:`~.SimDir`.


MultipoleOneDet
_______________

:py:class:`~.MultipoleOneDet` represent the entire available multipolar
decomposition for one variable on one radius.

You can see the available values of l and m with the ``available_lm`` attribute.
:py:class:`~.MultipoleOneDet` is like a dictionary, so you can access the variables
with the bracket operator, alternatively you can call with parentheses:

.. code-block:: python

   # Assuming mdet is a MultipoleOneDet
   l2m2 = mdet(2,2)
   l2m2 = mdet[(2,2)]

These are :py:class:`~.TimeSeries` for the requested multipole. Conviently, you
can loop over the available multipoles:

.. code-block:: python

   for l, m, mult_ts in mdet:
       # do stuff


:py:class:`~.MultipoleOneDet` has a useful to method to operate on each single
multipole component and accumulate all the results. This is a convenient way to
"loop over all the monopoles" performing some operation. This is how, for
example, :py:meth:`~.get_strain` is computed. This method,
:py:meth:`~.MultipoleOneDet.total_function_on_available_lm` takes as input a function. This
function will be called with ``function(mp_timeseries, mult_l, mult_m,
mult_r)``, plus all the additional arguments and keyword arguments that are
passed to :py:meth:`~.MultipoleOneDet.total_function_on_available_lm`. Hence, the function has
to have a compatible signature. In case some of the quantities passed are not
used, you can always add a ``*args*`` to the argument of your function to
capture them.


MultipoleAllDets
________________

:py:class:`~.MultipoleAllDets` collects all the :py:class:`~.MultipoleOneDet`
for a given variable and multiple radii. ``available_lm`` can be used to see
what multipoles are available and ``radii`` to see which radii. In case you want
to check, you can use the method :py:meth:`~.has_detector(l, m, re)`.

:py:class:`~.MultipoleAllDets` is similar to class:`~.MultipoleOneDet` with the
:py:exception that the index is the radius and the return value is a
:py::py:class:`~.MultipoleOneDet`.

.. code-block:: python

   # Assuming mall is a MultipoleAllDets
   mul_r100 = mall(100)
   mul_r100 = mall[100]

``mull_r100`` is a :py:class:`~.MultipoleOneDet`, so to access a specific timeseries
you have to use another bracket or parentheses operator:

.. code-block:: python

   l2_m2_r100 = mall[100][(2,2)]

Once again, :py:class:`~.MultipoleAllDets` can be looped over, with the difference
that the loop is on the radii.

You can quickly obtain the outer most detector with the ``outermost`` attribute.
This returns a :py:class:`~.MultipoleOneDet`.


MultipolesDir
______________

:py:class:`~.MultipolesDir` organizes all the variables for which there's
multipole information available. The structure is similar to
:py:class:`~.ScalarsDir`: :py:class:`~.MultipolesDir` is a dictionary like
object and the keys are the names of the variables and the values are
:py:class:`~.MultipoleAllDets`. So we can see the three levels of multipoles:
:py:class:`~.MultipoleOneDet` is one variable, one radius;
:py:class:`~.MultipoleAllDets` is one variable, multiple radii;
:py:class:`~.MultipolesDir` is multiple variable, multiple radii.

:py:class:`~.MultipolesDir` is initialized by providing a :py:class:`~.SimDir`.
The class finds both ASCII file and h5 files with multipole information. These
files are read when needed, with h5 files having precedence. As in
:py:class:`~.ScalarsDir`, there are three ways to access data:

.. code-block:: python

   # Assuming mdir is MultipolesDir
   psi4 = mdir['Psi4']
   psi4 = mdir.get('Psi4')
   psi4 = mdir.fields.psi4

The return value is a :py:class:`~.MultipoleAllDets`, so to obtain a timeseries
for the :math:`l = 2, m = 2, r=100` monopole:

.. code-block:: python

   psi4_l2_m2_r100 = mdir['Psi4'][100][(2,2)]

Or, alternatively you can combine the other possiblities described.
