Masking invalid data
==============================

It is often useful to ignore specific pieces of data. For example, it is wise to
exclude the atmosphere when we compute the maximum temperature GRMHD
simulations. For this, ``kuibit`` inherits from NumPy the concept of masks:
masked data carries along the information of where the data is valid and where
it is not. In ``kuibit``, classes derived from :py:class:`~.BaseNumerical`
(mainly, :py:class:`~.TimeSeries`, :py:class:`~.FrequencySeries`,
:py:class:`~.UniformGridData`, :py:class:`~.HierarchicalGridData`) support
masks, meaning that operations like :py:meth:`~.max` will not include the data
marked as invalid. In this page we describe how to work with masks
(:ref:`series_ref:Reference on kuibit.masks`).

Creating masked objects
------------------------------------------

Since the interface for is the same for all the classes defined in kuibit, we
will consider a :py:class:`~.TimeSeries` as an example.

To create a masked object, you first need to start from the clean version.
Suppose ``ts`` is a :py:class:`~.TimeSeries`, there are multiple ways to return
a new object ``ts_masked``:

.. code-block:: python

     # Data is invalid when it is equal to 1
     ts1 = ts.masked_equal(1)

     # Data is invalid when it is larger than 2
     ts2 = ts.masked_greater(2)

     # Data is invalid when it is larger or equal than 3
     ts3 = ts.masked_greater_equal(3)

     # Data is invalid when it is between 4 and 5
     ts4 = ts.masked_inside(4, 5)

     # Data is invalid when it is NaN or inf
     ts5 = ts.masked_invalid()

     # Data is invalid when it is larger than 6
     ts6 = ts.masked_less(6)

     # Data is invalid when it is larger or equal than 7
     ts7 = ts.masked_less_equal(7)

     # Data is invalid when it is not 8
     ts8 = ts.masked_not_equal(8)

     # Data is invalid when it is outside the range (8,9)
     ts9 = ts.masked_outside(8, 9)

All these methods return new objects. Alternatively, it is possible to edit the
object in place using methods with the imperative form (e.g., ``mask_equal`` instead
of ``masked_equal``).

The second way to create masked objects is by using the functions in
:py:mod:`~.masks`, which contains methods for mathematical functions that are
defined on a domain. For instance, if you want to compute the natural logarithm
of some data, you can use the function :py:func:`masks.log`, which automatically
applies a mask where the operation is not defined.

.. code-block:: python

     import kuibit.masks as ma

     log_ts = ma.log(ts)

     log_ts.is_masked()  # => True

The method :py:meth:`~.is_masked` checks whether the object is masked or not.
When objects are masked, some methods become unavailable. For example, it is not
possible to compute splines or perform interpolations. For
:py:class:`~.TimeSeries` and :py:class:`~.FrequencySeries`, you can go around
this limitation by removing the invalid points with the methods
:py:meth:`~.mask_remove` or :py:meth:`~.mask_removed`. This is not possible with
grid data because we assume that the data is defined on regular grids.

Suppose you want to mask the atmosphere. You have your density variable ``rho``
and you want to remove everything that is below ``1e-8``, and you want to plot
the pressure ``press``. For that you would first construct a masked grid data
with ``rho``, and then apply the mask to ``press``:

.. code-block:: python

     masked_rho = rho.masked_less(1e-8)
     masked_press = press.mask_applied(masked_rho.mask)

If you want to plot this with :py:mod:`~.visualize_matplotlib`, you need to pay
attention that resampling erases mask information. Therefore, if you want to
plot the mask, you have to pass directly the :py:class:`~.UniformGridData` you
want to plot (and not :py:class:`~.HierarchicalGridData`).

.. warning::

   We only mask the data, not the independent variable (e.g., the time in
   :py:class:`~.TimeSeries`). If your computations required this variable to be
   masked too, you should extract the mask array with the :py:meth:`~.mask`
   method and manually apply the mask.

.. warning::

   Some methods will not work with masked data (e.g. splines and interpolation).
   Therefore, resampling operations will not carry over the masks. You have to
   apply the masks again. One instance in which this is important is plotting
   with :py:mod:`~.visualize_matplotlib`. Also, the :py:meth:`~.save` method
   will discard the mask information.
