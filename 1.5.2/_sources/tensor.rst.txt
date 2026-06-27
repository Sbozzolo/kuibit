Working with tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module :py:mod:`~.tensor` (:ref:`tensor_ref:Reference on kuibit.tensor`)
provides abstractions to work with mathematical tensors. In practice, at the
moment, this mostly mean working with :py:class:`~.Vector` and
:py:class:`~.Matrix` objects (both derived from :py:class:`~.Tensor`).

The :py:class:`~.Tensor` interface is a container that takes nested lists of
:py:class:`~.BaseNumerical` data (e.g., :py:class:`~.TimeSeries`,
:py:class:`~.FrequencySeries`, :py:class:`~.UniformGridData`,
:py:class:`~.HierarchicalGridData`) and provides an high-level abstractions that
supports all the usual mathematical operations and inherits methods from the
contained objects. Examples will clarify what we mean with this.

Vectors
-------

The :py:class:`~.Vector` implements a mathematical vector. You can make vectors
out of anything that is derived from :py:class:`~.BaseNumerical`. In ``kuibit``,
this means grid and series data. The convenience of working with
:py:class:`~.Vector` is that you can act on multiple components of the same
entity in one go.

For instance, assume you have the three components of the velocity field on the
equatorial plane at a given iteration as ``vx``, ``vy``, ``vz``.

.. code-block:: python

    from kuibit.tensor import Vector

    # To create a vector
    vel = Vector([vx, vy, vz])

    # Evaluate the velocity on location (2, 3)
    vel([2, 3])  # Will return a 2-element NumPy array

    # Vectors can be indexed
    vz = vel[2]

    # We can compute the magnitude
    norm = vel.norm()  # Will return a grid variable

    # Vectors support all the mathematical operations
    vx_vy = (vx ** 2 + vy ** 2).sqrt()

    # Vectors inherit all the methods from the base class:
    # to compute the derivative of vx_vy along the axis, we can use
    # the partial_differentiated method
    deriv_x = vx_vy.partial_differentiated(0)

    # We could also have used the in-place version (notice the imperative name)
    vx_vy.partial_differentiate(0)


This would not change much if ``vx``, ``vy``, and ``vz`` where other objects,
like :py:class:`~.TimeSeries`.

Matrices and Tensors
--------------------

The :py:class:`~.Matrix` class implements a mathematical matrix of arbitrary
dimensions. It behaves exactly like the :py:class:`~.Vector` described in the
previous section. At the moment, no specific matrix method is implemented, but
all the infrastructure is in place to support them.

Developing a generic :py:class:`~.Tensor` class is a hard task, so, as it
stands, :py:class:`~.Tensor` is does not have many features by itself but it is
rather the generic implementation that underpins both :py:class:`~.Vector` and
:py:class:`~.Matrix`. For example, no notion of rank or covariance is
implemented yet.
