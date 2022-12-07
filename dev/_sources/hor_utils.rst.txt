Horizon utilities
==============================

The :py:mod:`~.hor_utils` module contains functions that are useful in working
with horizons and their orbits

:ref:`hor_utils_ref:Reference on kuibit.hor_utils`

Separation and separation vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In simulations of binary black holes, it is often convenient to look at the
separation between the two horizons. This is can be easily achieved with the
functions :py:func:`~.compute_separation_vector` and
:py:func:`~.compute_separation`. These functions take two horizons as input and
return a :py:class:`~.Vector` and a :py:class:`~.TimeSeries` respectively. The
vector is the separation along the three Cartesian coordinates, while the
timeseries is the magnitude of the separation. Note that the vector has sign:
the separation is always computed as ``hor1 - hor2``.

Newtonian center of mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A second useful quantity is the center of mass. Computing the center of mass in
a gauge-invariant way is not easy (the best way is looking at the linear
momentum carried away by gravitational waves). However, we can easily compute
the Newtonian center of mass:

.. math::

   x^i_{CM} = \frac{m_1 x^i_1 + m_2 x^i_2}{m_1 + m_2}

Note that in this formula, we use the irreducible mass and this might not be the
most appropriate choice in presence of spin. This is implemented in
:py:func:`~.compute_center_of_mass`.

Newtonian angular velocity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the same spirit, we can compute the Newtonian angular velocity vector as

.. math::

       \mathbf{\Omega} = \frac{\mathbf{r} \times \mathbf{\dot{r}}}{r^2}


For orbits on the equatorial plane, the z component is the usual angular
velocity. This is implemented in :py:func:`~.compute_angular_velocity_vector`.
