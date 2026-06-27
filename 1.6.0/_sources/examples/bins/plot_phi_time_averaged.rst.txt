plot_phi_time_averaged.py
=============================================

``plot_phi_time_averaged`` plots the azimuthal and time average of a 2D grid
function. The scripts interpolates the quantity over concentric rings around a
given center, and average all the values at fixed radius. Then, these are
averaged over a time window defined by passing ``--tmin``, ``--tmax``, and
``--time-every``. This last parameter specifies the frequency of snapshots to
use for the time average (i.e., for the time average, use one snapshot every N
available).


.. literalinclude:: ../../../examples/bins/plot_phi_time_averaged.py
  :language: python
