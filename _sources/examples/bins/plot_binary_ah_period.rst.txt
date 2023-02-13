plot_binary_ah_period.py
=============================================

``plot_binary_ah_period`` plots the period on the equatorial plane estimated in
the following way. First, we compute the Newtonian angular velocity with the
cross product between the separation and its time derivative, and then we
compute the associated period.

This leads to a lot junk values, so the script contains an heuristics to improve
the plotting range, but it that does not work, you can manually choose the
bounds with ``--ymin`` and ``--ymax``.


.. literalinclude:: ../../../examples/bins/plot_binary_ah_period.py
  :language: python
