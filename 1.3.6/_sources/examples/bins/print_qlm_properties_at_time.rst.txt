print_qlm_properties_at_time.py
=============================================

``print_qlm_properties_at_time`` prints mass, irreducible mass, and angular
momentum for a given horizon at a given time. Cubic splines are used to
interpolate between timesteps. If ``--estimate-gamma`` is passed, the script
also computes the Lorentz factor with the following formula:

.. math::

   \gamma = \sqrt{1 + \frac{P^2}{M_{\textrm{irr}}^2}}\,,
   P^2 = \sum_{i} P^i P^i\,,

where :math:`P^i` is the Weinberg linear momentum. This choice is motivated by
the fact that this value coincides with the one set in ``TwoPunctures``.

.. literalinclude:: ../../../examples/bins/print_qlm_properties_at_time.py
  :language: python
