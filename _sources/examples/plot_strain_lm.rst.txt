plot_strain_lm.py
=============================================

``plot_strain_lm`` plots the (l,m) mode of the gravitational-wave strain as
measured from :math:`\Psi_4` on a given detector. The script take as optional
arguments a window function to apply before performing the integration. The
accepted values are the names of the methods in the class
:py:class:`~.TimeSeries`.

.. literalinclude:: ../../examples/bins/plot_strain_lm.py
  :language: python
