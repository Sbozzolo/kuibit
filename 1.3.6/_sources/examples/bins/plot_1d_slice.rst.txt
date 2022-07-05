plot_1d_slice.py
=============================================

``plot_1d_slice`` plots a grid function along one of the coordinate axis. If 1D
is available, then it is used. If not, then 2D data is used (slicing it so that
the other coordinate is set to 0). If 2D data is not available, then 3D is used.


.. literalinclude:: ../../../examples/bins/plot_1d_slice.py
  :language: python
