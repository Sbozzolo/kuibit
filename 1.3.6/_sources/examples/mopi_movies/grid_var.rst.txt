grid_var
=============================================

``plot_grid_var`` plots a 2D grid function as output by Carpet. It takes as
input the variable, and the region of space where to plot it. This is defined by
giving the resolution, the lowermost leftmost coordinate (``--x0``), and the
topmost rightmost one (``--x1``). Optionally, it is possible to take the
logarithm base 10 and/or the absolute value. It is also possible to perform
interpolation in two different ways: one at the level of the data (slower, but
more accurate), and the other at the level of the image using the interpolations
available in matplotlib.


.. literalinclude:: ../../../examples/mopi_movies/grid_var
  :language: python
