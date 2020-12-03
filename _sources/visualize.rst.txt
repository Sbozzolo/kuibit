Visualizing data
=============================

The module :py:mod:`~.visualize` contains functions to immediately visualize
Einstein Toolkit data. (:ref:`visualize_ref:Reference on postcactus.visualize`)

Grid functions
--------------

The main way to visualize grid data is with the function
:py:mod:`~.plot_contourf`. This method support a large variety of inputs:

- NumPy array: it will be plotted with ``imshow``, unless the keyword
  ``coordinates`` is provided, in which case they will be plotted as
  ``contourf``
- :py:class:`~.UniformGridData`: it will be plotted over the entire domain,
  unless the keywords ``x0``, ``x1`` and ``shape`` are provided. These
  specify the extent of the grid to be plotted. ``shape`` is essentially the
  resolution of the plot. See documentation on grid data for details.
- :py:class:`~.HierarchicalGridData`: it will be plotted over the entire domain,
  with resolution provided by ``shape``. The keywords ``x0``, ``x1`` can change
  the grid extent.
- :py:class:`~.BaseOneGridFunction`: same as :py:class:`~.HierarchicalGridData`,
  but you also have to provide the iteration with the ``iteration`` keyword.

``shape``, ``x0``, ``x1`` are all 2D lists or tuples. The function also takes
``xlabel`` and ``ylabel`` to customize the labels.

Implementation details
----------------------

In :py:mod:`~.visualize`, we embrace duck-typing. Ideally, we want to be able to
plot what the user wants to plot, taking care of the nuisances. This includes:
numpy arrays, :py:class:`~.UniformGridData`, :py:class:`~.HierarchicalGridData`,
and so on. To abstract away the details of how to handle all the different
types, we have decorators. For example, with the decorator
:py:class:`~._preprocess_plot_grid`, we give support to handling all the
differnt types to any function that takes ``data`` and and ``coordiantes`` as
arguments. This can be used for other functions too.
