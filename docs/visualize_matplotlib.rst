Visualizing data with matplotlib
================================

The module :py:mod:`~.visualize_matplotlib` contains functions to easily
visualize Einstein Toolkit data with ``matplotlib``
(:ref:`visualize_matplotlib_ref:Reference on kuibit.visualize_matplotlib`).

.. note::

   Visualizations are a subjective matter: different people have different
   tastes, and it is impossible to have one-size-fits-all solutions.
   :py:mod:`~.visualize_matplotlib` comes with convenient functions to quickly
   visualize data, but the module has default values that are opinionated. Where
   possible, customization has been exposed for your convenience. However, some
   settings and choices cannot be changed.

Series
------

Time and frequency series in ``kuibit`` can be plotted with ``matplotlib``
native functions. For example,

.. code-block:: python

    # Assuming rho_b is a TimeSeries

    import matplotlib.pyplot as plt

    plt.plot(rho_b)

Grid functions
--------------

The main way to visualize grid data is with the functions :py:mod:`~.plot_color`
and :py:mod:`~.plot_contourf`. The first plots directly the values of the given
data, the second plots contours (filled). The functions support a large variety
of inputs:

- NumPy arrays, with coordintes specified by the ``coordinates`` keyword.
- :py:class:`~.UniformGridData`: it will be plotted over the entire domain,
  unless the keywords ``x0``, ``x1`` and ``shape`` are provided. These
  specify the extent of the grid to be plotted. ``shape`` is essentially the
  resolution of the plot. See documentation on grid data for details.
- :py:class:`~.HierarchicalGridData`: it will be plotted over the entire domain,
  with resolution provided by ``shape``. The keywords ``x0``, ``x1`` can change
  the grid extent.
- :py:class:`~.BaseOneGridFunction`: same as :py:class:`~.HierarchicalGridData`,
  but you also have to provide the iteration with the ``iteration`` keyword.

A table at the end of this document sums up these options. ``shape``, ``x0``,
``x1`` are all 2D lists or tuples.

The functions take additional arguments (for example, the labels for the axes).
Details can be found in the reference material
(:ref:`visualize_matplotlib_ref:Reference on kuibit.visualize_matplotlib`).

.. warning::

   Mask information will be lost upon resampling. If you want masks, you need to
   provide the :py:class:`~.UniformGridData` you want to plot.

Plots can be enanched by adding a colorbar with :py:mod:`~.plot_colorbar` (or
passing the ``colorbar`` option).

The function :py:func:`~.plot_contour` plots the contours without filling. It
requires the number of levels, or an array that specify where the levels are.

With :py:func:`~.plot_components_boundaries`, you can plot the grid structure.
The function takes a :py:class:`~.HierarchicalGridData` as argument as plots its
grid structure. By defaults, ghost zones are not included, but they can be
passing ``remove_ghosts=False``. The function plots the boundaries of the single
components, but when multiple components can be merged into a single refinement
level, then only the outer boundary is plotted (a notable case in which this is
currently not possible is when there are multiple refinement centers).


Horizons
--------

Apparent horizons can be plotted with :py:func:`~.plot_horizon`. This
function takes the 2D shape of an horizon as returned by
:py:meth:`~.shape_outline_at_iteration` or :py:meth:`~.shape_outline_at_time`.
There are two wrappers around this function to simplify plotting horizons on the
usual planes ``xy``, ``xz``, and ``yz``:
:py:func:`~.plot_horizon_on_plane_at_iteration` and
:py:func:`~.plot_horizon_on_plane_at_time`. These take an
:py:class:`~.OneHorizon` object, the iteration/time that has to be plotted, and
the plane over which you are plotting. All these functions take also the color
of the horizon, the color of its boundary, and the opacity (``alpha``).

.. warning::

   When you take a cross section (an outline) of an horizon, ``kuibit`` finds
   points that are within a threshold to the plane that cuts the surface.
   However, the way points are distributed on apparent horizons is highly
   non-uniform. So, if you are cutting the horizon along an axis that is not one
   of the coordinate ones (for the horizon), it is likely that too few points
   will be close enough to the intersecting plane, resulting in a malformed or
   absent outline. In some distant future, ``kuibit`` will perform
   interpolations to solve this problem.


Other utilities
----------------------

setup_matplotlib
^^^^^^^^^^^^^^^^

The default settings in ``matplotlib`` are not great (e.g., the text is
typically too small). The function :py:func:`~.setup_matplotlib` sets some
better defaults. Since "better" is relative, the function takes an optional
argument ``params``. This has to be dictionary with keys the parameters in
``matplotlib`` and values their new values. This can be used to override some of
the defaults in :py:func:`~.setup_matplotlib`.

.. note::

   The function :py:func:`~.setup_matplotlib` simply updates the settings in
   ``matplotlib``. It does not have any other effect.

add_text_to_corner
^^^^^^^^^^^^^^^^^^

The function :py:func:`~.add_text_to_corner` annotates a figure adding a label.
The location of the label can be specified with the ``node`` argument. This is
identified with cardinal point (N,S,W, or E) or a combination of two of them.
For example, the default behavior is to place the text in the bottom right
corner (corresponding to South-East--SE). The distance from the border can also
be customized by passing the ``offset`` argument.

save
^^^^

The :py:func:`~.save` function saves the figure to file. The function takes the
path of the output and saves the current figure. If the file has extension
``.tikz``, then ``tikzplotlib`` instead of ``matplotlib`` is used to save the
figure. This results in a PGFPlots/TikZ ASCII file ready to be compiled with
LaTeX.


preprocess_plot and preprocess_plot_grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:func:`~.preprocess_plot` and :py:func:`~.preprocess_plot_grid` are two
decorators. The first one adds support for passing ``figure`` and ``axis`` to a
function. Let us see how it works:

.. code-block:: python

    from kuibit.visualize_matplotlib import preprocess_plot

    @preprocess_plot
    def my_plot(data, figure=None, axis=None):
        # My plotting, for example
        ax.plot(data)

What :py:func:`~.preprocess_plot` is the following: if the user provides
``figure`` and/or ``axis``, then those are used. If the user does not provide
those arguments, then the current one are used. This is roughly equivalent to
checking if ``figure is None`` and if it is, then set ``figure = plt.gcf()``.

The second decorator is :py:func:`~.preprocess_plot_grid`. With this, you can
forget about all the classes defined in ``kuibit`` and simply plot NumPy arrays.
In more details: when you work with ``kuibit``, you will typically work with
:py:class:`~.HierarchicalGridData` and :py:class:`~.UniformGridData`. These are
complex structures that cannot be plotted immediately. The decorator
:py:func:`~.preprocess_plot_grid` takes care of all the boilerplate needed to
work with those two classes so that the user can provide a
:py:class:`~.OneGridFunction`, :py:class:`~.UniformGridData`, a
:py:class:`~.HierarchicalGridData`, or a NumPy array. Let us see how it works:

.. code-block:: python

    from kuibit.visualize_matplotlib import preprocess_plot_grid

    @preprocess_plot_grid
    def my_plot(data, coordinates=None, figure=None, axis=None):
        # My plotting, for example
        ax.imshow(data)

    # bob here is a H5OneGridFunction

    # Some of the arguments are optional
    my_plot(bob, shape=[500, 500], iteration=0, x0=[0, 0], x1=[1,1])

    # rho_b here is a HierarchicalGridData

    # Some of the arguments are optional
    my_plot(rho_b, shape=[500, 500], x0=[0, 0], x1=[1,1])

    # press here is a UniformGridData

    # Some of the arguments are optional
    my_plot(press, x0=[0,0])

    # eps here is a NumPy array
    my_plot(eps)


Depending on the type of object passed, additional arguments might be needed.
See table below for details.

+------------------------------------+------------------------------------------------------+------------------------------+
|                Type                |                   Arguments needed                   |     Arguments supported      |
+====================================+===========================+==========================+==============================+
| :py:class:`~.BaseOneGridFunction`  | ``iteration``, ``shape``                             | ``x0``, ``x1``, ``resample`` |
+------------------------------------+------------------------------------------------------+------------------------------+
| :py:class:`~.HierarchicalGridData` | ``shape``                                            | ``x0``, ``x1``, ``resample`` |
+------------------------------------+------------------------------------------------------+------------------------------+
| :py:class:`~.UniformGridData`      | ``shape`` (if ``x0`` or ``x1`` are passed)           | ``x0``, ``x1``, ``resample`` |
+------------------------------------+------------------------------------------------------+------------------------------+
| 2D NumPy array                     | ``coordinates`` (depending on the specific function) |                              |
+------------------------------------+------------------------------------------------------+------------------------------+
