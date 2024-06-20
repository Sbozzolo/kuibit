Visualizing data with mayavi
================================

The module :py:mod:`~.visualize_mayavi` contains functions to easily
visualize Einstein Toolkit data with ``mayavi``
(:ref:`visualize_mayavi_ref:Reference on kuibit.visualize_mayavi`). For 2D
visualization, look at :py:mod:`~.visualize_matplotlib`.

Introduction to mayavi
----------------------

``mayavi`` is a Python package for 3D visualization based on VTK. The main focus
of ``mayavi`` is ease of use: instead of dealing with the low-level details of
VTK, ``mayavi`` provides simple functions to perform common renderings.

``mayavi`` provides an interactive environment where to play with the data. This
has to be disabled when we write scripts. To do that, make sure to call
`~.disable_interactive_window` every time you use ``mayavi``.
