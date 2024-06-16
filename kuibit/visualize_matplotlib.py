#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
#
# Inspired by code originally developed by Wolfgang Kastaun. This file may
# contain algorithms and/or structures first implemented in
# GitHub:wokast/PyCactus/PostCactus/visualize.py
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

"""The :py:mod:`~.visualize_matplotlib` module provides methods to plot
``kuibit`` objects with matplotlib and other convenience functions.

Utilities:

- :py:func:`~.setup_matplotlib` adjusts the configuration in matplotlib.
- :py:func:`~.add_text_to_corner` adds a label near an edge or a corner of
  a figure (useful for annotations like time).
- :py:func:`~.save` saves the figure, optionally with tikzplotlib.
- :py:func:`~.save_from_dir_filename_ext` saves the figure and assembles the
  name automatically from the output directory, file name and extension.
- :py:func:`~.set_axis_limits` sets the range on the two axes of a given axis.
  :py:func:`~.set_axis_limits_from_args` does the same but reading the data
  from a given ``args`` (from ``ArgParse``).

Two decorators:

- :py:func:`~.preprocess_plot`. The goal of this is to add support to the
  keyword arguments ``figure`` and ``axis``. If you decorate a function with
  :py:func:`~.preprocess_plot` and the function takes those two arguments, then
  the decorator will automatically check if the arguments were passed and if not
  set the correct default.

- :py:func:`~.preprocess_plot_grid`. This decorator takes some form of grid data
  and returns a NumPy array (and other useful quantities). Functions decorated
  with :py:func:`~.preprocess_plot_grid` automatically gain support for
  :py:class:`~.HierarchicalGridData` and :py:class:`~.UniformGridData`, so they
  only need to worry about plotting NumPy arrays. Functions have to take a
  positional argument ``data`` and a keyword argument ``coordinates``.

Grid data:

- :py:func:`~.plot_color` to plot directly some data with its value.
- :py:func:`~.plot_contourf` to draw a filled contour plot using the data.

Horizons:

- :py:func:`~.plot_horizon` to plot a given shape in 2D.
- :py:func:`~.plot_horizon_on_plane_at_iteration` to plot a given horizon
   at a given iteration in 2D.
- :py:func:`~.plot_horizon_on_plane_at_time` to plot a given horizon
   at a given time in 2D.

Most of the functions here take optional arguments ``figure`` and ``axis``. You
can specify them, or the current ones will be used.

"""

import itertools
import os
import warnings
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from kuibit import grid_data as gd
from kuibit.cactus_grid_functions import BaseOneGridFunction

# UTILITIES


def setup_matplotlib(
    params: Optional[Dict[str, Any]] = None, rc_par_file: Optional[str] = None
) -> None:
    """Setup matplotlib with some reasonable defaults for better plots.

    If ``params`` is provided, add these parameters to matplotlib's settings
    (``params`` updates ``matplotlib.rcParams``).

    If ``rc_par_file`` is provided, first set the parameters reading the values
    from the ``rc_par_file``. (``params`` has the precedence over the parameters
    read from the file.)

    Matplotlib behaves differently on different machines. With this, we make
    sure that we set all the relevant paramters that we care of to the value we
    prefer. The default values are highly opinionated.

    :param params: Parameters to update matplotlib with.
    :type params: dict

    :param rc_par_file: File where to read parameters. The file has to use
                        matplotlib's configuration language. ``params``
                        overwrites the values set from this file, but this file
                        overrides the default values set in this function.
    :type rc_par_file: str

    """

    matplotlib.rcParams.update(
        {
            "lines.markersize": 4,
            "axes.labelsize": 16,
            "font.weight": "light",
            "font.size": 16,
            "legend.fontsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.formatter.limits": [-3, 3],
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "image.cmap": "inferno",
            "legend.fancybox": False,
            "legend.edgecolor": "inherit",
        }
    )

    if rc_par_file is not None:
        matplotlib.rc_file(rc_par_file)

    if params is not None:
        matplotlib.rcParams.update(params)


def preprocess_plot(func):
    """Decorator to set-up plot functions.

    When we plot anything, there is always some boilerplate that has to be
    executed. For example, we want to provide an axis keyword so that the user
    can specify where to plot, but if the keyword is not provided, we want to
    plot on the current figure.

    Essentially, this decorator sets default values. Why don't we do
    axis=plt.gca() then? The problem is that the default values are set when
    the function is defined, not when it is called. So, this will not work.

    This decorator takes care of everything.

    1. It handles the axis keyword setting it to plt.gca() if it was not
       provided.
    2. It handles the figure keyword setting it to plt.gcf() if it was not
       provided.

    func has to take as keyword arguments:
    1. 'axis=None', where the plot will be plot, or plt.gca() if None
    2. 'figure=None', where the plot will be plot, or plt.gcf() if None

    """

    def inner(*args, **kwargs):
        # Setdetault addes the key if it is not already there
        kwargs.setdefault("axis", plt.gca())
        kwargs.setdefault("figure", plt.gcf())
        return func(*args, **kwargs)

    return inner


def preprocess_plot_grid(func):
    """Decorator to set-up plot functions that plot grid data.

    This decorator extends :py:func:`~.preprocess_plot` for specific functions.

    1. It handles differt types to plot what intuitively one would want to
       plot.
    1a. If the data is a NumPy array with shape 2, just pass the data,
        otherwise raise an error
    1b. If the data is a NumPy array, just pass the data.
    1c. If data is :py:class:`~.UniformGridData`, pass the data and the
        coordinates.
    1d. If data is :py:class:`~.HierarchicalGridData`, read resample it to
        the given grid, then pass do 1c.
    1e. If data is a :py:class:`~.BaseOneGridFunction`, we read the iteration
        and pass to 1d.

    func has to take as keyword arguments (in addition to the ones in
    :py:func`~.preprocess_plot`):
    1. 'data'. data will be passed as a NumPy array, unless it is
               already so.
    2. 'coordinates=None'. coordinates will be passed as a list of NumPy
                           arrays, unless it is not None. Each NumPy
                           array is the coordinates along one axis.

    """

    @preprocess_plot
    def inner(data, *args, **kwargs):
        # The flow is: We check if data is BaseOneGridFunction or derived. If
        # yes, we read the requested iteration. Then, we check if data is
        # HierachicalGridData, if yes, we resample to UniformGridData. Then we
        # work with UniformGridData and handle coordinates, finally we work
        # with NumPy arrays, which is what we pass to the function.

        def attr_not_available(attr):
            """This is a helper function to see if the user passed an attribute
            or if the attribute is None
            """
            return attr not in kwargs or kwargs[attr] is None

        def default_or_kwargs(attr, default):
            """Return default if the attribute is not available in kwargs, otherwise return
            the attribute

            """
            if attr_not_available(attr):
                return default
            return kwargs[attr]

        if isinstance(data, BaseOneGridFunction):
            if attr_not_available("iteration"):
                raise TypeError(
                    "Data has multiple iterations, specify what do you want to plot"
                )

            # Overwrite data with HierarchicalGridData
            data = data[kwargs["iteration"]]

        if isinstance(data, gd.HierarchicalGridData):
            if attr_not_available("shape"):
                raise TypeError(
                    "The data must be resampled but the shape was not provided"
                )

            # If x0 or x1 are None, we use the ones of the grid
            x0 = default_or_kwargs("x0", data.x0)
            x1 = default_or_kwargs("x1", data.x1)
            resample = default_or_kwargs("resample", False)

            # Overwrite data with UniformGridData
            if data.is_masked():
                warnings.warn(
                    "Mask information will be lost with the resampling"
                )

            data = data.to_UniformGridData(
                shape=kwargs["shape"], x0=x0, x1=x1, resample=resample
            )

        if isinstance(data, gd.UniformGridData):
            # We check if the user has passed coordinates too.
            if "coordinates" in kwargs and kwargs["coordinates"] is not None:
                warnings.warn(
                    "Ignoring provided coordinates (data is UniformGridData)."
                    " To specify boundaries, use x0 and x1."
                )

            # If x0 or x1 are None, we use the ones of the grid
            x0 = default_or_kwargs("x0", data.x0)
            x1 = default_or_kwargs("x1", data.x1)
            # If x0 or x1 are provided, then we resample. So, we don't resample
            # only if x0 AND x1 are not provided.
            resampling = not (
                attr_not_available("x0") and attr_not_available("x1")
            )

            if resampling and attr_not_available("shape"):
                raise TypeError(
                    "The data must be resampled but the shape was not provided"
                )

            if resampling:
                resample = default_or_kwargs("resample", False)
                new_grid = gd.UniformGrid(shape=kwargs["shape"], x0=x0, x1=x1)

                if data.is_masked():
                    warnings.warn(
                        "Mask information will be lost with the resampling"
                    )

                data = data.resampled(
                    new_grid, piecewise_constant=(not resample)
                )

            kwargs["coordinates"] = data.coordinates_from_grid()
            # Overwrite data with NumPy array
            data = data.data_xyz

        if isinstance(data, np.ndarray) and data.ndim != 2:
            raise ValueError("Only 2-dimensional data can be plotted")

        # TODO: Check that coordinates are compatible with data

        # We remove what we don't need from kwargs, so that it is not
        # accidentally passed to the function
        def remove_attributes(*attributes):
            for attr in attributes:
                if attr in kwargs:
                    del kwargs[attr]

        remove_attributes("shape", "x0", "x1", "iteration", "resample")

        return func(data, *args, **kwargs)

    return inner


def _process_anchor_info(anchor, offset):
    """Prepare the correct arguments for text_function in
    :py:func:`~.add_text_to_corner`

    :param anchor: Where to place the text? This is defined
                 via cardinal points (e.g., NW for top left).
    :type anchor: str
    :param offset: How far from the edge to put the text? In percentage
                   of the entire figure.
    :type offset: float

    :returns: Horizontal position, vertical position, horizontal alignment
             vertical alignment.
    :rtype: tuple

    """
    # This function has been split off add_text_to_corner only because it is
    # easier to test this way.
    possible_combinations = [
        n1 + n2 for n1, n2 in itertools.product("NSWE", repeat=2) if n1 != n2
    ] + ["N", "S", "W", "E"]

    if anchor not in possible_combinations:
        raise ValueError("Given anchor is invalid. Use cardinal points.")

    # Now we have to parse the cardinal points to find where
    # to put the

    ver_al = None
    ver_pos = 0.5
    if "S" in anchor:
        ver_al = "bottom"
        ver_pos = offset
    elif "N" in anchor:
        ver_al = "top"
        ver_pos = 1 - offset

    hor_al = None
    hor_pos = 0.5
    if "E" in anchor:
        hor_al = "right"
        hor_pos = 1 - offset
    elif "W" in anchor:
        hor_al = "left"
        hor_pos = offset

    return hor_pos, ver_pos, hor_al, ver_al


@preprocess_plot
def add_text_to_corner(text, anchor="SE", figure=None, axis=None, offset=0.01):
    """Add text to a figure.

    Specify the location of the label using cardinal points (NSWE).
    For example, SE is bottom right.

    :param anchor: Where to place the text? This is defined
                 via cardinal points (e.g., NW for top left).
    :type anchor: str
    :param offset: How far from the edge to put the text? In percentage
                   of the entire figure.
    :type offset: float
    """

    # text_function is the correct function to use depending if we are working
    # with 2D or 3D plots.

    text_function = axis.text2D if isinstance(axis, Axes3D) else axis.text

    hor_pos, ver_pos, hor_al, ver_al = _process_anchor_info(anchor, offset)

    kwargs = {}
    if hor_al is not None:
        kwargs.update({"horizontalalignment": hor_al})

    if ver_al is not None:
        kwargs.update({"verticalalignment": ver_al})

    return text_function(
        hor_pos, ver_pos, text, transform=figure.transFigure, **kwargs
    )


@preprocess_plot
def save(
    outputpath,
    figure=None,
    axis=None,
    tikz_clean_figure=False,
    **kwargs,
):
    """Save figure to the given location.

    If the file extension is ``.tikz``, the file will be saved with
    ``tikzplotlib``.

    Unknown arguments are passed to the ``matplotlib.savefig`` or
    ``tikzplotlib.save`` (depending on the extension). In this second case, if
    ``tikz_clean_figure = True``, unknown arguments are first passed to
    ``tikzplotlib.clean_figure``.

    :param outputpath:  Output path with or without extension. If the
                        extension is ``.tikz``, the file is saved with
                        ``tikzplotlib``.
    :type outputpath:  str
    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param tikz_clean_figure: If ``tikzplotlib`` is begin used, reduce the size
                              of the output ``tikz`` file. When this is set to
                              True, unknown arguments are first passed to
                              ``tikzplotlib.clean_figure``, then to
                              ``tikzplotlib.save``. ``tikzplotlib.clean_figure``
                              will change the given figure.
    :type tikz_clean_figure: bool

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``

    """
    if os.path.splitext(outputpath)[-1] == ".tikz":
        import tikzplotlib

        # If clean_figure is True, we extract from kwargs those argument
        # that tikzplotlib.clean_figure would take. For this, we need to
        # know what argument that function takes.
        #
        # Form https://stackoverflow.com/a/40363565
        if tikz_clean_figure:
            args_names = tikzplotlib.clean_figure.__code__.co_varnames[
                : tikzplotlib.clean_figure.__code__.co_argcount
            ]

            kwargs_clean_figure = {}

            # We split kwargs in those that are supposed to be passed to
            # clean_figure and those that have to be passed to save. For this,
            # we iterate over the arguments taken by clean_figure, if they are
            # in kwargs, then we move them to a new dictionary
            for arg in args_names:
                if arg in kwargs:
                    kwargs_clean_figure.update({arg: kwargs[arg]})
                    del kwargs[arg]

            tikzplotlib.clean_figure(fig=figure, **kwargs_clean_figure)

        tikzplotlib.save(outputpath, **kwargs)
    else:
        figure.savefig(outputpath, **kwargs)


@preprocess_plot
def save_from_dir_filename_ext(
    output_dir,
    file_name,
    file_ext,
    figure=None,
    axis=None,
    tikz_clean_figure=False,
    **kwargs,
):
    """Save figure to a location defined by a folder, a name, and an extension.

    If the ``file_ext`` is ``tikz``, the file will be saved with
    ``tikzplotlib``.

    Unknown arguments are passed to the ``matplotlib.savefig`` or
    ``tikzplotlib.save`` (depending on the extension).

    :param output_dir: Path of a directory where to save the figure
    :type output_dir: str
    :param file_name: Name of the file.
    :type file_name: str
    :param file_extension: Extension of the file, with or without dot.
    :type file_extension: str
    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``

    """
    ext = file_ext if file_ext[0] == "." else f".{file_ext}"
    outputpath = os.path.join(output_dir, f"{file_name}{ext}")
    return save(
        outputpath,
        figure=figure,
        axis=axis,
        tikz_clean_figure=tikz_clean_figure,
        **kwargs,
    )


@preprocess_plot
def set_axis_limits(
    xmin=None, xmax=None, ymin=None, ymax=None, figure=None, axis=None
):
    """Set limits on the two axes of axis.

    :param xmin: Minimum on the horizontal axis.
    :type xmin: float
    :param xmax: Maximum on the horizontal axis.
    :type xmax: float
    :param ymin: Minimum on the horizontal axis.
    :type ymin: float
    :param ymax: Maximum on the vertical axis.
    :type ymax: float

    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``

    """
    axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax))


@preprocess_plot
def set_axis_limits_from_args(args, figure=None, axis=None):
    """Set limits on the two axes of axis with data read from ``args``.

    It uses the ``xmin``, ``xmax``, ``ymin``, ``ymax`` attributes.

    :param args: Options provided by the user.
    :type args: `argparse.Namespace`

    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``

    """
    set_axis_limits(
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        figure=figure,
        axis=axis,
    )


def get_figname(args, default):
    """Return the figure name checking if the user has passed one.

    If it is defined, return ``args.figname``, otherwise return default.

    :param args: Options provided by the user.
    :type args: `argparse.Namespace`
    :param default: Default name if ``figname`` is not in ``args``.
    :type default: str

    :returns: Name of the output figure.
    :rtype: str

    """
    if args.figname is None:
        return default
    # figname is not None
    return args.figname


# GRID FUNCTIONS


def _vmin_vmax_extend(data, vmin=None, vmax=None):
    """Helper function to decide what to do with the colorbar (to extend it or not?)."""

    colorbar_extend = "neither"

    if vmin is None:
        vmin = data.min()

    if data.min() < vmin:
        colorbar_extend = "min"

    if vmax is None:
        vmax = data.max()

    if data.max() > vmax:
        if colorbar_extend == "min":
            colorbar_extend = "both"
        else:
            colorbar_extend = "max"

    return vmin, vmax, colorbar_extend


# All the difficult stuff is in preprocess_plot_grid
@preprocess_plot_grid
def _plot_grid(
    data,
    plot_type="color",
    figure=None,
    axis=None,
    coordinates=None,
    xlabel=None,
    ylabel=None,
    colorbar=False,
    label=None,
    logscale=False,
    vmin=None,
    vmax=None,
    aspect_ratio="equal",
    **kwargs,
):
    """Backend of the :py:func:`~.plot_color` and similar functions.

    The type of plot is specified by the variable ``plot_type``.

    Unknown arguments are passed to
    ``imshow`` if plot is color
    ``contourf`` if plot is contourf.
    ``contour`` if plot is contour.

    :param plot_type: Type of plot. It can be: 'color', 'contourf', 'contour'.
    :type plot_type: str
    """

    _known_plot_types = ("color", "contourf", "contour")

    if plot_type not in _known_plot_types:
        raise ValueError(
            f"Unknown plot_type {plot_type} (Options available {_known_plot_types})"
        )

    # Considering all the effort put in preprocess_plot_grid, we we can plot
    # as we were plotting normal NumPy arrays.

    if logscale:
        # We mask the values that are smaller or equal than 0
        data = np.ma.log10(data)

    vmin, vmax, colorbar_extend = _vmin_vmax_extend(data, vmin=vmin, vmax=vmax)

    # To implement vmin and vmax, we clamp the data to vmin and vmax instead of
    # using the options in matplotlib. This greatly simplifies handling things
    # like colormaps.
    data = np.clip(data, vmin, vmax)

    if aspect_ratio is not None:
        axis.set_aspect(aspect_ratio)

    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)

    if plot_type == "color":
        if coordinates is None:
            grid = None
        else:
            # We assume equally-spaced points.

            # TODO: (Refactoring)
            #
            # This is not a very pythonic way to write this...

            X, Y = coordinates

            dx, dy = X[1] - X[0], Y[1] - Y[0]
            grid = [
                X[0] - 0.5 * dx,
                X[-1] + 0.5 * dx,
                Y[0] - 0.5 * dy,
                Y[-1] + 0.5 * dy,
            ]

        image = axis.imshow(
            data, vmin=vmin, vmax=vmax, origin="lower", extent=grid, **kwargs
        )
    elif plot_type == "contourf":
        if coordinates is None:
            raise ValueError(
                f"You must provide the coordiantes with plot_type = {plot_type}"
            )
        image = axis.contourf(
            *coordinates, data, extend=colorbar_extend, **kwargs
        )
    elif plot_type == "contour":
        if coordinates is None:
            raise ValueError(
                f"You must provide the coordiantes with plot_type = {plot_type}"
            )
        # We need to pass the levels for the contours
        if "levels" not in kwargs:
            raise ValueError(
                f"You must provide the levels with plot_type = {plot_type}"
            )
        image = axis.contour(
            *coordinates, data, extend=colorbar_extend, **kwargs
        )

    if colorbar:
        plot_colorbar(image, figure=figure, axis=axis, label=label)

    return image


def plot_contourf(data, **kwargs):
    """Plot the given data drawing filled contours.

    You can pass (everything is processed by :py:func:`~.preprocess_plot_grid` so
    that at the end we have a 2D NumPy array):
    - A 2D NumPy array,
    - A :py:class:`~.UniformGridData`,
    - A :py:class:`~.HierarchicalGridData`,
    - A :py:class:`~.BaseOneGridFunction`.

    Depending on what you pass, you might need additional arguments.

    If you pass a :py:class:`~.BaseOneGridFunction`, you need also to pass
    ``iteration``, and ``shape``. If you pass
    :py:class:`~.HierarchicalGridData`, you also need to pass ``shape``. In all
    cases you can also pass ``x0`` and ``x1`` to define origin and corner of the
    grid. You can pass the option ``resample=True`` if you want to do bilinear
    resampling at the grid data level, otherwise, nearest neighbor resampling is
    done. When you pass the NumPy array, you also have to pass the
    ``coordinates``.

    All the unknown arguments are passed to ``contourf``.

    .. note

       Read the documentation for a concise table on what arguments are
       supported.

    :param data: Data that has to be plotted. The function expects a 2D NumPy
                 array, but the decorator :py:func:`~.preprocess_plot_grid`
                 allows it to take different kind of data.
    :type data: 2D NumPy array, or object that can be cast to 2D NumPy array.

    :param x0: Lowermost leftmost coordinate to plot. If passed, resampling will
               be performed.
    :type x0: 2D array or list

    :param x1: Uppermost rightmost coordinate to plot. If passed, resampling will
               be performed.
    :type x1: 2D array or list

    :param coordiantes: Coordinates to use for the plot. Used only if data is a
                        NumPy array.
    :type coordinates: 2D array or list

    :param shape: Resolution of the image. This parameter is used if resampling
                  is needed or requested.
    :type shape: tuple or list

    :param iteration: Iteration to plot. Relevant only if data is a
                      :py:class:`~.BaseOneGridData`.
    :type iteration: int

    :param resample: If resampling has to be done, do bilinear resampling at the
                     level of the grid data. If not passed, use nearest neighbors.
    :type resample: bool

    :param logscale: If True, take the log10 of the data before plotting.
    :type logscale: bool

    :param colorbar: If True, add a colorbar.
    :type colorbar: bool

    :param vmin: Remove all the data below this value. If logscale, this has to
                 be the log10.
    :type vmin: float
    :param vmax: Remove all the data above this value. If logscale, this has to
                 be the log10.
    :type vmax: float

    :param xlabel: Label of the x axis. If None (or not passed), no label is
                   placed.
    :type xlabel: str

    :param ylabel: Label of the y axis. If None (or not passed), no label is
                   placed.
    :type ylabel: str

    :param aspect_ratio: Aspect ratio of the plot, as passed to the function
                         ``set_aspect_ratio`` in matplotlib.
    :type aspect_ratio: str

    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``

    :param kwargs: All the unknown arguments are passed to ``imshow``.
    :type kwargs: dict

    """
    # This function is a convinence function around _plot_grid.
    return _plot_grid(data, plot_type="contourf", **kwargs)


def plot_color(data, **kwargs):
    """Plot the given data.

    You can pass (everything is processed by :py:func:`~.preprocess_plot_grid` so
    that at the end we have a 2D NumPy array):
    - A 2D NumPy array,
    - A :py:class:`~.UniformGridData`,
    - A :py:class:`~.HierarchicalGridData`,
    - A :py:class:`~.BaseOneGridFunction`.

    Depending on what you pass, you might need additional arguments.

    If you pass a :py:class:`~.BaseOneGridFunction`, you need also to pass
    ``iteration``, and ``shape``. If you pass
    :py:class:`~.HierarchicalGridData`, you also need to pass ``shape``. In all
    cases you can also pass ``x0`` and ``x1`` to define origin and corner of the
    grid. You can pass the option ``resample=True`` if you want to do bilinear
    resampling at the grid data level, otherwise, nearest neighbor resampling is
    done. When you pass the NumPy array, passing ``coordinates`` will argument
    will make sure that those coordinates are used.

    All the unknown arguments are passed to ``imshow``.

    .. note

       Read the documentation for a concise table on what arguments are
       supported.

    :param data: Data that has to be plotted. The function expects a 2D NumPy
                 array, but the decorator :py:func:`~.preprocess_plot_grid`
                 allows it to take different kind of data.
    :type data: 2D NumPy array, or object that can be cast to 2D NumPy array.

    :param x0: Lowermost leftmost coordinate to plot. If passed, resampling will
               be performed.
    :type x0: 2D array or list

    :param x1: Uppermost rightmost coordinate to plot. If passed, resampling will
               be performed.
    :type x1: 2D array or list

    :param coordiantes: Coordinates to use for the plot. Used only if data is a
                        NumPy array.
    :type coordinates: 2D array or list

    :param shape: Resolution of the image. This parameter is used if resampling
                  is needed or requested.
    :type shape: tuple or list

    :param iteration: Iteration to plot. Relevant only if data is a
                      :py:class:`~.BaseOneGridData`.
    :type iteration: int

    :param resample: If resampling has to be done, do bilinear resampling at the
                     level of the grid data. If not passed, use nearest neighbors.
    :type resample: bool

    :param logscale: If True, take the log10 of the data before plotting.
    :type logscale: bool

    :param colorbar: If True, add a colorbar.
    :type colorbar: bool

    :param vmin: Remove all the data below this value. If logscale, this has to
                 be the log10.
    :type vmin: float
    :param vmax: Remove all the data above this value. If logscale, this has to
                 be the log10.
    :type vmax: float

    :param xlabel: Label of the x axis. If None (or not passed), no label is
                   placed.
    :type xlabel: str

    :param ylabel: Label of the y axis. If None (or not passed), no label is
                   placed.
    :type ylabel: str

    :param aspect_ratio: Aspect ratio of the plot, as passed to the function
                         ``set_aspect_ratio`` in matplotlib.
    :type aspect_ratio: str

    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``

    :param kwargs: All the unknown arguments are passed to ``imshow``.
    :type kwargs: dict

    """
    # This function is a convinence function around _plot_grid.
    return _plot_grid(data, plot_type="color", **kwargs)


def plot_contour(data, levels=5, **kwargs):
    """Plot the given data drawing the contours.

    You can pass (everything is processed by :py:func:`~.preprocess_plot_grid` so
    that at the end we have a 2D NumPy array):
    - A 2D NumPy array,
    - A :py:class:`~.UniformGridData`,
    - A :py:class:`~.HierarchicalGridData`,
    - A :py:class:`~.BaseOneGridFunction`.

    Depending on what you pass, you might need additional arguments.

    If you pass a :py:class:`~.BaseOneGridFunction`, you need also to pass
    ``iteration``, and ``shape``. If you pass
    :py:class:`~.HierarchicalGridData`, you also need to pass ``shape``. In all
    cases you can also pass ``x0`` and ``x1`` to define origin and corner of the
    grid. You can pass the option ``resample=True`` if you want to do bilinear
    resampling at the grid data level, otherwise, nearest neighbor resampling is
    done. When you pass the NumPy array, you also have to pass the
    ``coordinates``.

    ``levels`` can be an integer (the number of levels), or an array with the
    specific values where to put the levels.

    All the unknown arguments are passed to ``contour``.

    .. note

       Read the documentation for a concise table on what arguments are
       supported.

    :param data: Data that has to be plotted. The function expects a 2D NumPy
                 array, but the decorator :py:func:`~.preprocess_plot_grid`
                 allows it to take different kind of data.
    :type data: 2D NumPy array, or object that can be cast to 2D NumPy array.

    :param x0: Lowermost leftmost coordinate to plot. If passed, resampling will
               be performed.
    :type x0: 2D array or list

    :param x1: Uppermost rightmost coordinate to plot. If passed, resampling will
               be performed.
    :type x1: 2D array or list

    :param coordiantes: Coordinates to use for the plot. Used only if data is a
                        NumPy array.
    :type coordinates: 2D array or list

    :param shape: Resolution of the image. This parameter is used if resampling
                  is needed or requested.
    :type shape: tuple or list

    :param iteration: Iteration to plot. Relevant only if data is a
                      :py:class:`~.BaseOneGridData`.
    :type iteration: int

    :param resample: If resampling has to be done, do bilinear resampling at the
                     level of the grid data. If not passed, use nearest neighbors.
    :type resample: bool

    :param logscale: If True, take the log10 of the data before plotting.
    :type logscale: bool

    :param colorbar: If True, add a colorbar.
    :type colorbar: bool

    :param vmin: Remove all the data below this value. If logscale, this has to
                 be the log10.
    :type vmin: float
    :param vmax: Remove all the data above this value. If logscale, this has to
                 be the log10.
    :type vmax: float

    :param xlabel: Label of the x axis. If None (or not passed), no label is
                   placed.
    :type xlabel: str

    :param ylabel: Label of the y axis. If None (or not passed), no label is
                   placed.
    :type ylabel: str

    :param levels: If int, the number of levels, if array, the specific levels
                   where to place the contour lines.
    :type levels: int or list

    :param aspect_ratio: Aspect ratio of the plot, as passed to the function
                         ``set_aspect_ratio`` in matplotlib.
    :type aspect_ratio: str

    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``

    :param kwargs: All the unknown arguments are passed to ``imshow``.
    :type kwargs: dict

    """
    # This function is a convinence function around _plot_grid.
    return _plot_grid(data, plot_type="contour", levels=levels, **kwargs)


@preprocess_plot
def plot_colorbar(
    mpl_artist,
    figure=None,
    axis=None,
    label=None,
    where="right",
    size="5%",
    pad=0.25,
    **kwargs,
):
    """Add a colorbar to an existing image.

    :param mpl_artist: Image from which to generate the colorbar.
    :type mpl_artist: ``matplotlib.cm.ScalarMappable``
    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``
    :param label: Label to place near the colorbar.
    :type label: str
    :param where: Where to place the colorbar (left, right, bottom, top).
    :type where: str
    :param size: Width of the colorbar with respect to ``axis``.
    :type size: float
    :param pad: Pad between the colorbar and ``axis``.
    :type pad: float

    """
    # The next two lines guarantee that the colorbar is the same size as
    # the plot. From https://stackoverflow.com/a/18195921
    divider = make_axes_locatable(axis)
    cax = divider.append_axes(where, size=size, pad=pad)
    cb = plt.colorbar(mpl_artist, cax=cax, **kwargs)
    if label is not None:
        cb.set_label(label)

    # When we draw a colorbar, that changes the selected axis. We do not
    # want that, so we select back the original one.
    plt.sca(axis)

    return cb


@preprocess_plot
def plot_components_boundaries(
    hierarchical_data, figure=None, axis=None, remove_ghosts=True, **kwargs
):
    """Plot the boundaries of all the components available in the data.

    If the components in a given refinement levels can be merged into a single
    one, they be.

    By default, the grids are plotted with black boundaries. This can be
    customized passing the ``edgecolor`` argument.

    :param hierarchical_data: 2D :py:class:`~.HierarchicalGridData` from which
                              to extract the grid structure.
    :type hierarchical_data: :py:class:`~.HierarchicalGridData`,

    :param remove_ghosts: If True, ghosts zones are not included in the plotted
                          grids.
    :type remove_ghosts: bool

    :param kwargs: All the unknown arguments are passed to ``Rectangle``.
    :type kwargs: dict

    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``

    """
    if not isinstance(hierarchical_data, gd.HierarchicalGridData):
        raise TypeError("The input has to be a HierarchicalGridData")

    if hierarchical_data.num_dimensions != 2:
        raise ValueError("Only 2D HierarchicalGridData can be plotted")

    # Add default color
    if "edgecolor" not in kwargs:
        kwargs["edgecolor"] = "black"

    for _1, _2, comp in hierarchical_data:
        # grid is the UniformGrid of the component under consideration with or
        # without ghost zones depending on the value of remove_ghosts
        grid = comp.grid.ghost_zones_removed() if remove_ghosts else comp.grid

        # comp.highest_vertex and comp.lowest_vertex are 2D NumPy arrays, so
        # comp.highest_vertex - comp.lowest_vertex is the length of the
        # component along the two directions. We unpack it to width and height.
        # We need vertices as opposed to x0 and x1 because we want to take into
        # account the size of the boundary cells
        width, height = grid.highest_vertex - grid.lowest_vertex
        axis.add_patch(
            Rectangle(
                grid.lowest_vertex, width, height, facecolor="none", **kwargs
            )
        )


# HORIZONS


@preprocess_plot
def plot_horizon(
    shape,
    color=None,
    edgecolor=None,
    alpha=None,
    figure=None,
    axis=None,
    **kwargs,
):
    """Plot outline of horizon in 2D.

    Unknown arguments are passed to the ``fill`` function.

    :param shape: Shape of the horizon as returned by
                  `~.shape_outline_at_iteration` or `~.shape_outline_at_time`.
    :type shape: two NumPy arrays
    :param color: Color of the interior of the horizon.
    :type color: color as supported by Matplotlib
    :param edgecolor: Color of the edge of the horizon.
    :type edgecolor: color as supported by Matplotlib
    :param alpha:  Opacity of the horizon.
    :type alpha:  float

    """
    return axis.fill(
        *shape, color=color, edgecolor=edgecolor, alpha=alpha, **kwargs
    )


@preprocess_plot
def _plot_horizon_on_plane(
    horizon,
    plot_type="iteration",
    iteration=None,
    time=None,
    plane="xy",
    time_tolerance=1e-10,
    color=None,
    edgecolor=None,
    alpha=None,
    figure=None,
    axis=None,
    **kwargs,
):
    """Backend for :py:func:`~.plot_horizon_on_plane_at_iteration` and
    :py:func:`~.plot_horizon_on_plane_at_time`.

    Unknown arguments are passed to :py:func:`~.plot_horizon`.

    :param horizon: Horizon to plot.
    :type horizon: :py:class:`~.OneHorizon`
    :param iteration: Iteration to plot.
    :type iteration: int
    :param plane: Plane where to plot (options: `xy`, `xz`, `yz`)
    :type plane: str
    :param time: Time to plot.
    :type time: float
    :param time_tolerance: Tolerance in the determination of the time.
    :type time_tolerance: float
    :param color: Color of the interior of the horizon.
    :type color: color as supported by Matplotlib
    :param edgecolor: Color of the edge of the horizon.
    :type edgecolor: color as supported by Matplotlib
    :param alpha:  Opacity of the horizon.
    :type alpha:  float
    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``
    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``
    """

    cut = {
        "xy": (None, None, 0),
        "xz": (None, 0, None),
        "yz": (0, None, None),
    }

    if plane not in cut:
        raise ValueError(f"Plane has to be one of {list(cut.keys())}")

    if plot_type == "iteration":
        shape = horizon.shape_outline_at_iteration(iteration, cut[plane])
    elif plot_type == "time":
        shape = horizon.shape_outline_at_time(
            time, cut[plane], tolerance=time_tolerance
        )

    if shape is None:
        raise RuntimeError(
            "No outline found on given plane. "
            "This might be due to lack of shape interpolation"
        )

    return plot_horizon(
        shape,
        color=color,
        edgecolor=edgecolor,
        alpha=alpha,
        figure=figure,
        axis=axis,
        **kwargs,
    )


@preprocess_plot
def plot_horizon_on_plane_at_iteration(
    horizon,
    iteration,
    plane,
    color=None,
    edgecolor=None,
    alpha=None,
    figure=None,
    axis=None,
    **kwargs,
):
    """Plot outline of horizon in 2D on a given plane at a given iteration.

    Unknown arguments are passed to :py:func:`~.plot_horizon`.

    .. warning::

       When you take a cross section (an outline) of an horizon, ``kuibit``
       finds points that are within a threshold to the plane that cuts the
       surface. However, the way points are distributed on apparent horizons is
       highly non-uniform. So, if you are cutting the horizon along an axis that
       is not one of the coordinate ones (for the horizon), it is likely that
       too few points will be close enough to the intersecting plane, resulting
       in a malformed or absent outline. In some distant future, ``kuibit`` will
       perform interpolations to solve this problem.

    :param horizon: Horizon to plot.
    :type horizon: :py:class:`~.OneHorizon`
    :param iteration: Iteration to plot.
    :type iteration: int
    :param plane: Plane where to plot (options: `xy`, `xz`, `yz`)
    :type plane: str
    :param color: Color of the interior of the horizon.
    :type color: color as supported by Matplotlib
    :param edgecolor: Color of the edge of the horizon.
    :type edgecolor: color as supported by Matplotlib
    :param alpha:  Opacity of the horizon.
    :type alpha:  float
    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``
    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``
    """
    return _plot_horizon_on_plane(
        horizon,
        plot_type="iteration",
        plane=plane,
        iteration=iteration,
        color=color,
        edgecolor=edgecolor,
        alpha=alpha,
        figure=figure,
        axis=axis,
        **kwargs,
    )


@preprocess_plot
def plot_horizon_on_plane_at_time(
    horizon,
    time,
    plane,
    time_tolerance=1e-10,
    color=None,
    edgecolor=None,
    alpha=None,
    figure=None,
    axis=None,
    **kwargs,
):
    """Plot outline of horizon in 2D on a given plane at a given time.

    Unknown arguments are passed to :py:func:`~.plot_horizon`.

    .. warning::

       When you take a cross section (an outline) of an horizon, ``kuibit``
       finds points that are within a threshold to the plane that cuts the
       surface. However, the way points are distributed on apparent horizons is
       highly non-uniform. So, if you are cutting the horizon along an axis that
       is not one of the coordinate ones (for the horizon), it is likely that
       too few points will be close enough to the intersecting plane, resulting
       in a malformed or absent outline. In some distant future, ``kuibit`` will
       perform interpolations to solve this problem.

    :param horizon: Horizon to plot.
    :type horizon: :py:class:`~.OneHorizon`
    :param time: Time to plot.
    :type time: float
    :param time_tolerance: Tolerance in the determination of the time.
    :type time_tolerance: float
    :param plane: Plane where to plot (options: `xy`, `xz`, `yz`)
    :type plane: str
    :param color: Color of the interior of the horizon.
    :type color: color as supported by Matplotlib
    :param edgecolor: Color of the edge of the horizon.
    :type edgecolor: color as supported by Matplotlib
    :param alpha:  Opacity of the horizon.
    :type alpha:  float
    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``
    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``
    """

    return _plot_horizon_on_plane(
        horizon,
        plane=plane,
        plot_type="time",
        time_tolerance=time_tolerance,
        time=time,
        color=color,
        edgecolor=edgecolor,
        alpha=alpha,
        figure=figure,
        axis=axis,
        **kwargs,
    )
