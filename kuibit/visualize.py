#!/usr/bin/env python3

# Copyright (C) 2020 Gabriele Bozzola
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

"""The :py:mod:`~.visualize` module provides functions to plot ``PostCactus``
objects with matplotlib.

"""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from postcactus import grid_data as gd
from postcactus.cactus_grid_functions import BaseOneGridFunction


def setup_matplotlib():
    """Setup matplotlib with some reasonable defaults for better plots.

    Matplotlib behaves differently on different machines. With this, we make
    sure that we set all the relevant paramters that we care of.

    This is highly opinionated.
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
        }
    )


def _preprocess_plot(func):
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

    func has to take as keyword arguments:
    1. 'axis=None', where the plot will be plot, or plt.gca() if None

    """

    def inner(data, *args, **kwargs):
        # Setdetault addes the key if it is not already there
        kwargs.setdefault("axis", plt.gca())
        return func(data, *args, **kwargs)

    return inner


def _preprocess_plot_grid(func):
    """Decorator to set-up plot functions that plot grid data.

    This dectorator exends _preprocess_plot for specific functions.

    1. It handles differt types to plot what intuitively one would want to
       plot.
    1a. If the data is a numpy array with shape 2, just pass the data,
        otherwise raise an error
    1b. If the data is a numpy array, just pass the data.
    1c. If data is UniformGridData, pass the data and the coordinates.
    1d. If data is HierarchicalGridData, read resample it to the given grid,
        then pass do 1c.
    1e. If data is a BaseOneGridFunction, we read the iteration and pass to
        1d.

    func has to take as keyword arguments (in addition to the ones in
    _preprocess_plot):
    1. 'data'. data will be passed as a numpy array, unless it is
               already so.
    2. 'coordinates=None'. coordinates will be passed as a list of numpy
                           arrays, unless it is not None. Each numpy
                           array is the coordinates along one axis.

    """

    @_preprocess_plot
    def inner(data, *args, **kwargs):
        # The flow is: We check if data is BaseOneGridFunction or derived. If
        # yes, we read the requested iteration. Then, we check if data is
        # HierachicalGridData, if yes, we resample to UniformGridData. Then we
        # work with UniformGridData and handle coordinates, finally we work
        # with numpy arrays, which is what we pass to the function.

        def not_in_kwargs_or_None(attr):
            """This is a helper function to see if the user passed an attribute
            or if the attribute is None
            """
            return attr not in kwargs or kwargs[attr] is None

        if isinstance(data, BaseOneGridFunction):
            if not_in_kwargs_or_None("iteration"):
                raise TypeError(
                    "Data has multiple iterations, specify what do you want to plot"
                )

            # Overwrite data with HierarchicalGridData
            data = data[kwargs["iteration"]]

        if isinstance(data, gd.HierarchicalGridData):
            if not_in_kwargs_or_None("shape"):
                raise TypeError(
                    "The data must be resampled but the shape was not provided"
                )

            # If x0 or x1 are None, we use the ones of the grid
            if not_in_kwargs_or_None("x0"):
                x0 = data.x0
            else:
                x0 = kwargs["x0"]

            if not_in_kwargs_or_None("x1"):
                x1 = data.x1
            else:
                x1 = kwargs["x1"]

            if not_in_kwargs_or_None("resample"):
                resample = False
            else:
                resample = kwargs["resample"]

            # Overwrite data with UniformGridData
            data = data.to_UniformGridData(
                shape=kwargs["shape"], x0=x0, x1=x1, resample=resample
            )

        if isinstance(data, gd.UniformGridData):
            # We check if the user has passed coordinates too.
            if "coordinates" in kwargs and kwargs["coordinates"] is not None:
                warnings.warn(
                    "Ignoring provided coordinates (data is UniformGridData)."
                    " To specify boundaries use x0 and x1."
                )
            resampling = False

            if not_in_kwargs_or_None("x0"):
                x0 = data.x0
            else:
                x0 = kwargs["x0"]
                resampling = True

            if not_in_kwargs_or_None("x1"):
                x1 = data.x1
            else:
                x1 = kwargs["x1"]
                resampling = True

            if resampling and not_in_kwargs_or_None("shape"):
                raise TypeError(
                    "The data must be resampled but the shape was not provided"
                )

            if resampling:
                new_grid = gd.UniformGrid(shape=kwargs["shape"], x0=x0, x1=x1)
                data = data.resampled(new_grid)

            kwargs["coordinates"] = data.coordinates_from_grid()
            # Overwrite data with numpy array
            data = data.data_xyz

        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError("Only 2-dimensional data can be plotted")

        # TODO: Check that coordinates are good

        # We remove what we don't need from kwargs, so that it is not
        # accidentally passed to the function
        if "shape" in kwargs:
            del kwargs["shape"]
        if "x0" in kwargs:
            del kwargs["x0"]
        if "x1" in kwargs:
            del kwargs["x1"]
        if "iteration" in kwargs:
            del kwargs["iteration"]
        if "resample" in kwargs:
            del kwargs["resample"]
        return func(data, *args, **kwargs)

    return inner


# All the difficult stuff is in _preprocess_plot_grid
@_preprocess_plot_grid
def plot_contourf(
    data, axis=None, coordinates=None, xlabel=None, ylabel=None, **kwargs
):
    """Plot 2D grid from numpy array, UniformGridData, HierarhicalGridData,
    or OneGridFunction.

    Read the full documentation to see how to use this function.
    """

    # Considering all the effort put in _preprocess_plot_grid, we we can plot
    # as we were plotting normal numpy arrays.
    if coordinates is None:
        cf = axis.imshow(data, **kwargs)
    else:
        cf = axis.contourf(*coordinates, data, **kwargs)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    plt.draw()
    return cf
