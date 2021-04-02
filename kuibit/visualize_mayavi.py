#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
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

"""The :py:mod:`~.visualize_mayavi` module provides functions to perform 3D
renderings of ``kuibit`` objects using mayavi.

The functions provided are:
- :py:func:`~.enable_mayavi_offscreen_rendering` disables mayavi's interactive
  window. This function must be called in batch scripts.

"""

from mayavi import mlab
import numpy as np

from kuibit.cactus_horizons import OneHorizon

def disable_interactive_window():
    """Disable mayavi interactive window.

    mayavi requires a window to be drawn to produce the image. With this
    function, this window is created but is not presented to the user.
    Always invoke this function in scripts!

    """
    mlab.options.offscreen = True

def plot_apparent_horizon(horizon, iteration, color=None, **kwargs):
    """Plot given apparent horizon.

    TODO: This is very tentative and will change in the future!

    Unknown arguments are passed to ``mayavi.mesh``. One of these arguments
    can be ``color``. You can set ``color`` to

    :param horizon: Apparent horizon to plot.
    :type horizon: `~.:py:class:OneHorizon`
    :param color: RGB values of the color of the horizon.
                  ``(1,1,1)`` for a white hole, ``(0,0,0)`` for a black hole.
    :type color: tuple of three floats
    :param iteration: Iteration to plot.
    :type iteration: int
    """
    if not isinstance(horizon, OneHorizon):
        raise TypeError("Invalid horizon")

    # We add this here because one could set colormap instead of color.
    if color is None:
        color = (0, 0, 0)

    # TODO (FEATURE): Add time as parameter alternative to iteration.

    # HACK: We are plotting multiple patches on top of each other.
    #
    # Apparent horizons are described by multiple patches. At the moment, kuibit
    # cannot merge them in a meaningful way. So, we cheat and plot everything
    # with the same color so that it is impossible to see that there are
    # multiple patches.

    # First, we get the patches
    #
    # patches is a list of three elements, the three coordinates. Each coordinate
    # is a list with the various patches describes as a list of np.array.
    patches = horizon.shape_at_iteration(iteration)

    # We must remove one layer from patches, the one that describes the various
    # patches. We do this in a rather brutal way by looping over all the
    # elements.
    #
    # What is going on here? We have an other loop (dim in range(3)) that loops
    # over the three coordinates. The elements of this list comprehension are
    # np.concatenate of the various points in the patch.
    shape_xyz = [
        np.concatenate(patch for patch in patches[dim]) for dim in range(3)
    ]

    # Here we plot!
    return mlab.mesh(*shape_xyz, color=color, **kwargs)


def plot_ah_trajectory(horizon, time=None, **kwargs):
    """Plot the 3D trajectory of the given horizon.

    :param horizon: Apparent horizon to plot.
    :type horizon: `~.:py:class:OneHorizon`
    :param time: If not None, plot up to this time.
    :type time: float or None
    """
    return mlab.plot3d(
        horizon.ah.centroid_x.cropped(end=time).y,
        horizon.ah.centroid_y.cropped(end=time).y,
        horizon.ah.centroid_z.cropped(end=time).y,
        **kwargs,
    )


def add_text_to_figure_corner(text, color=(1, 1, 1), **kwargs):
    """Add text to the bottom right corner of a figure.

    TODO: This is very tentative and will change in the future!

    Unknown arguments are passed to mayavi.text.

    :param text: Text to be inserted.
    :type text: str
    :param color: RGB colors of the text ``(1,1,1)`` for a white,
                  ``(0,0,0)`` for black.
    :type color: tuple of three floats

    :returns: mayavi's Text object
    :rtype: mayavi.Text
    ``"""

    return mlab.text(0.0, 0.93, text, color=color, **kwargs)

def save(outputpath, figure_extension, **kwargs):
    """Save figure to outputpath with figure_extension.

    :param outputpath:  Output path without extension.
    :type outputpath:  str
    :param figure_extension: Extension of the figure to save.
    """
    mlab.savefig(f"{outputpath}.{figure_extension}", **kwargs)
