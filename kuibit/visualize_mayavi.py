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

def save(outputpath, figure_extension):
    """Save figure to outputpath with figure_extension.

    :param outputpath:  Output path without extension.
    :type outputpath:  str
    :param figure_extension: Extension of the figure to save.
    """
    mlab.savefig(f"{outputpath}.{figure_extension}")
