#!/usr/bin/env python3

# Copyright (C) 2021 Gabriele Bozzola
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

import os
import unittest

from mayavi import mlab

from kuibit import visualize_mayavi as viz

class TestVisualizeMayavi(unittest.TestCase):

    def test_enable_mayavi_offscreen_rendering(self):
        viz.disable_interactive_window()

        self.assertTrue(mlab.options.offscreen)

    def test_save(self):

        mlab.test_contour3d()
        viz.save("test", "png")
        self.assertTrue(os.path.exists("test.png"))
        os.remove("test.png")
