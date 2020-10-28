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

import os
import unittest

import numpy as np

from postcactus import cactus_grid_functions as cg
from postcactus import simdir as sd


class TestCactusGrid(unittest.TestCase):
    def setUp(self):
        sim = sd.SimDir("tests/tov")
        self.gd = cg.GridFunctionsDir(sim)

    def test_init_griddir(self):

        # Not a SimDir
        with self.assertRaises(TypeError):
            cg.GridFunctionsDir(0)

    def test_GridFunctionsDir_string_or_tuple(self):

        # Test not recognized dimension
        with self.assertRaises(ValueError):
            self.gd._string_or_tuple_to_dimension_index("hey")

        self.assertEqual(
            self.gd._string_or_tuple_to_dimension_index("x"), (0,)
        )

        self.assertEqual(
            self.gd._string_or_tuple_to_dimension_index("xyz"), (0, 1, 2)
        )

        self.assertEqual(
            self.gd._string_or_tuple_to_dimension_index((0, 1, 2)), (0, 1, 2)
        )
