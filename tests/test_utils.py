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

import logging
import unittest

from kuibit import utils as kbu


class TestUtils(unittest.TestCase):
    def test_get_logger(self):

        # Check name
        self.assertEqual(kbu.get_logger().name, "kuibit")

    def test_set_verbosity(self):

        kbu.set_verbosity("INFO")
        self.assertEqual(kbu.get_logger().level, logging.INFO)
        kbu.set_verbosity("DEBUG")
        self.assertEqual(kbu.get_logger().level, logging.DEBUG)
        kbu.set_verbosity("DEBUG", kbu.get_logger("bob"))
        self.assertEqual(kbu.get_logger("bob").level, logging.DEBUG)

        with self.assertRaises(TypeError):
            kbu.set_verbosity("ERROR")
