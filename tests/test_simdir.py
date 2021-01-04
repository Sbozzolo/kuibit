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

import unittest

from postcactus import simdir as sd


class TestSimDir(unittest.TestCase):
    def setUp(self):

        # We use the output of a simple simulation to test that everything
        # is okay
        self.sim = sd.SimDir("tests/tov")

    def test__sanitize_path(self):

        with self.assertRaises(RuntimeError):
            # Not existing folder
            self.sim._sanitize_path("bubu")

        with self.assertRaises(RuntimeError):
            # File, not folder
            self.sim._sanitize_path("test_simdir")

    def test__scan_folders(self):

        # 5 par files: 2 in each output folders, and 1 in the main SIMFACTORY
        self.assertEqual(len(self.sim.parfiles), 5)

        # 2 out files, one in each folder
        self.assertEqual(len(self.sim.logfiles), 2)

        # 2 err files, one in each folder
        self.assertEqual(len(self.sim.errfiles), 2)

        # 5 dirs: tov, tov/output-000i, tov/output-000i/static_tov
        self.assertEqual(len(self.sim.dirs), 5)

        # find . -type f | grep -v "SIMFACTORY" | grep -v "NODES" | wc -l
        # 446
        self.assertEqual(len(self.sim.allfiles), 446)

        # Checking max_depth
        sim_max_depth = sd.SimDir("tests/tov", max_depth=2)

        # 3 par files: 1 in each output folders, and 1 in the main SIMFACTORY
        self.assertEqual(len(sim_max_depth.parfiles), 3)

        # 2 out files, one in each folder
        self.assertEqual(len(sim_max_depth.logfiles), 2)

        # 2 err files, one in each folder
        self.assertEqual(len(sim_max_depth.errfiles), 2)

        # 3 dirs: tov, tov/output-000i
        self.assertEqual(len(sim_max_depth.dirs), 3)

        # find . -maxdepth 2 -type f | grep -v "SIMFACTORY" | grep -v "NODES" | wc -l
        # 8
        self.assertEqual(len(sim_max_depth.allfiles), 8)

        # Check that all the expected components are in
        # the string representation
        #
        self.assertIn(self.sim.ts.__str__(), self.sim.__str__())
        self.assertIn(self.sim.multipoles.__str__(), self.sim.__str__())

        # Test for a simdir with no information
        # This is a fake folder
        empty_sim = sd.SimDir("postcactus")
        self.assertIn("No horizon found", empty_sim.__str__())
