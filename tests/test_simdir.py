#!/usr/bin/env python3

# Copyright (C) 2020-2025 Gabriele Bozzola
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
import pickle
import unittest

from kuibit import simdir as sd


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
        self.assertEqual(len(self.sim.allfiles), 450)

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
        empty_sim = sd.SimDir("kuibit")
        self.assertIn("No horizon found", empty_sim.__str__())

        # Test symlink
        sim_with_symlink = sd.SimDir("tests/tov", ignore_symlinks=False)
        self.assertEqual(len(sim_with_symlink.allfiles), 451)

    def test_pickle(self):
        path = "/tmp/sim.pickle"

        self.sim.save(path)

        loaded_sim = sd.load_SimDir(path)

        self.assertCountEqual(self.sim.__dict__, loaded_sim.__dict__)

        # Test load from pickle

        loaded_sim2 = sd.SimDir(
            "tests/tov", max_depth=0, pickle_file="/tmp/sim.pickle"
        )

        self.assertCountEqual(self.sim.__dict__, loaded_sim2.__dict__)

        # Test as a context manager

        with sd.SimDir("tests/tov", pickle_file="/tmp/sim.pickle") as sim:
            self.assertCountEqual(self.sim.__dict__, sim.__dict__)
            # Make a change
            sim.max_depth = 10

        loaded_sim3 = sd.load_SimDir(path)
        self.assertEqual(loaded_sim3.max_depth, 10)

        os.remove(path)

        # Test with pickle not being a simdir

        with open(path, "wb") as file_:
            pickle.dump(1, file_)

        with self.assertRaises(RuntimeError):
            sd.load_SimDir(path)

        os.remove(path)

    def test_rescan(self):
        # This is not a real test ...
        self.sim.rescan()
