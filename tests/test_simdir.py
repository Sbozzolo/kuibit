#!/usr/bin/env python3

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
        # 341
        self.assertEqual(len(self.sim.allfiles), 341)

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
        # 7
        self.assertEqual(len(sim_max_depth.allfiles), 7)
