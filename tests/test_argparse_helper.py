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

import sys
import unittest

from kuibit import argparse_helper as kah


class TestArgparseHelper(unittest.TestCase):
    def setUp(self):
        self.parser = kah.init_argparse()

    def test_init(self):
        args = self.parser.parse_args([])

        self.assertIs(args.configfile, None)
        self.assertEqual(args.datadir, ".")
        self.assertEqual(args.outdir, ".")
        self.assertFalse(args.verbose)
        self.assertFalse(args.ignore_symlinks)

    def test_get_args(self):
        sys.argv = ["program_name", "--datadir", "test"]

        # Test with args = None
        args = kah.get_args(self.parser)
        self.assertEqual(args.datadir, "test")

        # Tests with non None args
        args2 = kah.get_args(self.parser, ["--outdir", "test2"])
        self.assertEqual(args2.outdir, "test2")

    def test_add_grid_to_parser(self):

        # Wrong dimensions
        with self.assertRaises(ValueError):
            kah.add_grid_to_parser(self.parser, dimensions=5)

        parser_3D = kah.init_argparse()

        kah.add_grid_to_parser(parser_3D, dimensions=3)
        # The [] essentially means "use defaults"
        args = kah.get_args(parser_3D, [])

        self.assertEqual(args.resolution, 500)
        self.assertCountEqual(args.origin, [0, 0, 0])
        self.assertCountEqual(args.corner, [1, 1, 1])

        # 2D
        parser_2D = kah.init_argparse()

        kah.add_grid_to_parser(parser_2D, dimensions=2)
        # The [] essentially means "use defaults"
        args = kah.get_args(parser_2D, [])

        self.assertEqual(args.plane, "xy")
        self.assertCountEqual(args.origin, [0, 0])
        self.assertCountEqual(args.corner, [1, 1])

        # 1D
        parser_1D = kah.init_argparse()

        kah.add_grid_to_parser(parser_1D, dimensions=1)
        # The [] essentially means "use defaults"
        args = kah.get_args(parser_1D, [])

        self.assertEqual(args.axis, "x")
        self.assertCountEqual(args.origin, [0])
        self.assertCountEqual(args.corner, [1])

    def test_add_figure_to_parser(self):

        kah.add_figure_to_parser(self.parser, default_figname="figure")
        # The [] essentially means "use defaults"
        args = kah.get_args(self.parser, [])

        self.assertEqual(args.figname, "figure")
        self.assertEqual(args.fig_extension, "png")

    def test_add_horizon_to_parser(self):

        kah.add_horizon_to_parser(
            self.parser, color="w", edge_color="r", alpha=0.5, time_tolerance=1
        )

        # The [] essentially means "use defaults"
        args = kah.get_args(self.parser, [])

        self.assertFalse(args.ah_show)
        self.assertEqual(args.ah_color, "w")
        self.assertEqual(args.ah_edge_color, "r")
        self.assertEqual(args.ah_alpha, 0.5)
        self.assertEqual(args.ah_time_tolerance, 1)

    def test_program_name(self):
        sys.argv = ["/tmp/program_name.py", "--datadir", "test"]

        self.assertEqual(kah.get_program_name(), "program_name.py")
