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

import sys
import unittest

from postcactus import argparse_helper as pah


class TestArgparseHelper(unittest.TestCase):
    def setUp(self):
        self.parser = pah.init_argparse()

    def test_init(self):
        args = self.parser.parse_args([])

        self.assertIs(args.configfile, None)
        self.assertEqual(args.datadir, ".")
        self.assertEqual(args.outdir, ".")
        self.assertFalse(args.verbose)

    def test_get_args(self):
        sys.argv = ["program_name", "--datadir", "test"]

        # Test with args = None
        args = pah.get_args(self.parser)
        self.assertEqual(args.datadir, "test")

        # Tests with non None args
        args2 = pah.get_args(self.parser, ["--outdir", "test2"])
        self.assertEqual(args2.outdir, "test2")

    def test_add_grid_to_parser(self):

        pah.add_grid_to_parser(self.parser)
        # The [] essentially means "use defaults"
        args = pah.get_args(self.parser, [])

        self.assertEqual(args.resolution, 500)
        self.assertEqual(args.plane, "xy")
        self.assertCountEqual(args.origin, [0, 0])
        self.assertCountEqual(args.corner, [1, 1])

    def test_add_figure_to_parser(self):

        pah.add_figure_to_parser(self.parser, default_figname='figure')
        # The [] essentially means "use defaults"
        args = pah.get_args(self.parser, [])

        self.assertEqual(args.figname, 'figure')
