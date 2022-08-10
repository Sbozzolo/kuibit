#!/usr/bin/env python3

# Copyright (C) 2022 Gabriele Bozzola
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

from kuibit.simdir import SimDir


class TestCactusTimers(unittest.TestCase):
    def test_nometadata(self):

        nometadata = SimDir("tests/tov").twopunctures

        with self.assertRaises(RuntimeError):
            nometadata["initial-ADM-energy"]

        self.assertEqual(nometadata.keys(), {}.keys())

        self.assertFalse(nometadata.has_metadata)

    def test_metadata(self):

        sim = SimDir("tests/gwsample")

        met = sim.twopunctures

        self.assertTrue(met.has_metadata)

        self.assertEqual(len(met.keys()), 35)

        self.assertEqual(met["initial-ADM-energy"], 0.9195476019513054711)

        # Missing key
        with self.assertRaises(KeyError):
            met["bob"]

        # Test when there are two metadata files (should never happen)

        # We fake this by manipulating the allfiles field in sim
        sim_broken = SimDir("tests/gwsample")
        sim_broken.allfiles.append("TwoPunctures.bbh")

        with self.assertRaises(RuntimeError):
            sim_broken.twopunctures
