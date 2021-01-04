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

"""Tests for postcactus.unitconv
"""

import unittest

from postcactus import unitconv as uc


class TestUnitconv(unittest.TestCase):
    def test_constants(self):
        """Test that the constants are the ones we expect.

        We just test for the correct order of magnitude to catch typing errors.
        You should recognize the values.

        """

        # assertAlmostEqual doesn't play well with big numbers, so
        # we compare the ratio to 1
        c = 299.8e6
        self.assertAlmostEqual(uc.C_SI / c, 1, places=3)

        G = 6.674e-11
        self.assertAlmostEqual(uc.G_SI / G, 1, places=3)

        M_sun = 1.988e30
        self.assertAlmostEqual(uc.M_SOL_SI / M_sun, 1, places=3)
        self.assertTrue(uc.M_SOL_SI == uc.M_SUN_SI)

        pc = 3.086e16
        self.assertAlmostEqual(uc.PARSEC_SI / pc, 1, places=3)
        self.assertTrue(uc.MEGAPARSEC_SI == 1e6 * uc.PARSEC_SI)
        self.assertTrue(uc.GIGAPARSEC_SI == 1e9 * uc.PARSEC_SI)

        ly = 9.46e15
        self.assertAlmostEqual(uc.LIGHTYEAR_SI / ly, 1, places=3)

        h0 = 67.63
        h0 = h0 / uc.MEGAPARSEC_SI / 1e-3
        self.assertAlmostEqual(h0 / uc.H0_SI, 1, places=3)

    def test_Units(self):
        """Test that Units conversion are well coded."""
        # This a made-up unit system with non-trivial, but simple, conversions
        # Hardcoded, so that I have to go through all the computations
        CU = uc.Units(1e-2, 3e-3, 5e-4)
        self.assertAlmostEqual(CU.length, 1e-2)
        self.assertAlmostEqual(CU.time, 3e-3)
        self.assertAlmostEqual(CU.mass, 5e-4)
        self.assertAlmostEqual(CU.velocity, 3.333333333333333)
        self.assertAlmostEqual(CU.accel, 1111.1111111111)
        self.assertAlmostEqual(CU.force, 0.5555555555)
        self.assertAlmostEqual(CU.area, 1e-4)
        self.assertAlmostEqual(CU.volume, 1e-6)
        self.assertAlmostEqual(CU.density, 500)
        self.assertAlmostEqual(CU.pressure, 5555.55555555)
        self.assertAlmostEqual(CU.power, 1.85185185)
        self.assertAlmostEqual(CU.energy, 0.00555555555)
        self.assertAlmostEqual(CU.energy_density, 5555.55555555)
        self.assertAlmostEqual(CU.angular_moment, 1.666666666e-5)
        self.assertAlmostEqual(CU.moment_inertia, 5e-10)

    def test_geom_ulength(self):
        """Test geom_ulength"""
        CU = uc.geom_ulength(1476.6436)  # m
        self.assertAlmostEqual(CU.mass / uc.M_SOL_SI, 1, places=3)

    def test_geom_umass(self):
        """Test geom_umass"""
        CU = uc.geom_umass(1)  # 1 kg
        self.assertAlmostEqual(CU.length, 7.426e-28, places=3)

    def test_geom_umass_msun(self):
        """Test geom_umass_msun"""
        CU = uc.geom_umass_msun(1)
        self.assertAlmostEqual(CU.length, 1476.6436, places=3)
