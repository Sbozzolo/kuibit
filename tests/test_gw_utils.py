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

import unittest

from postcactus import gw_utils as gwu


class TestGWUtils(unittest.TestCase):

    def test_luminosity_distance_to_redshift(self):

        self.assertAlmostEqual(gwu.luminosity_distance_to_redshift(450),
                               0.0948809)

        with self.assertRaises(RuntimeError):
            print(gwu.luminosity_distance_to_redshift(1e15,
                                                      Omega_m=0))

    def test_sYlm(self):
        # Test values froom kerrgeodesic_gw

        self.assertAlmostEqual(gwu.sYlm(0, -1, 0, 0, 1), 0)
        self.assertAlmostEqual(gwu.sYlm(0, 1, 2, 0, 1), 0)
        self.assertAlmostEqual(gwu.sYlm(-2, 2, 1, 1.0, 2.0),
                               -0.170114676286891 + 0.371707349012686j)
        self.assertAlmostEqual(gwu.sYlm(-2, 2, 1, 1.5, 2.0),
                               -0.140181365376761 + 0.306301871434652j)
        self.assertAlmostEqual(gwu.sYlm(-2, 2, -1, 1.5, 2.0),
                               -0.121659476911011 - 0.265830806794102j)
        self.assertAlmostEqual(gwu.sYlm(-2, 2, -2, 1.5, 2.0),
                               -0.0890098785065999 + 0.103057531674292j)
        self.assertAlmostEqual(gwu.sYlm(0, 2, 1, 1.5, 2.0),
                               0.0226845879069160 - 0.0495667288582717j)
        self.assertAlmostEqual(gwu.sYlm(0, 3, 1, 1.5, 2.0),
                               -0.130797156679223 + 0.285797001345366j)

    def test_ra_dec_to_theta_phi(self):

        time = "2015-09-14 09:50:45"
        angle = gwu.ra_dec_to_theta_phi(8, -70, time)

        self.assertAlmostEqual(angle.hanford[0], 2.36971740)
        self.assertAlmostEqual(angle.hanford[1], 3.01937907)

        self.assertAlmostEqual(angle.livingston[0], 1.95758940)
        self.assertAlmostEqual(angle.livingston[1], 1.61110394)

        self.assertAlmostEqual(angle.virgo[0], 2.02280322)
        self.assertAlmostEqual(angle.virgo[1], -3.00853112)

    def test_antenna_responses(self):

        antenna_gw150914 = gwu.antenna_responses(8, -70,
                                                 "2015-09-14 09:50:45")

        # This test is extremely weak: the numbers that are here were
        # obtained with the function itself
        self.assertAlmostEqual(antenna_gw150914.hanford[0], 0.173418558)
        self.assertAlmostEqual(antenna_gw150914.hanford[1], 0.734266762)
        self.assertAlmostEqual(antenna_gw150914.livingston[0], 0.030376784)
        self.assertAlmostEqual(antenna_gw150914.livingston[1], -0.569292709)
        self.assertAlmostEqual(antenna_gw150914.virgo[0], -0.11486789)
        self.assertAlmostEqual(antenna_gw150914.virgo[1], 0.57442590)
