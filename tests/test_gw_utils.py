#!/usr/bin/env python3

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
