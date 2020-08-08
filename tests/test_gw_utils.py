#!/usr/bin/env python3

import unittest
from postcactus import gw_utils as gwu

class TestTimeseries(unittest.TestCase):

    def test_luminosity_distance_to_redshift(self):

        self.assertAlmostEqual(gwu.luminosity_distance_to_redshift(450),
                               0.0948809)

        with self.assertRaises(RuntimeError):
            print(gwu.luminosity_distance_to_redshift(1e15,
                                                      Omega_m=0))
