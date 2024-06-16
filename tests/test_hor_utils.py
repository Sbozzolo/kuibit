#!/usr/bin/env python3

# Copyright (C) 2022-2024 Gabriele Bozzola
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

import numpy as np

from kuibit import hor_utils as hu
from kuibit.cactus_horizons import OneHorizon
from kuibit.tensor import Vector
from kuibit.timeseries import TimeSeries


class TestHorUtils(unittest.TestCase):
    def setUp(self):
        # Let's set up a quasi-circular inspiral
        times = np.linspace(0, 100, 1000)
        # We add a 0.1 to avoid having identically zero separation
        cen_x = 0.1 + 10 * (times[-1] - times) * np.cos(times)
        cen_y = 0.1 + 10 * (times[-1] - times) * np.sin(times)
        cen_z = np.zeros_like(times)

        self.times = times
        self.cen_x = cen_x
        self.cen_y = cen_y

        # We also need an "area" attribute
        area = np.ones_like(times)

        self.ho1 = OneHorizon(
            qlm_vars=None,
            ah_vars={
                "centroid_x": TimeSeries(times, cen_x),
                "centroid_y": TimeSeries(times, cen_y),
                "centroid_z": TimeSeries(times, cen_z),
                "area": TimeSeries(times, area),
            },
            shape_files=None,
            vtk_files=None,
        )

        # Opposite on the (x, y) plane, and with thrice the area (to make some
        # calculations more interesting, e.g. for the center of mass, which uses
        # the area)
        self.ho2 = OneHorizon(
            qlm_vars=None,
            ah_vars={
                "centroid_x": TimeSeries(times, -cen_x),
                "centroid_y": TimeSeries(times, -cen_y),
                "centroid_z": TimeSeries(times, cen_z),
                "area": TimeSeries(times, 3 * area),
            },
            shape_files=None,
            vtk_files=None,
        )

    def test__two_centroids_as_Vectors(self):
        expected1 = Vector(
            [
                TimeSeries(self.times, self.cen_x),
                TimeSeries(self.times, self.cen_y),
                TimeSeries(self.times, np.zeros_like(self.times)),
            ],
        )

        expected2 = Vector(
            [
                TimeSeries(self.times, -self.cen_x),
                TimeSeries(self.times, -self.cen_y),
                TimeSeries(self.times, np.zeros_like(self.times)),
            ],
        )

        vec1, vec2 = hu._two_centroids_as_Vectors(self.ho1, self.ho2)

        self.assertEqual(expected1, vec1)
        self.assertEqual(expected2, vec2)

        # Check with missing information
        with self.assertRaises(RuntimeError):
            hu._two_centroids_as_Vectors(
                OneHorizon(None, None, None, None), self.ho2
            )

        with self.assertRaises(RuntimeError):
            hu._two_centroids_as_Vectors(
                self.ho1, OneHorizon(None, None, None, None)
            )

    def test_compute_separation_vector(self):
        expected = Vector(
            [
                2 * TimeSeries(self.times, self.cen_x),
                2 * TimeSeries(self.times, self.cen_y),
                TimeSeries(self.times, np.zeros_like(self.times)),
            ],
        )

        separation_vec = hu.compute_separation_vector(self.ho1, self.ho2)

        self.assertEqual(expected, separation_vec)

    def test_compute_separation(self):
        separation = hu.compute_separation(self.ho1, self.ho2)

        np.testing.assert_allclose(separation.t, self.times)
        np.testing.assert_allclose(
            separation.y, 2 * np.sqrt(self.cen_x**2 + self.cen_y**2)
        )

    def test_compute_center_of_mass(self):
        # The mass of the first is 1, the mass of the second is 3, so the center
        # of mass should be in -1/2 (cen_x, cen_y, 0)

        com = hu.compute_center_of_mass(self.ho1, self.ho2)

        np.testing.assert_allclose(com[0].y, -0.5 * self.cen_x)
        np.testing.assert_allclose(com[1].y, -0.5 * self.cen_y)

    def test_compute_anmgular_velocity_vector(self):
        Omega = hu.compute_angular_velocity_vector(self.ho1, self.ho2)

        sep = Vector(
            [
                2 * TimeSeries(self.times, self.cen_x),
                2 * TimeSeries(self.times, self.cen_y),
                TimeSeries(self.times, np.zeros_like(self.times)),
            ],
        )

        dot_sep = sep.differentiated()

        # z and dot(z) are zero
        om_x = np.zeros_like(self.times)
        om_y = np.zeros_like(self.times)

        om_z = (
            sep[0] * dot_sep[1] - sep[1] * dot_sep[0]
        ).y / sep.norm().y ** 2

        np.testing.assert_allclose(Omega[0].y, om_x)
        np.testing.assert_allclose(Omega[1].y, om_y)
        np.testing.assert_allclose(Omega[2].y, om_z)
