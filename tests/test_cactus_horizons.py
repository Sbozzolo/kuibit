#!/usr/bin/env python3

# Copyright (C) 2020-2022 Gabriele Bozzola
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
import re
import unittest

import numpy as np

from kuibit import cactus_horizons as ch
from kuibit import simdir as sd


class TestHorizonsDir(unittest.TestCase):
    def setUp(self):
        # This folder contains all the files, but we will experiment also in
        # the case we only have AH files by looking at the subdir diagnostics
        self.files_dir = "tests/horizons"
        self.sim = sd.SimDir(self.files_dir)

        # All the information
        self.hor = self.sim.horizons

        # Only AH
        self.sim_ah = sd.SimDir(self.files_dir + "/diagnostics")
        self.ahs = ch.HorizonsDir(self.sim_ah)

        # QLM + SHAPE but not AH
        self.sim_shape = sd.SimDir(self.files_dir, max_depth=1)
        self.qlm_shape = ch.HorizonsDir(self.sim_shape)

    def test_init(self):
        # Testing the init tests also the _populate functions

        # Now let's test that self.hor has all the correct variables

        # qlm
        expected_qlm_vars = {}

        # Here we use a variation on _populate_qlm_vars
        for var_name in self.sim.ts.scalar.keys():
            if var_name.startswith("qlm"):
                var_name_stripped = var_name[3:]
                for char in var_name_stripped:
                    if char.isdigit():
                        horizon_number = int(char)
                        break

                horizon_vars = expected_qlm_vars.setdefault(horizon_number, {})
                horizon_vars[var_name_stripped] = self.sim.ts.scalar[var_name]

        self.assertCountEqual(self.hor._qlm_vars, expected_qlm_vars)
        self.assertCountEqual(self.ahs._qlm_vars, {})
        self.assertCountEqual(self.qlm_shape._qlm_vars, expected_qlm_vars)

        # ah
        # Fully testing this is quite complicated, so we will start by testing
        # that the variables are the expected ones
        expected_ah_vars = [
            "cctk_iteration",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "min_radius",
            "max_radius",
            "mean_radius",
            "quadrupole_xx",
            "quadrupole_xy",
            "quadrupole_xz",
            "quadrupole_yy",
            "quadrupole_yz",
            "quadrupole_zz",
            "min_x",
            "max_x",
            "min_y",
            "max_y",
            "min_z",
            "max_z",
            "xy_plane_circumference",
            "xz_plane_circumference",
            "yz_plane_circumference",
            "ratio_of_xz_xy_plane_circumferences",
            "ratio_of_yz_xy_plane_circumferences",
            "area",
            "m_irreducible",
            "areal_radius",
            "expansion_Theta_l",
            "inner_expansion_Theta_n",
            "product_of_the_expansions",
            "mean_curvature",
            "gradient_of_the_areal_radius",
            "gradient_of_the_expansion_Theta_l",
            "gradient_of_the_inner_expansion_Theta_n",
            "gradient_of_the_product_of_the_expansions",
            "gradient_of_the_mean_curvature",
            "minimum__of_the_mean_curvature",
            "maximum__of_the_mean_curvature",
            "integral_of_the_mean_curvature",
        ]

        self.assertCountEqual(
            list(self.hor._ah_vars[1].keys()), expected_ah_vars
        )
        self.assertCountEqual(
            list(self.ahs._ah_vars[1].keys()), expected_ah_vars
        )
        self.assertCountEqual(list(self.qlm_shape._ah_vars[1].keys()), [])

        # Shape
        expected_shape_files = [
            os.path.join(self.files_dir, f)
            for f in os.listdir(self.files_dir)
            if f.endswith(".gp")
        ]
        exp_shape1 = [f for f in expected_shape_files if f.endswith("ah1.gp")]
        exp_shape2 = [f for f in expected_shape_files if f.endswith("ah2.gp")]

        dict_shape = {1: exp_shape1, 2: exp_shape2}

        self.assertTrue(self.hor.found_any)
        self.assertTrue(self.ahs.found_any)
        self.assertTrue(self.qlm_shape.found_any)

        self.assertCountEqual(self.hor._shape_files, dict_shape)
        # Here we also test that the keys in _shape_files is correct.
        # It has to be the same as ah_vars
        self.assertCountEqual(self.ahs._shape_files, {1: {}, 2: {}})
        self.assertCountEqual(self.qlm_shape._shape_files, dict_shape)

    def test_properties(self):
        self.assertCountEqual(self.hor.available_qlm_horizons, [0, 1, 2])
        self.assertCountEqual(self.ahs.available_qlm_horizons, [])
        self.assertCountEqual(self.qlm_shape.available_qlm_horizons, [0, 1, 2])

        self.assertCountEqual(self.hor.available_apparent_horizons, [1, 2])
        self.assertCountEqual(self.ahs.available_apparent_horizons, [1, 2])
        self.assertCountEqual(
            self.qlm_shape.available_apparent_horizons, [1, 2]
        )

    def test__str(self):
        self.assertEqual(
            str(sd.SimDir("tests/tov").horizons), "No horizon found"
        )

        expected_str = "Horizons found:\n"
        expected_str += "3 horizons from QuasiLocalMeasures\n"
        expected_str += "2 horizons from AHFinderDirect"

        self.assertEqual(str(self.hor), expected_str)

    def test__getitem__(self):
        # Incorrect key
        with self.assertRaises(TypeError):
            self.hor["hey"]

        # Key with incorrect length
        with self.assertRaises(KeyError):
            self.hor[(0, 1, 2)]

        # Horizon not available
        with self.assertRaises(KeyError):
            self.hor[(5, 7)]

        # Here we test the OneHorizon is initialied with the correct
        # arguments, we test OneHorizon below.
        self.assertCountEqual(
            self.hor[(0, 1)]._qlm_vars, self.hor._qlm_vars[0]
        )
        self.assertCountEqual(self.hor[(0, 1)]._ah_vars, self.hor._ah_vars[1])
        self.assertCountEqual(
            self.hor[(0, 1)]._shape_files, self.hor._shape_files[1]
        )

        self.assertCountEqual(self.ahs[(0, 1)]._qlm_vars, {})
        self.assertCountEqual(self.ahs[(0, 1)]._ah_vars, self.ahs._ah_vars[1])
        self.assertCountEqual(
            self.ahs[(0, 1)]._shape_files, self.ahs._shape_files[1]
        )

        # Horizons in qlm_shape will emit a warning because there's no AH data
        with self.assertWarns(Warning):
            self.assertCountEqual(
                self.qlm_shape[(1, 2)]._qlm_vars, self.qlm_shape._qlm_vars[1]
            )
            self.assertCountEqual(self.qlm_shape[(1, 2)]._ah_vars, {})
            self.assertCountEqual(
                self.qlm_shape[(1, 2)]._shape_files,
                self.qlm_shape._shape_files[1],
            )

    def test_get_apparent_horizon(self):
        # Horizon not available
        with self.assertRaises(KeyError):
            self.hor.get_apparent_horizon(6)

        # Check that we have the variables that we expect
        self.assertCountEqual(
            self.hor.get_apparent_horizon(1)._ah_vars, self.hor._ah_vars[1]
        )
        self.assertCountEqual(
            self.hor.get_apparent_horizon(1)._shape_files,
            self.hor._shape_files[1],
        )

    def test_get_qlm_horizon(self):
        # Horizon not available
        with self.assertRaises(KeyError):
            self.hor.get_qlm_horizon(6)

        # Check that we have the variables that we expect
        self.assertCountEqual(
            self.hor.get_qlm_horizon(0)._qlm_vars, self.hor._qlm_vars[0]
        )


class TestOneHorizon(unittest.TestCase):
    def setUp(self):
        # This folder contains all the files, but we will experiment also in
        # the case we only have AH files by looking at the subdir diagnostics
        self.files_dir = "tests/horizons"
        self.sim = sd.SimDir(self.files_dir)

        # All the information
        self.hor = self.sim.horizons

        # Only AH
        self.sim_ah = sd.SimDir(self.files_dir + "/diagnostics")
        self.ahs = ch.HorizonsDir(self.sim_ah)

        # QLM + SHAPE but not AH
        self.sim_shape = sd.SimDir(self.files_dir, max_depth=1)
        self.qlm_shape = ch.HorizonsDir(self.sim_shape)

        # One horizons in each
        self.ho = self.hor[(0, 1)]
        self.ah = self.ahs[(0, 1)]

        # Here we capture the warning
        with self.assertWarns(Warning):
            self.sh = self.qlm_shape[(0, 1)]

    def test_init(self):
        # Check that attributes for qlm vars are set (or not)
        self.assertTrue(hasattr(self.ho, "mass"))
        self.assertFalse(hasattr(self.ah, "mass"))
        self.assertTrue(hasattr(self.sh, "mass"))

        # Check that _final properties are defined when qlm is available
        self.assertIsNot(self.ho.mass_final, None)
        self.assertIs(self.ah.mass_final, None)
        self.assertIsNot(self.sh.mass_final, None)

        # Check that attributes for ah vars are set (or not)
        self.assertTrue(hasattr(self.ho.ah, "area"))
        self.assertTrue(hasattr(self.ah.ah, "area"))
        self.assertFalse(hasattr(self.sh.ah, "area"))

        # Test qlm/ah_available
        self.assertTrue(self.ho.ah_available)
        self.assertTrue(self.ho.qlm_available)

        # Test formation time
        self.assertEqual(self.ho.formation_time, 0)
        self.assertEqual(self.ah.formation_time, 0)
        self.assertIs(self.sh.formation_time, None)

        # Test shape available
        self.assertTrue(self.ho.shape_available)
        self.assertFalse(self.ah.shape_available)
        self.assertTrue(self.sh.shape_available)

        # Test shape
        exp_iterations = [0, 128, 256, 384, 512, 640, 768, 896, 1024]
        exp_times = [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4]
        self.assertCountEqual(self.ho.shape_iterations, exp_iterations)
        self.assertCountEqual(self.ho.shape_times, exp_times)

        self.assertCountEqual(self.sh.shape_iterations, exp_iterations)
        self.assertIs(self.sh.shape_times, None)
        self.assertIs(self.sh.shape_time_min, None)
        self.assertIs(self.sh.shape_time_max, None)

    def test__getitem(self):
        # Test key not available
        with self.assertRaises(KeyError):
            self.ho["hey"]

        self.assertEqual(self.ho["mass"], self.ho._qlm_vars["mass"])

    def test_ah_property(self):
        self.assertEqual(
            self.ho.get_ah_property("area"), self.ho._ah_vars["area"]
        )

    def test__str(self):
        # Test various components
        self.assertIn("Formation time", str(self.ho))
        self.assertIn("Formation time", str(self.ah))
        self.assertNotIn("Formation time", str(self.sh))

        self.assertIn("Shape available", str(self.ho))
        self.assertNotIn("Shape available", str(self.ah))
        self.assertIn("Shape available", str(self.sh))

        self.assertIn("Final Mass", str(self.ho))
        self.assertNotIn("Final Mass", str(self.ah))
        self.assertIn("Final Mass", str(self.sh))

    def test_shape(self):
        # Test shape information not available
        with self.assertRaises(ValueError):
            self.ah._patches_at_iteration(0)

        # Test iteration not available
        with self.assertRaises(ValueError):
            self.ho._patches_at_iteration(10465)

        # Here we test all the infrastructure for loading the patches
        #
        self.assertCountEqual(
            self.ho.ah_origin_at_iteration(0), (5.35384615385, 0, 0)
        )

        patches, _ = self.ho._patches_at_iteration(0)

        # We test that we have the expected components with the expected
        # length
        patches_names = ["+z", "-z", "+x", "-x", "+y", "-y"]
        self.assertCountEqual(patches_names, list(patches.keys()))
        for patch in patches_names:
            with self.subTest(patch=patch):
                # patches[patch][0] is the x coordinates of the two angular
                # dimensions
                self.assertEqual(patches[patch][0].shape, (19, 19))

        expected_x = []
        for patch in patches:
            expected_x.append(patches[patch][0])

        np.testing.assert_allclose(
            expected_x, self.ho.shape_at_iteration(0)[0]
        )

    def test_shape_time_at_iteration(self):
        # Test iteration not available
        with self.assertRaises(ValueError):
            self.ho.shape_time_at_iteration(10465)

        self.assertAlmostEqual(self.ho.shape_time_at_iteration(1024), 6.400)

    def test_shape_at_time(self):
        # Test time not available
        with self.assertRaises(ValueError):
            self.ho.shape_at_time(10465)

        # Test time available
        np.testing.assert_allclose(
            self.ho.shape_at_time(6.4, 1e-2),
            self.ho.shape_at_iteration(1024),
        )

    def test_shape_outline_at_iteration(self):
        # Iteration not present
        with self.assertRaises(ValueError):
            self.ho.shape_outline_at_iteration(10465, [0, 2, 3])

        # Cut not a list
        with self.assertRaises(TypeError):
            self.ho.shape_outline_at_iteration(0, 3)

        # Cut not a list with three elements
        with self.assertRaises(ValueError):
            self.ho.shape_outline_at_iteration(0, [0, 1])

        # Three Nones
        np.testing.assert_allclose(
            self.ho.shape_outline_at_iteration(0, [None, None, None]),
            self.ho.shape_at_iteration(0),
        ),

        # Three non-Nones
        with self.assertRaises(ValueError):
            self.ho.shape_outline_at_iteration(0, [1, 2, 3])

        # 1D cut
        np.testing.assert_allclose(
            self.ho.shape_outline_at_iteration(0, [None, 0, 0]),
            [np.array([5.61956723]), np.array([5.08838347])],
        )

        # 2D cut with no points
        self.assertIs(
            self.ho.shape_outline_at_iteration(0, [None, None, 10]), None
        )

        # 2D cut with points
        # We check the length
        self.assertEqual(
            len(self.ho.shape_outline_at_iteration(0, [None, None, 0])[0]), 76
        )

    def test_shape_outline_at_time(self):
        # Test time not available
        with self.assertRaises(ValueError):
            self.ho.shape_outline_at_time(10465, [None, None, 0])

        # Test time available
        np.testing.assert_allclose(
            self.ho.shape_outline_at_time(6.4, [None, None, 0], 1e-2),
            self.ho.shape_outline_at_iteration(1024, [None, None, 0]),
        )

    def test_compute_horizons_separation(self):
        ho1 = self.hor[(0, 1)]
        ho2 = self.hor[(0, 2)]

        # Deprecated
        with self.assertWarns(FutureWarning):
            separation = ch.compute_horizons_separation(ho1, ho2)

        self.assertEqual(len(separation), 552)
        # They are the same file, so the separation should be 0 everywhere
        self.assertAlmostEqual(separation.max(), 0)

    def test_vtk_available(self):
        self.assertTrue(self.hor.get_qlm_horizon(1).vtk_available)
        self.assertFalse(self.hor.get_apparent_horizon(1).vtk_available)

        self.assertCountEqual(
            self.hor.get_qlm_horizon(1).vtk_available_iterations, [0, 208896]
        )

    def test_vtk_variables(self):
        ho = self.hor.get_qlm_horizon(1)

        # Iteration not existing
        with self.assertRaises(KeyError):
            ho._load_vtk_at_iteration(1)

        # We parse the vtk file here
        path = ho._vtk_files[0]

        with open(path) as file_:
            lines = file_.readlines()

        # Find all the variables in the VTK file by matching the header of the
        # section
        rx_scalars = re.compile(r"^SCALARS\s(\w+)\sfloat\s1\n$")

        # We add these manually
        variables = ["coordinates", "connectivity"]

        for line in lines:
            if match := rx_scalars.match(line):
                variables.append(match.groups()[0])

        self.assertCountEqual(
            ho.available_vtk_variables_at_iteration(0), variables
        )

        # Let's check the coordiantes. We hard-code the indeces for the given
        # VTK file
        coordinates = np.loadtxt(lines[5:2817])

        np.testing.assert_array_equal(
            ho.vtk_variable_at_iteration("coordinates", 0),
            coordinates,
        )

        # Now the connectivity. We hard-code the indeces for the given VTK file
        connectivity = np.loadtxt(lines[2819:5555])

        np.testing.assert_array_equal(
            ho.vtk_variable_at_iteration("connectivity", 0),
            connectivity,
        )
