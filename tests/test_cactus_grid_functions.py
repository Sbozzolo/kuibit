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

import os
import unittest
from unittest import mock

import h5py
import numpy as np

from postcactus import cactus_grid_functions as cg
from postcactus import grid_data
from postcactus import simdir as sd


class TestGridFunctionsDir(unittest.TestCase):
    def setUp(self):
        self.files_dir = "tests/grid_functions"
        self.sim = sd.SimDir(self.files_dir)
        self.gd = cg.GridFunctionsDir(self.sim)

    def test_init_griddir(self):

        # Not a SimDir
        with self.assertRaises(TypeError):
            cg.GridFunctionsDir(0)

    def test_GridFunctionsDir_string_or_tuple(self):

        # Test not recognized dimension
        with self.assertRaises(ValueError):
            self.gd._string_or_tuple_to_dimension_index("hey")

        self.assertEqual(
            self.gd._string_or_tuple_to_dimension_index("x"), (0,)
        )

        self.assertEqual(
            self.gd._string_or_tuple_to_dimension_index("xyz"), (0, 1, 2)
        )

        self.assertEqual(
            self.gd._string_or_tuple_to_dimension_index((0, 1, 2)), (0, 1, 2)
        )

    def test_contains(self):

        self.assertIn("xyz", self.gd)

    def test__getitem(self):

        self.assertIs(self.gd["xy"], self.gd._all_griddata[(0, 1)])

        # Use tuple as key
        self.assertIs(self.gd[(0, 1)], self.gd["xy"])

    def test__getattr(self):

        # Error for invalid attribute
        with self.assertRaises(AttributeError):
            self.gd.hey

        # Valid attribute
        self.assertIs(self.gd.xy, self.gd["xy"])

    def test__str(self):

        self.assertIn(str(self.gd.xyz), str(self.gd))

    def test_total_filesize(self):

        filesize_B = sum(os.path.getsize(fil) for fil in self.sim.allfiles)
        self.assertEqual(filesize_B, self.gd.total_filesize("B"))

        filesize_MB = filesize_B / (1024 ** 2)
        self.assertEqual(filesize_MB, self.gd.total_filesize())

        # Test invalid unit
        with self.assertRaises(ValueError):
            self.gd.total_filesize("bob")


class TestAllGridFunctions(unittest.TestCase):
    def setUp(self):
        self.gf = sd.SimDir("tests/grid_functions").gf.xy

    def test__init(self):
        self.assertCountEqual(self.gf.dimension, (0, 1))

        # Test wrong number of ghosts
        with self.assertRaises(ValueError):
            cg.AllGridFunctions(
                self.gf.allfiles, dimension=(0, 1), num_ghost=(3,)
            )

        # Here we check that we indexed the correct variables. We must check
        # HDF5 files and ASCII files with both one variable per file and one
        # group per file.

        # There are four file in the test folder:
        # 1. illinoisgrmhd-grmhd_primitives_allbutbi.xy.asc (ASCII one group)
        # 2. rho_star.xy.asc (ASCII one var)
        # 3. rho.xy.h5 (HDF5 one var)
        # 4. illinoisgrmhd-grmhd_primitives_allbutbi.xy.h5 (HDF5 one group)
        #
        # In ASCII files we also check for compressed files

        # Here we can we find all the variables
        self.assertCountEqual(
            list(self.gf._vars_ascii.keys()),
            ["rho_b", "P", "vx", "vy", "vz", "rho_star"],
        )
        self.assertCountEqual(
            list(self.gf._vars_h5.keys()),
            ["rho_b", "P", "vx", "vy", "vz", "rho"],
        )

        # Here we are not testing that files are correctly organized...

    def test_keys(self):

        self.assertCountEqual(
            self.gf.keys(), ["rho_b", "P", "vx", "vy", "vz", "rho", "rho_star"]
        )

        # Test .fields, which depends on keys()
        self.assertCountEqual(
            self.gf.fields.keys(),
            ["rho_b", "P", "vx", "vy", "vz", "rho", "rho_star"],
        )

    def test__contains(self):

        self.assertIn("P", self.gf)
        self.assertNotIn("Hey", self.gf)

    def test__getitem(self):

        # Test variable not present
        with self.assertRaises(KeyError):
            self.gf["hey"]

        # Test a variable from ASCII and one from HDF5

        # We don't test the details, we just test that object is initialized
        # with the correct data

        # HDF5
        self.assertTrue(isinstance(self.gf["P"], cg.OneGridFunctionH5))
        self.assertEqual(self.gf["P"].var_name, "P")
        # self.gf['P']allfiles is a set with one element
        filename = os.path.split(next(iter(self.gf["P"].allfiles)))[-1]
        self.assertEqual(
            filename, "illinoisgrmhd-grmhd_primitives_allbutbi.xy.h5"
        )

        # ASCII

        # First, we test that a warning is emitted when we don't have ghost
        # zone information
        with self.assertWarns(Warning):
            self.gf["rho_star"]

        # Next we set the ghost zones
        self.gf.num_ghost = (3, 3)

        self.assertTrue(
            isinstance(self.gf["rho_star"], cg.OneGridFunctionASCII)
        )
        self.assertEqual(self.gf["rho_star"].var_name, "rho_star")

        # self.gf['rho_star']allfiles is a set with one element
        filename = os.path.split(next(iter(self.gf["rho_star"].allfiles)))[-1]
        self.assertEqual(filename, "rho_star.xy.asc")

        # Test get
        # default value
        self.assertIs(self.gf.get("hey"), None)
        # Test filename
        filename = os.path.split(next(iter(self.gf.get("P").allfiles)))[-1]
        self.assertEqual(
            filename, "illinoisgrmhd-grmhd_primitives_allbutbi.xy.h5"
        )

    def test_allfiles(self):

        # This is a weak test, we are just testing how many files we have...

        # There should be 4 files
        self.assertEqual(len(self.gf.allfiles), 4)

    def test_total_filesize(self):

        size_B = sum({os.path.getsize(path) for path in self.gf.allfiles})
        self.assertEqual(size_B, self.gf.total_filesize("B"))

        size_KB = size_B / 1024
        self.assertEqual(size_KB, self.gf.total_filesize("KB"))

    def test__str(self):

        self.assertIn("vz", str(self.gf))
        self.assertIn("Available grid data of dimension 2D (xy)", str(self.gf))


class TestOneGridFunction(unittest.TestCase):
    def setUp(self):
        # First we set the correct number of ghost zones
        reader = sd.SimDir("tests/grid_functions").gf.xy
        reader.num_ghost = (3, 3)

        # ASCII
        self.rho_star = reader["rho_star"]
        # There's only one file
        self.rho_star_file = self.rho_star.allfiles[0]

        # We are going to test all the methods of the baseclass with
        # HDF5 files

        # HDF5
        self.P = reader["P"]
        # There's only one file
        self.P_file = self.P.allfiles[0]

    # This is to make coverage happy and test the abstract methods
    # There's no real test here
    @mock.patch.multiple(cg.BaseOneGridFunction, __abstractmethods__=set())
    def test_baseclass(self):

        abs_base = cg.BaseOneGridFunction("", "")
        abs_base._parse_file("")
        abs_base._read_component_as_uniform_grid_data("", 0, 0, 0)
        abs_base.time_at_iteration("")

    def test__properties_in_file(self):

        self.assertCountEqual(
            self.P._iterations_in_file(self.P_file), [0, 1, 2]
        )
        self.assertEqual(self.P._min_iteration_in_file(self.P_file), 0)
        self.assertEqual(self.P._max_iteration_in_file(self.P_file), 2)
        self.assertCountEqual(
            self.P._ref_levels_in_file(self.P_file, 2), [0, 1]
        )
        self.assertCountEqual(
            self.P._components_in_file(self.P_file, 2, 0), [0, 1]
        )

    def test_restarts(self):

        # TODO: This test is not robust when we are dealing with only one file...
        #       Add a second file and rewrite tests.

        self.assertCountEqual(self.P.restarts, [(0, 2, [self.P_file])])

    def test_iterations(self):

        self.assertEqual(self.P.min_iteration, 0)
        self.assertEqual(self.P.max_iteration, 2)
        self.assertCountEqual(self.P.available_iterations, [0, 1, 2])
        self.assertCountEqual(self.P.iterations, [0, 1, 2])

        # Iteration not available
        with self.assertRaises(ValueError):
            self.P._files_with_iteration(3)
        with self.assertRaises(ValueError):
            self.P._files_with_iteration(-1)

        self.assertEqual(self.P._files_with_iteration(1), [self.P_file])

    def test_times(self):

        self.assertCountEqual(self.P.available_times, [0, 0.25, 0.5])
        self.assertCountEqual(self.P.times, [0, 0.25, 0.5])

    def test_iteration_at_time(self):

        self.assertEqual(self.P.iteration_at_time(0.5), 2)

        # Time not available
        with self.assertRaises(ValueError):
            self.P.iteration_at_time(5)

    def test_total_filesize(self):

        # We already tested that KB and MB work
        self.assertEqual(
            os.path.getsize(self.P_file), self.P.total_filesize("B")
        )

    # Here we test the details of the HDF5 reader
    def test_init_hdf5(self):

        self.assertEqual(self.P.thorn_name, "ILLINOISGRMHD")
        self.assertEqual(self.P.map, "")
        self.assertEqual(self.P.var_name, "P")

        # This also tests parse_file
        expected_alldata = {
            self.P_file: {
                0: {  # Iteration 0
                    0: {0: None, 1: None},  # ref_level 0 (components)
                    1: {0: None, 1: None},
                },  # ref_level 1 (components)
                1: {  # Iteration 1
                    1: {0: None, 1: None}
                },  # ref_level 1 (components)
                2: {  # Iteration 2
                    0: {0: None, 1: None},  # ref_level 0 (components)
                    1: {0: None, 1: None},  # ref_level 1 (components)
                },
            }
        }
        self.assertCountEqual(self.P.alldata, expected_alldata)

        # TODO: Here we are not testing the case in which
        #       IOHDF5::output_ghost_points is inconsistant across files.
        #       We are also not testing all the different combinations of
        #       parameters and ways to set "true" in Cactus

        # Here we are testing _are_ghostzones_in_file
        self.assertTrue(self.P.are_ghostzones_in_files)

    def test_read_hdf5(self):

        expected_grid = grid_data.UniformGrid(
            [29, 29],
            x0=[-14, -14],
            x1=[14, 14],
            ref_level=0,
            num_ghost=[3, 3],
            time=0,
            iteration=0,
            component=0,
        )

        with h5py.File(self.P_file, "r") as fil:
            # Notice the .T
            expected_data = fil["ILLINOISGRMHD::P it=0 tl=0 rl=0 c=0"][()].T

        expected_grid_data = grid_data.UniformGridData(
            expected_grid, expected_data
        )

        self.assertEqual(
            self.P._read_component_as_uniform_grid_data(self.P_file, 0, 0, 0),
            expected_grid_data,
        )

    def test_time_at_iteration(self):

        self.assertEqual(self.P.time_at_iteration(2), 0.5)

    def test_get(self):

        # Iteration not present
        with self.assertRaises(KeyError):
            self.P[9]

        # Let's read iteration 0, we have two ref levels and two components
        data00 = self.P._read_component_as_uniform_grid_data(
            self.P_file, 0, 0, 0
        )
        data01 = self.P._read_component_as_uniform_grid_data(
            self.P_file, 0, 0, 1
        )
        data10 = self.P._read_component_as_uniform_grid_data(
            self.P_file, 0, 1, 0
        )
        data11 = self.P._read_component_as_uniform_grid_data(
            self.P_file, 0, 1, 1
        )

        expected_hgd = grid_data.HierarchicalGridData(
            [data00, data01, data10, data11]
        )

        self.assertEqual(self.P[0], expected_hgd)

        # default
        self.assertIs(self.P.get_iteration(3), None)
        self.assertIs(self.P.get_time(3), None)

        self.assertEqual(self.P.get_iteration(0), self.P[0])
        self.assertEqual(self.P.get_time(0.5), self.P[2])

        # Test iteration
        self.assertEqual(next(iter(self.P)), self.P[0])

        new_grid = grid_data.UniformGrid([10, 10], x0=[1, 1], x1=[2, 2])

        self.assertEqual(
            self.P.read_on_grid(0, new_grid),
            expected_hgd.to_UniformGridData_from_grid(new_grid),
        )

        # Here we test the details of the ASCII reader

    def test_init(self):

        self.assertCountEqual(
            self.rho_star._iterations_to_times, {0: 0, 1: 0.25, 2: 0.5}
        )

        # TODO: This test doesn't test compressed files nor 3D files,
        #       and it tests only one component/refinement level/iteration

        # The first component ends at line 880 of rho_star.xy.asc
        # Let's check this first one
        #
        # We have to remove the blank lines from the count
        expected_data = np.loadtxt(
            self.rho_star_file, usecols=(12,), max_rows=875
        )

        expected_grid = grid_data.UniformGrid(
            [29, 29],
            x0=[-14, -14],
            x1=[14, 14],
            ref_level=0,
            num_ghost=[3, 3],
            time=0,
            iteration=0,
            component=0,
        )

        expected_grid_data = grid_data.UniformGridData(
            expected_grid, expected_data.reshape((29, 29)).T
        )

        self.assertEqual(
            self.rho_star._read_component_as_uniform_grid_data(
                self.rho_star_file, 0, 0, 0
            ),
            expected_grid_data,
        )

        # Test time_at_iteration
        self.assertEqual(self.rho_star.time_at_iteration(2), 0.5)
        # Iteration not available
        with self.assertRaises(ValueError):
            self.rho_star.time_at_iteration(20)

        # Test file with wrong name
        with self.assertRaises(RuntimeError):
            self.rho_star._parse_file("/tmp/wrongname")
