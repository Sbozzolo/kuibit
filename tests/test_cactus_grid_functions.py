#!/usr/bin/env python3

# Copyright (C) 2020-2025 Gabriele Bozzola
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
from tempfile import TemporaryDirectory
from unittest import mock

import h5py
import numpy as np

from kuibit import cactus_grid_functions as cg
from kuibit import grid_data
from kuibit import simdir as sd


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

    def test_contains_OpenPMDVars(self):
        vars3D = self.gd.xyz
        self.assertIn("wavetoyx_u", vars3D)

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

        filesize_MB = filesize_B / (1024**2)
        self.assertEqual(filesize_MB, self.gd.total_filesize())

        # Test invalid unit
        with self.assertRaises(ValueError):
            self.gd.total_filesize("bob")


class TestAllGridFunctions(unittest.TestCase):
    def setUp(self):
        self.gf = sd.SimDir("tests/grid_functions").gf.xy
        self.gf_xyz = sd.SimDir("tests/grid_functions").gf.xyz

    def test_carpetx_tsv_variables_by_dimension(self):
        sim = sd.SimDir("tests/carpetx/boxinbox")
        coordinates = ["ccoordx", "ccoordy", "ccoordz"]
        radii = ["radius[0]", "radius[1]", "radius[2]"]
        # CarpetX writes 1D lines and full 3D data, and arrays as 1D lines.
        # There is no 2D TSV output, so no TSV file is read as 2D data.
        self.assertCountEqual(sim.gf.x.keys(), coordinates + radii)
        self.assertCountEqual(sim.gf.y.keys(), coordinates)
        self.assertCountEqual(sim.gf.z.keys(), coordinates)
        self.assertCountEqual(sim.gf.xy.keys(), [])
        self.assertCountEqual(sim.gf.xyz.keys(), coordinates)

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
            list(self.gf._vars_ascii_files.keys()),
            ["rho_b", "P", "vx", "vy", "vz", "rho_star"],
        )

        # Here we also have the grid arrays in
        # diskdiagnostics-integrated_quantities.xy.h5
        self.assertCountEqual(
            list(self.gf._vars_h5_files.keys()),
            [
                "rho_b",
                "P",
                "vx",
                "vy",
                "vz",
                "rho",
                "int_em_T_rph",
                "int_hydro_T_rph",
                "int_surface_density",
                "int_torque_dens",
            ],
        )

        self.assertCountEqual(list(self.gf._vars_openpmd_files.keys()), [])

        # Here we are not testing that files are correctly organized...

    def test_keys(self):
        self.assertCountEqual(
            self.gf.keys(),
            [
                "rho_b",
                "P",
                "vx",
                "vy",
                "vz",
                "rho",
                "rho_star",
                "int_em_T_rph",
                "int_hydro_T_rph",
                "int_surface_density",
                "int_torque_dens",
            ],
        )

        # Test .fields, which depends on keys()
        self.assertCountEqual(
            self.gf.fields.keys(),
            [
                "rho_b",
                "P",
                "vx",
                "vy",
                "vz",
                "rho",
                "rho_star",
                "int_em_T_rph",
                "int_hydro_T_rph",
                "int_surface_density",
                "int_torque_dens",
            ],
        )

    def test__contains(self):
        self.assertIn("P", self.gf)
        self.assertNotIn("Hey", self.gf)

    def test__getitem(self):
        # Test variable not present
        with self.assertRaises(KeyError):
            self.gf["hey"]

        # Test a variable from ASCII, one from HDF5, and one from OpenPMD

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

        # OpenPMD
        self.assertTrue(
            isinstance(self.gf_xyz["wavetoyx_u"], cg.OneGridFunctionOpenPMD)
        )
        self.assertEqual(self.gf_xyz["wavetoyx_u"].var_name, "wavetoyx_u")
        # self.gf_xyz['wavetoy-u']allfiles is a set with two elements, the
        # two iterations
        filenames = {
            os.path.split(f)[-1] for f in self.gf_xyz["wavetoyx_u"].allfiles
        }
        self.assertCountEqual(
            filenames, {"batman.it00000000.bp4", "batman.it00000001.bp4"}
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
        self.assertEqual(len(self.gf.allfiles), 5)

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
        self.reader = sd.SimDir("tests/grid_functions").gf.xy
        self.reader.num_ghost = (3, 3)

        # ASCII
        self.rho_star = self.reader["rho_star"]
        # There's only one file
        self.rho_star_file = self.rho_star.allfiles[0]

        # OpenPMD
        self.wavetoy = sd.SimDir("tests/grid_functions").gf.xyz["wavetoyx_rho"]
        self.wavetoy_file_it0 = sorted(self.wavetoy.allfiles)[0]

        # We are going to test all the methods of the baseclass with
        # HDF5 files

        # HDF5
        self.P = self.reader["P"]
        # There's only one file
        self.P_file = self.P.allfiles[0]

    # This is to make coverage happy and test the abstract methods
    # There's no real test here
    @mock.patch.multiple(cg.BaseOneGridFunction, __abstractmethods__=set())
    def test_baseclass(self):
        abs_base = cg.BaseOneGridFunction("", "")
        with self.assertRaises(NotImplementedError):
            abs_base._parse_file("")
        with self.assertRaises(NotImplementedError):
            abs_base._read_component_as_uniform_grid_data("", 0, 0, 0)
        with self.assertRaises(NotImplementedError):
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

        self.assertCountEqual(
            self.wavetoy._iterations_in_file(self.wavetoy_file_it0), [0]
        )
        self.assertEqual(
            self.wavetoy._min_iteration_in_file(self.wavetoy_file_it0), 0
        )
        self.assertEqual(
            self.wavetoy._max_iteration_in_file(self.wavetoy_file_it0), 0
        )
        self.assertCountEqual(
            self.wavetoy._ref_levels_in_file(self.wavetoy_file_it0, 0),
            [0, 1],
        )
        self.assertCountEqual(
            self.wavetoy._components_in_file(
                self.wavetoy_file_it0,
                0,
                0,
            ),
            [0, 1, 2, 3],
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

        self.assertEqual(self.wavetoy.min_iteration, 0)
        self.assertEqual(self.wavetoy.max_iteration, 1)
        self.assertCountEqual(self.wavetoy.available_iterations, [0, 1])
        self.assertCountEqual(self.wavetoy.iterations, [0, 1])

        # Iteration not available
        with self.assertRaises(ValueError):
            self.P._files_with_iteration(3)
        with self.assertRaises(ValueError):
            self.P._files_with_iteration(-1)

        self.assertEqual(self.P._files_with_iteration(1), [self.P_file])

    def test_times(self):
        self.assertCountEqual(self.P.available_times, [0, 0.25, 0.5])
        self.assertCountEqual(self.P.times, [0, 0.25, 0.5])

        self.assertCountEqual(self.wavetoy.available_times, [0, 0.0078125])
        self.assertCountEqual(self.wavetoy.times, [0, 0.0078125])

    def test_iteration_at_time(self):
        self.assertEqual(self.P.iteration_at_time(0.5), 2)

        self.assertEqual(self.wavetoy.iteration_at_time(0.0078125), 1)

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

        # Test that we can read a grid array
        self.assertTrue(
            isinstance(
                self.reader["int_em_T_rph"][0], grid_data.HierarchicalGridData
            )
        )

    def test_hdf5_out3D_ghosts(self):
        # 1D data also respects out3D_ghosts = no
        rho_1d = cg.OneGridFunctionH5(["tests/tov_alt/rho.x.h5"], "rho", (0,))
        self.assertFalse(rho_1d.are_ghostzones_in_files)
        (merged_1d,) = rho_1d[0][0]
        np.testing.assert_array_equal(merged_1d.shape, [30])

        # This data has CarpetIOHDF5::out3D_ghosts = no, which applies to 2D
        # outputs too, so there are no ghost points between the components
        rho = cg.OneGridFunctionH5(
            ["tests/tov/output-0000/static_tov/rho.xz.h5"], "rho", (0, 2)
        )
        self.assertFalse(rho.are_ghostzones_in_files)
        (merged,) = rho[0][0]
        np.testing.assert_array_equal(merged.shape, [30, 30])

        # The refinement buffer points are kept
        rho = cg.OneGridFunctionH5(
            ["tests/tov/output-0000/static_tov/rho.xy.h5"], "rho", (0, 1)
        )
        (merged,) = rho[0][1]
        np.testing.assert_array_equal(merged.shape, [40, 40])

        # 3D data also respects out3D_ghosts = no
        rho_3d = cg.OneGridFunctionH5(
            [
                "tests/tov/output-0000/static_tov/rho.xyz.file_0.h5",
                "tests/tov/output-0000/static_tov/rho.xyz.file_1.h5",
            ],
            "rho",
            (0, 1, 2),
        )
        self.assertFalse(rho_3d.are_ghostzones_in_files)
        (merged_3d,) = rho_3d[0][0]
        np.testing.assert_array_equal(merged_3d.shape, [30, 30, 30])

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

        # Test read on grid with resample = True
        self.assertEqual(
            self.P.read_on_grid(0, new_grid, resample=True),
            expected_hgd.to_UniformGridData_from_grid(new_grid, resample=True),
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
        expected_data = np.loadtxt(self.rho_star_file, usecols=(12,))[
            : 29 * 29
        ]

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

    def test_llama_1d_ascii(self):
        llama_files = ["tests/tov/files-with-llama/phi1.0.x.asc"]
        gf = cg.AllGridFunctions(llama_files, dimension=(0,))
        self.assertIn("phi1", gf)

    def test_llama_1d_h5(self):
        llama_files = ["tests/tov/files-with-llama/phi1.0.x.h5"]
        gf = cg.AllGridFunctions(llama_files, dimension=(0,))
        self.assertIn("phi1", gf)

    def test_openpmd(self):
        iteration = 0
        component = 0

        with cg.openpmd_series(self.wavetoy_file_it0) as series:
            mesh = series.iterations[iteration].meshes["wavetoyx_state_lev00"]
            self.assertEqual(tuple(mesh.axis_labels), ("z", "y", "x"))
            variable = mesh["wavetoyx_u"]
            chunk = variable.available_chunks()[component]
            self.assertEqual(tuple(chunk.extent), (9, 33, 33))

            # Do the actual reading
            data_zyx = variable.load_chunk(chunk.offset, chunk.extent)
            series.flush()

        # The fixture stores (Nz, Ny, Nx) = (9, 33, 33). kuibit uses
        # (Nx, Ny, Nz), so the expected grid shape is (33, 33, 9).
        expected_grid_xyz = grid_data.UniformGrid(
            [33, 33, 9],
            x0=[-1.0, -1.0, -1.0],
            x1=[0.0, 0.0, 1.0],
            ref_level=0,
            num_ghost=[0, 0, 0],
            time=0,
            iteration=0,
            component=0,
        )

        expected_grid_data_xyz = grid_data.UniformGridData(
            expected_grid_xyz, data_zyx.transpose(2, 1, 0)
        )

        self.assertEqual(
            self.wavetoy._read_component_as_uniform_grid_data(
                self.wavetoy_file_it0, 0, 0, 0
            ),
            expected_grid_data_xyz,
        )

        # Check that the four component origins are interpreted correctly.
        # Each origin is in kuibit's (x, y, z) coordinate order, and entry n
        # is the expected origin of component n.
        component_origins_xyz = [
            tuple(
                self.wavetoy._read_component_as_uniform_grid_data(
                    self.wavetoy_file_it0, 0, 0, component
                ).x0
            )
            for component in range(4)
        ]
        expected_component_origins_xyz = [
            (-1.0, -1.0, -1.0),  # component 0: lower-left x-y tile
            (0.0, -1.0, -1.0),  # component 1: lower-right x-y tile
            (-1.0, 0.0, -1.0),  # component 2: upper-left x-y tile
            (0.0, 0.0, -1.0),  # component 3: upper-right x-y tile
        ]
        self.assertEqual(component_origins_xyz, expected_component_origins_xyz)

    def test_openpmd_centering(self):
        iteration = 0
        reader_xyz = sd.SimDir("tests/grid_functions").gf.xyz
        base_grids = {
            0: grid_data.UniformGrid(
                [17, 17, 17],
                x0=[-1.0, -1.0, -1.0],
                x1=[1.0, 1.0, 1.0],
                ref_level=0,
                num_ghost=[0, 0, 0],
                time=0,
                iteration=iteration,
                component=-1,
            ),
            1: grid_data.UniformGrid(
                [17, 17, 17],
                x0=[-0.5, -0.5, -0.5],
                x1=[0.5, 0.5, 0.5],
                ref_level=1,
                num_ghost=[0, 0, 0],
                time=0,
                iteration=iteration,
                component=-1,
            ),
        }
        shifts = {
            "staggeredwavetoyx_u": np.array([0.0, 0.0, 0.0]),
            "staggeredwavetoyx_fx": np.array([0.5, 0.0, 0.0]),
            "staggeredwavetoyx_fy": np.array([0.0, 0.5, 0.0]),
            "staggeredwavetoyx_fz": np.array([0.0, 0.0, 0.5]),
            "staggeredwavetoyx_curlfx": np.array([0.0, 0.5, 0.5]),
            "staggeredwavetoyx_curlfy": np.array([0.5, 0.0, 0.5]),
            "staggeredwavetoyx_curlfz": np.array([0.5, 0.5, 0.0]),
            "staggeredwavetoyx_eps": np.array([0.5, 0.5, 0.5]),
        }

        def shifted_grid(base_grid, shift):
            shape = np.array(base_grid.shape) - (shift > 0).astype(int)
            x0 = base_grid.x0 + shift * base_grid.dx
            x1 = base_grid.x1 - shift * base_grid.dx
            return grid_data.UniformGrid(
                shape,
                x0=x0,
                x1=x1,
                ref_level=base_grid.ref_level,
                num_ghost=base_grid.num_ghost,
                time=base_grid.time,
                iteration=base_grid.iteration,
                component=base_grid.component,
            )

        for var_name, shift in shifts.items():
            actual_hgd = reader_xyz[var_name][iteration]
            self.assertEqual(actual_hgd.refinement_levels, [0, 1])
            self.assertEqual(actual_hgd.shape, {0: 1, 1: 1})

            for ref_level in (0, 1):
                expected_grid = shifted_grid(base_grids[ref_level], shift)
                self.assertEqual(actual_hgd[ref_level][0].grid, expected_grid)

    def test_clear_cache(self):
        # Read something
        self.P[0]

        # Check that we saved something
        self.assertIsNotNone(self.P.alldata[self.P_file][0][0][0])

        # Remove that something
        self.P.clear_cache()

        # Check that we are clear
        self.assertIsNone(self.P.alldata[self.P_file][0][0][0])


class TestOneGridFunctionTSV(unittest.TestCase):
    # tests/carpetx/boxinbox is a CarpetX run on two processes with two
    # refinement levels, where level 1 has two separate boxes. output-0000 has
    # comment headers and one file per iteration (iterations 0-4). output-0001
    # restarts from iteration 3, with plain headers and iterations 3-6 combined
    # in one file. The values of ccoordx/y/z are the coordinates themselves.
    def setUp(self):
        self.sim = sd.SimDir("tests/carpetx/boxinbox")

    def assert_coordinates(self, grid_function, axis):
        coordinates = grid_function.coordinates_from_grid(as_same_shape=True)
        np.testing.assert_allclose(grid_function.data, coordinates[axis])

    def test_carpetx_tsv_line(self):
        reader = self.sim.gf.x
        with self.assertWarns(Warning) as warn:
            ccoordx = reader["ccoordx"]
        self.assertIn("TSV", str(warn.warning))

        self.assertIsInstance(ccoordx, cg.OneGridFunctionTSV)
        self.assertEqual(ccoordx.available_iterations, list(range(7)))
        np.testing.assert_allclose(
            ccoordx.available_times, 0.03125 * np.arange(7)
        )

        # Without component numbers, the gap between the two level-1 boxes
        # separates two components
        for path in ccoordx.allfiles:
            for iteration in ccoordx._iterations_in_file(path):
                self.assertEqual(
                    ccoordx._components_in_file(path, iteration, 1), [0, 1]
                )

        for iteration in (0, 6):
            data = ccoordx[iteration]
            self.assertEqual(data.shape, {0: 1, 1: 2})
            for level, component in ((0, 0), (1, 0), (1, 1)):
                self.assert_coordinates(data[level][component], 0)
            # The ghost points are included
            np.testing.assert_allclose(data[0][0].x0, [-1.0625])
            np.testing.assert_allclose(data[0][0].dx, [0.125])
            np.testing.assert_allclose(data[1][0].x0, [-0.78125])
            np.testing.assert_allclose(data[1][1].x0, [0.21875])

        # The line along y passes between the refined boxes
        with self.assertWarns(Warning):
            ccoordy = self.sim.gf.y["ccoordy"]
        self.assertEqual(ccoordy[6].shape, {0: 1})
        self.assert_coordinates(ccoordy[6][0][0], 0)

    def test_carpetx_tsv_line_num_ghost(self):
        reader = self.sim.gf.x
        reader.num_ghost = (1,)
        data = reader["ccoordx"][0]

        # The ghost zones are removed on level 0, which is one grid. The two
        # level-1 boxes cannot be merged, so they keep their ghost zones.
        self.assertEqual(data.shape, {0: 1, 1: 2})
        np.testing.assert_allclose(data[0][0].x0, [-0.9375])
        np.testing.assert_array_equal(data[0][0].shape, [16])
        np.testing.assert_array_equal(data[1][0].num_ghost, [1])
        self.assert_coordinates(data[1][1], 0)

    def test_carpetx_tsv_array(self):
        reader = self.sim.gf.x
        # Grid arrays have no ghost zones, so num_ghost does not apply to them
        reader.num_ghost = (1,)
        radius = reader["radius[0]"]

        self.assertEqual(radius.available_iterations, list(range(7)))
        path = (
            "tests/carpetx/boxinbox/output-0000/carpetx_tsv/"
            "boxinbox-radii.it000000.x.tsv"
        )
        expected = np.loadtxt(path, usecols=3)
        array = radius[0]
        self.assertEqual(array.shape, {-1: 1})
        component = array[-1][-1]
        self.assertEqual(component.component, -1)
        np.testing.assert_allclose(component.x0, [0])
        np.testing.assert_allclose(component.dx, [1])
        np.testing.assert_array_equal(component.num_ghost, [0])
        np.testing.assert_array_equal(component.data, expected)

    def test_carpetx_tsv_complex_array(self):
        # Output of the CarpetX TestOutput test, a 3D complex array
        path = "tests/carpetx/testoutput/testoutput-a3_complex.it000000.%s.tsv"
        for axis, dimension in (("x", (0,)), ("z", (2,))):
            with self.subTest(axis=axis):
                arrays = cg.AllGridFunctions(
                    [path % axis], dimension=dimension
                )
                self.assertCountEqual(arrays.keys(), ["a3_complex"])
                real, imag = np.loadtxt(path % axis, usecols=(5, 6)).T
                np.testing.assert_array_equal(
                    arrays["a3_complex"][0][-1][-1].data, real + 1j * imag
                )

    def test_carpetx_tsv_3d(self):
        reader = self.sim.gf.xyz
        reader.num_ghost = (1, 1, 1)

        for axis, name in enumerate(("ccoordx", "ccoordy", "ccoordz")):
            with self.subTest(name=name):
                grid_function = reader[name]
                self.assertEqual(grid_function.available_iterations, [0])
                self.assertCountEqual(
                    grid_function._files_with_iteration(0),
                    grid_function.allfiles,
                )
                # One file per process, each with one component per level
                for path in grid_function.allfiles:
                    for level in (0, 1):
                        self.assertEqual(
                            len(
                                grid_function._components_in_file(
                                    path, 0, level
                                )
                            ),
                            1,
                        )

                data = grid_function[0]
                self.assertEqual(data.shape, {0: 1, 1: 2})
                for _, _, component in data:
                    self.assert_coordinates(component, axis)


class TestOneGridFunctionTSVSynthetic(unittest.TestCase):
    # Small files for the cases that the CarpetX fixtures do not cover
    columns_1d = "iteration time patch level i j k x y z rho"
    columns_3d = "iteration time patch level component i j k x y z rho"

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def _write_tsv(self, rows, is_3d=False, filename=None):
        if filename is None:
            filename = (
                "test-grid.it000000.p0000.tsv" if is_3d else "test-grid.x.tsv"
            )
        path = os.path.join(self.directory, filename)
        with open(path, "w") as tsv_file:
            tsv_file.write(self.columns_3d if is_3d else self.columns_1d)
            tsv_file.write("\n")
            for row in rows:
                tsv_file.write(" ".join(map(str, row)) + "\n")
        return path

    def _rows_3d(self, iteration=0, level=0, component=0, x0=0):
        # Distinct axis lengths and spacings expose transposes
        return [
            [iteration, iteration * 0.5, 0, level, component, i, j, k]
            + [x0 + i * 0.5, j * 0.25, k * 2, i + 10 * j + 100 * k]
            for i, j, k in np.ndindex(2, 3, 4)
        ]

    def test_discovery_skips_unreadable_and_empty_files(self):
        # Empty 1D lines are normal output, so only the file without column
        # labels (out_tsv_header = none) gives a warning
        self._write_tsv([], filename="test-empty.x.tsv")
        path = os.path.join(self.directory, "test-headerless.x.tsv")
        with open(path, "w") as stream:
            stream.write("0 0 1\n")
        self._write_tsv([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        files = sorted(os.listdir(self.directory))
        files = [os.path.join(self.directory, name) for name in files]
        with self.assertWarnsRegex(RuntimeWarning, "labels.*test-headerless"):
            variables = cg.AllGridFunctions(files, (0,))
        self.assertCountEqual(variables.keys(), ["rho"])

    def test_1d_components_are_separated_by_gaps(self):
        rows = []
        for iteration, level, indices in (
            (0, 0, [0, 1]),
            (0, 0, [4, 5]),
            (0, 1, [0, 1]),
            (1, 0, [0, 1]),
            (0, 0, [8, 9]),
        ):
            for index in indices:
                rows.append(
                    [iteration, iteration * 0.5, 0, level, index, 0, 0]
                    + [index, 0, 0, index + 1]
                )
        path = self._write_tsv(rows)
        reader = cg.OneGridFunctionTSV([path], "rho", num_ghost=(0,))

        # Inspect the components before HierarchicalGridData merges them
        components = reader.alldata[path]
        np.testing.assert_array_equal(components[0][0][0].data, [1, 2])
        np.testing.assert_array_equal(components[0][0][1].data, [5, 6])
        np.testing.assert_array_equal(components[0][0][2].data, [9, 10])
        self.assertCountEqual(components[0][1], [0])
        self.assertCountEqual(components[1][0], [0])
        self.assertEqual(reader.time_at_iteration(1), 0.5)

    def test_3d_rows_in_any_order(self):
        rows = []
        for iteration, level, component in np.ndindex(2, 2, 2):
            for row in self._rows_3d(iteration, level, component, component):
                row[-1] += 1000 * iteration + 100 * level + 10 * component
                rows.append(row)
        np.random.default_rng(42).shuffle(rows)
        path = self._write_tsv(rows, is_3d=True)
        reader = cg.OneGridFunctionTSV([path], "rho", num_ghost=(0, 0, 0))

        i, j, k = np.indices((2, 3, 4))
        for iteration, level, component in np.ndindex(2, 2, 2):
            grid = reader.alldata[path][iteration][level][component]
            np.testing.assert_array_equal(
                grid.data,
                i
                + 10 * j
                + 100 * k
                + 1000 * iteration
                + 100 * level
                + 10 * component,
            )
            np.testing.assert_array_equal(grid.x0, [component, 0, 0])
            np.testing.assert_array_equal(grid.dx, [0.5, 0.25, 2])
            self.assertEqual(grid.time, iteration * 0.5)

    def test_invalid_data(self):
        rows = self._rows_3d()
        path = self._write_tsv(rows[:-1], is_3d=True)
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            cg.OneGridFunctionTSV([path], "rho")

        rows[0][2] = 1  # patch
        path = self._write_tsv(rows, is_3d=True)
        with self.assertRaisesRegex(NotImplementedError, "patch 0"):
            cg.OneGridFunctionTSV([path], "rho")


class TestOneGridFunctionOpenPMD(unittest.TestCase):
    def setUp(self):
        self.gf = sd.SimDir("tests/grid_functions").gf.xyz

    def test__init(self):
        self.assertCountEqual(self.gf.dimension, (0, 1, 2))
        # Here we check that we indexed the correct variables. We must check
        # the OpenPMD file

        # There are seven files including three OpenPMD files in the test
        # folder:
        # 1. illinoisgrmhd-grmhd_primitives_allbutbi.xyz.asc (ASCII one group)
        # 2. rho_star.xyz.asc (ASCII one var)
        # 3. illinoisgrmhd-grmhd_primitives_allbutbi.xyz.file_0.h5 (HDF5 one group)
        # 4. illinoisgrmhd-grmhd_primitives_allbutbi.xyz.file_1.h5 (HDF5 one group)
        # 5. batman.it00000000.bp4 (OpenPMD bp4)
        # 6. batman.it00000001.bp4 (OpenPMD bp4)
        # 7. staggeredwavetoyx_openpmd_16.it00000000.bp5 (OpenPMD bp5)

        # Here we can we find all the variables

        # assert variables from OpenPMD files
        self.assertCountEqual(
            list(self.gf._vars_openpmd_files.keys()),
            [
                "staggeredwavetoyx_curlfx",
                "staggeredwavetoyx_curlfy",
                "staggeredwavetoyx_curlfz",
                "staggeredwavetoyx_eps",
                "staggeredwavetoyx_ft",
                "staggeredwavetoyx_fx",
                "staggeredwavetoyx_fy",
                "staggeredwavetoyx_fz",
                "staggeredwavetoyx_u",
                "wavetoyx_rho",
                "wavetoyx_u",
            ],
        )

        # assert variables from h5 file
        self.assertCountEqual(
            list(self.gf._vars_h5_files.keys()),
            ["P", "rho_b", "vx", "vy", "vz"],
        )

        # assert variables from ascii file
        self.assertCountEqual(
            list(self.gf._vars_ascii_files.keys()),
            ["P", "rho_b", "vx", "vy", "vz", "rho_star"],
        )

    def test_allfiles(self):
        # This is a weak test, we are just testing how many files we have...
        # There should be 7 files including the OpenPMD bp4/bp5 files
        self.assertEqual(len(self.gf.allfiles), 7)

    def test_iterations(self):
        wavetoyx_rho = self.gf.fields.wavetoyx_rho
        self.assertEqual(wavetoyx_rho.iterations, [0, 1])

    def test_multiple_mesh_refinement_levels(self):
        # wavetouyx_rho variable in the OpenPMD file has 2 mesh refinement levels
        wavetoyx_rho0 = self.gf.fields.wavetoyx_rho[0]
        self.assertEqual(wavetoyx_rho0.refinement_levels, [0, 1])
