#!/usr/bin/env python3

# Copyright (C) 2020-2023 Gabriele Bozzola
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
import tempfile
import unittest

# We use the agg backend because it should work everywhere
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kuibit import cactus_grid_functions as cgf  # noqa: E402
from kuibit import grid_data as gd  # noqa: E402
from kuibit import grid_data_utils as gdu  # noqa: E402
from kuibit import masks as km  # noqa: E402
from kuibit import visualize_matplotlib as viz  # noqa: E402
from kuibit.simdir import SimDir  # noqa: E402


class TestVisualizeMatplotlib(unittest.TestCase):
    def setUp(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2)

    def test_setup_matplotlib(self):
        viz.setup_matplotlib()

        self.assertEqual(matplotlib.rcParams["font.size"], 16)

        # Test with optional argument
        viz.setup_matplotlib({"font.size": 18})
        self.assertEqual(matplotlib.rcParams["font.size"], 18)

        # Test with rc file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matplotlib.rc")
            with open(path, "w") as tmpfile:
                tmpfile.write("image.cmap: viridis")

            viz.setup_matplotlib(rc_par_file=path)
        self.assertEqual(matplotlib.rcParams["image.cmap"], "viridis")

    def test_preprocess_plot_functions(self):
        # We test preprocess_plot with a function that returns the argument, so
        # we can check that they are what we expect
        def func(data, **kwargs):
            return data, kwargs

        dec_func = viz.preprocess_plot(func)

        # Default
        self.assertIs(dec_func("")[1]["axis"], plt.gca())
        # Passing axis
        self.assertIs(dec_func("", axis=self.ax2)[1]["axis"], self.ax2)

        # Default
        self.assertIs(dec_func("")[1]["figure"], plt.gcf())
        # Passing figure
        self.assertIs(dec_func("", figure=self.fig)[1]["figure"], self.fig)

        dec_func_grid = viz.preprocess_plot_grid(func)

        # Check data not provided
        with self.assertRaises(TypeError):
            dec_func_grid()

        # Data numpy array, but not of dimension 2
        with self.assertRaises(ValueError):
            dec_func_grid(data=np.linspace(0, 1, 2))

        # Data numpy array of dimension 2
        self.assertTrue(
            np.array_equal(
                dec_func_grid(data=np.array([[1, 2], [3, 4]]))[0],
                np.array([[1, 2], [3, 4]]),
            )
        )

        # Check with UniformGridData

        # 2D
        ugd = gdu.sample_function(lambda x, y: x + y, [10, 20], [0, 1], [2, 5])
        # Passing coordinates
        with self.assertWarns(Warning):
            ret = dec_func_grid(
                data=ugd, coordinates=ugd.coordinates_from_grid()
            )
            self.assertTrue(np.array_equal(ret[0], ugd.data_xyz))
            self.assertIs(ret[1]["coordinates"], ugd.coordinates_from_grid())

        # Passing x0 and x1 but not shape
        with self.assertRaises(TypeError):
            dec_func_grid(data=ugd, x0=ugd.x0, x1=ugd.x1)

        # Now HierachicalGridData or resampling UniformGridData

        hg = gd.HierarchicalGridData([ugd])

        # Shape not provided
        with self.assertRaises(TypeError):
            dec_func_grid(data=hg)

        # Valid (passing x0 and x1)
        ret2 = dec_func_grid(data=hg, shape=ugd.shape, x0=ugd.x0, x1=ugd.x1)
        # Check coordinates (which checks the resampling)
        np.testing.assert_allclose(
            ret2[1]["coordinates"][0], ugd.coordinates_from_grid()[0]
        )

        # Not passing x0 and x1
        ret2b = dec_func_grid(data=hg, shape=ugd.shape)
        np.testing.assert_allclose(
            ret2b[1]["coordinates"][0], ugd.coordinates_from_grid()[0]
        )

        # We create an empty OneGridFunctionASCII and populate it with ugd
        cactus_ascii = cgf.OneGridFunctionASCII([], "var_name", (0, 0))

        cactus_ascii.allfiles = ["file"]
        cactus_ascii.alldata = {"file": {0: {0: {0: ugd}}}}

        # Iteration not provided
        with self.assertRaises(TypeError):
            dec_func_grid(data=cactus_ascii)

        # Check coordinates (which checks the resampling)
        ret3 = dec_func_grid(
            data=cactus_ascii, iteration=0, shape=ugd.shape, xlabel="x"
        )
        np.testing.assert_allclose(
            ret3[1]["coordinates"][0], ugd.coordinates_from_grid()[0]
        )

        # Test xlabel and ylabel
        self.assertEqual(ret3[1]["xlabel"], "x")

        # Test with resample=True
        _ = dec_func_grid(
            data=cactus_ascii,
            iteration=0,
            shape=ugd.shape,
            xlabel="x",
            ylabel="y",
            resample=True,
        )

        # Test with masked data

        # HierarchicalGrid
        hg_m = km.arcsin(gd.HierarchicalGridData([ugd]))
        with self.assertWarns(Warning):
            dec_func_grid(data=hg_m, shape=[10, 10])

        ugd_m = km.arcsin(ugd)
        with self.assertWarns(Warning):
            dec_func_grid(data=ugd_m, shape=[10, 10], x0=[1, 3])

    def test_vmin_vmax_extend(self):
        data = np.array([1, 2])

        # Test vmin, vmax None
        self.assertCountEqual(viz._vmin_vmax_extend(data), (1, 2, "neither"))

        # Test vmin None, vmax 1.5
        self.assertCountEqual(
            viz._vmin_vmax_extend(data, vmax=1.5), (1, 1.5, "max")
        )

        # Test vmin 0.5 (< 1), vmax 1.5
        self.assertCountEqual(
            viz._vmin_vmax_extend(data, vmin=0.5, vmax=1.5), (0.5, 1.5, "max")
        )

        # Test vmin 1.2 (> 1), vmax 1.5
        self.assertCountEqual(
            viz._vmin_vmax_extend(data, vmin=1.2, vmax=1.5), (1.2, 1.5, "both")
        )

        # Test vmin 0.5, vmax None
        self.assertCountEqual(
            viz._vmin_vmax_extend(data, vmin=0.5), (0.5, 2, "neither")
        )

        # Test vmin 1.2, vmax None
        self.assertCountEqual(
            viz._vmin_vmax_extend(data, vmin=1.2), (1.2, 2, "min")
        )

    def test_plot_grid(self):
        ugd = gdu.sample_function(
            lambda x, y: x + y, [100, 20], [0, 1], [2, 5]
        )

        # Unknown plot type
        with self.assertRaises(ValueError):
            viz._plot_grid(ugd, plot_type="bubu")

        self.assertTrue(
            isinstance(
                viz.plot_contourf(
                    ugd, xlabel="x", ylabel="y", colorbar=True, label="test"
                ),
                matplotlib.contour.QuadContourSet,
            )
        )

        # Here we are not providing the coordinates
        with self.assertRaises(ValueError):
            viz.plot_contourf(ugd.data_xyz, logscale=True)

        # Plot_color
        self.assertTrue(
            isinstance(
                viz.plot_color(
                    ugd, xlabel="x", ylabel="y", colorbar=True, label="test"
                ),
                matplotlib.image.AxesImage,
            )
        )

        # Plot color, grid = None
        self.assertTrue(
            isinstance(
                viz.plot_color(
                    ugd.data_xyz,
                ),
                matplotlib.image.AxesImage,
            )
        )

        # Plot contour, no levels
        with self.assertRaises(ValueError):
            viz._plot_grid(ugd, plot_type="contour")

        self.assertTrue(
            isinstance(
                viz.plot_contour(
                    ugd, xlabel="x", ylabel="y", colorbar=True, label="test"
                ),
                matplotlib.contour.QuadContourSet,
            )
        )

    def test_plot_colorbar(self):
        ugd = gdu.sample_function(
            lambda x, y: x + y, [100, 20], [0, 1], [2, 5]
        )

        cf = viz.plot_contourf(ugd, xlabel="x", ylabel="y", colorbar=False)

        self.assertTrue(
            isinstance(
                viz.plot_colorbar(cf, label="test"),
                matplotlib.colorbar.Colorbar,
            )
        )

    def test_plot_horizon_shape(self):
        ah = SimDir("tests/horizons").horizons[0, 1]

        shape = ah.shape_outline_at_iteration(0, (None, None, 0))

        self.assertTrue(
            isinstance(
                viz.plot_horizon(shape)[0],
                matplotlib.patches.Polygon,
            )
        )

        with self.assertRaises(ValueError):
            viz.plot_horizon_on_plane_at_iteration(ah, 0, "bob")

        with self.assertRaises(ValueError):
            viz.plot_horizon_on_plane_at_time(ah, 0, "bob")

        np.testing.assert_allclose(
            viz.plot_horizon(shape)[0].xy,
            viz.plot_horizon_on_plane_at_iteration(ah, 0, "xy")[0].xy,
        )

        np.testing.assert_allclose(
            viz.plot_horizon(shape)[0].xy,
            viz.plot_horizon_on_plane_at_time(ah, 0, "xy")[0].xy,
        )

        with self.assertRaises(RuntimeError):
            # To trigger this error, we "move" the horizon up so that it does
            # not intersect the equatorial plane. This is going to be surgical
            # operation.

            # This is some seriously ugly code!

            # Moving all the coordinates by 10
            # ah._patches[0] is at iteration 0
            # ah._patches[0][0] is a dictionary with the various components
            new_ah_patches00 = {
                k: 10 + v for k, v in ah._patches[0][0].items()
            }
            new_ah_patches0 = (new_ah_patches00, ah._patches[0][1])
            ah._patches[0] = new_ah_patches0

            viz.plot_horizon_on_plane_at_time(ah, 0, "xy")

    def test_plot_components_boundaries(self):
        # Test invalid data
        with self.assertRaises(TypeError):
            viz.plot_components_boundaries("bob")

        # Not 2D data
        ugd1D = gdu.sample_function(lambda x: x, [10], [0], [2])
        hg1D = gd.HierarchicalGridData([ugd1D])

        # Test invalid data
        with self.assertRaises(ValueError):
            viz.plot_components_boundaries(hg1D)

        # Now with valid data
        ugd = gdu.sample_function(lambda x, y: x + y, [10, 20], [0, 2], [2, 5])
        hg = gd.HierarchicalGridData([ugd])

        viz.plot_components_boundaries(hg)

    def test_add_text_to_corner(self):
        with self.assertRaises(ValueError):
            viz._process_anchor_info("BW", 0.02)

        expected_out = 0.98, 0.02, "bottom", "right"

        self.assertCountEqual(
            viz._process_anchor_info("SE", 0.02), expected_out
        )

        expected_out = 0.5, 0.98, "top", None

        self.assertCountEqual(
            viz._process_anchor_info("N", 0.02), expected_out
        )

        expected_out = 0.02, 0.5, None, "left"

        self.assertCountEqual(
            viz._process_anchor_info("W", 0.02), expected_out
        )

        self.assertTrue(
            isinstance(
                viz.add_text_to_corner("test"),
                matplotlib.text.Text,
            )
        )

        # Test with a 3D plot
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402

        # Clear everything before creating new axis, needed for matplotlib > 3.8
        plt.clf()
        ax = plt.subplot(projection="3d")
        self.assertTrue(
            isinstance(
                viz.add_text_to_corner("test", axis=ax),
                matplotlib.text.Text,
            )
        )

    def test_save(self):
        x = np.linspace(0, 1, 100)
        y = np.sin(x)

        plt.plot(x, y)

        # Test matplotlib
        viz.save("test.pdf")
        self.assertTrue(os.path.exists("test.pdf"))
        os.remove("test.pdf")

        # Test tikzplotlib
        viz.save("test.tikz", include_disclaimer=False)
        self.assertTrue(os.path.exists("test.tikz"))

        # We save the file size so that we can compare it to the "cleaned"
        # version
        file_size_og = os.stat("test.tikz").st_size

        os.remove("test.tikz")

        # Test with clean figure
        viz.save(
            "test.tikz",
            tikz_clean_figure=True,
            target_resolution=200,
            include_disclaimer=False,
        )
        self.assertTrue(os.path.exists("test.tikz"))
        # The cleaned file should be smaller
        file_size_cleaned = os.stat("test.tikz").st_size
        self.assertLess(file_size_cleaned, file_size_og)
        os.remove("test.tikz")

    def test_save_from_dir_name_ext(self):
        plt.plot([1, 1], [2, 2])
        # Test without dot in the ext
        viz.save_from_dir_filename_ext(".", "test", "pdf")
        self.assertTrue(os.path.exists("test.pdf"))
        os.remove("test.pdf")

        # Test with dot in the ext
        viz.save_from_dir_filename_ext(".", "test", ".pdf")
        self.assertTrue(os.path.exists("test.pdf"))
        os.remove("test.pdf")

    def test_figname(self):
        # Instead of testing with an argparse.Namespace, we create a tiny class
        # here with the attribute figname

        class TestFigname:
            def __init__(self, figname):
                self.figname = figname

        name = "bob"
        # Test with None
        self.assertEqual(viz.get_figname(TestFigname(None), name), name)

        # Test with name
        self.assertEqual(viz.get_figname(TestFigname(name), "lol"), name)
