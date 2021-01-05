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

# We use the agg backend because it should work everywhere
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from postcactus import visualize as viz
from postcactus import grid_data as gd
from postcactus import grid_data_utils as gdu
from postcactus import cactus_grid_functions as cgf


class TestVisualize(unittest.TestCase):
    def setUp(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2)

    def test_setup_matplotlib(self):
        viz.setup_matplotlib()

        self.assertEqual(matplotlib.rcParams["font.size"], 16)

    def test__preprocess_plot_functions(self):

        # We test preprocess_plot with a function that returns the argument, so
        # we can check that they are what we expect
        def func(data, **kwargs):
            return data, kwargs

        dec_func = viz._preprocess_plot(func)

        # Default
        self.assertIs(dec_func("")[1]["axis"], plt.gca())
        # Passing axis
        self.assertIs(dec_func("", axis=self.ax2)[1]["axis"], self.ax2)

        # Default
        self.assertIs(dec_func("")[1]["figure"], plt.gcf())
        # Passing figure
        self.assertIs(dec_func("", figure=self.fig)[1]["figure"], self.fig)

        dec_func_grid = viz._preprocess_plot_grid(func)

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
        self.assertTrue(
            np.allclose(
                ret2[1]["coordinates"][0], ugd.coordinates_from_grid()[0]
            )
        )

        # Not passing x0 and x1
        ret2b = dec_func_grid(data=hg, shape=ugd.shape)
        self.assertTrue(
            np.allclose(
                ret2b[1]["coordinates"][0], ugd.coordinates_from_grid()[0]
            )
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
        self.assertTrue(
            np.allclose(
                ret3[1]["coordinates"][0], ugd.coordinates_from_grid()[0]
            )
        )
        # Test xlabel and ylabel
        self.assertEqual(ret3[1]["xlabel"], "x")

        # Test with resample=True
        ret5 = dec_func_grid(
            data=cactus_ascii,
            iteration=0,
            shape=ugd.shape,
            xlabel="x",
            ylabel="y",
            resample=True,
        )

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

        ugd = gdu.sample_function(lambda x, y: x + y, [100, 20], [0, 1], [2, 5])

        # Unknown plot type
        with self.assertRaises(ValueError):
            viz._plot_grid(ugd, plot_type="bubu")

        self.assertTrue(
            isinstance(
                viz.plot_contourf(ugd, xlabel="x", ylabel="y",
                                  colorbar=True, label="test"),
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

    def test_plot_colorbar(self):

        ugd = gdu.sample_function(lambda x, y: x + y, [100, 20], [0, 1], [2, 5])

        cf = viz.plot_contourf(ugd, xlabel="x", ylabel="y", colorbar=False)

        self.assertTrue(
            isinstance(
                viz.plot_colorbar(cf, label="test"),
                matplotlib.colorbar.Colorbar,
            )
        )

    def test_add_text_to_figure_corner(self):

        self.assertTrue(
            isinstance(
                viz.add_text_to_figure_corner("test"),
                matplotlib.text.Text,
            )
        )

    def test_save(self):

        plt.plot([1, 1], [2, 2])
        # Test matplotlib
        viz.save("test", "pdf")
        self.assertTrue(os.path.exists("test.pdf"))
        os.remove("test.pdf")

        # Test tikzplotlib
        viz.save("test", "pdf", as_tikz=True)
        self.assertTrue(os.path.exists("test.tikz"))
        os.remove("test.tikz")
