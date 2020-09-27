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

import numpy as np

from postcactus import sensitivity_curves as sc
from postcactus import frequencyseries as fs


class TestSensitivityCurves(unittest.TestCase):

    def test_LISA(self):

        freqs = np.array([1e-5, 1e-3])

        lisa = sc.Sn_LISA(freqs)

        self.assertEqual(lisa,
                         fs.FrequencySeries(freqs,
                                            [1.97249256e-27, 1.63410062e-38]))
