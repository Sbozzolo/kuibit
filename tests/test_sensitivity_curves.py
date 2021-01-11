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

import unittest

import numpy as np

from kuibit import frequencyseries as fs
from kuibit import sensitivity_curves as sc


class TestSensitivityCurves(unittest.TestCase):
    def test_LISA(self):

        freqs = np.array([1e-5, 1e-3])

        lisa = sc.Sn_LISA(freqs)

        self.assertEqual(
            lisa, fs.FrequencySeries(freqs, [1.97249256e-27, 1.63410062e-38])
        )

    def test_ETB(self):

        freqs = np.array([1, 1.0023060e03])

        etb = sc.Sn_ET_B(freqs)

        self.assertEqual(
            etb,
            fs.FrequencySeries(
                freqs, [4.8012536e-21 ** 2, 6.5667816e-25 ** 2]
            ),
        )

    def test_CE1(self):

        freqs = np.array([3.0002, 3577.8])

        ce1 = sc.Sn_CE1(freqs)

        self.assertEqual(
            ce1,
            fs.FrequencySeries(freqs, [1.6664e-19 ** 2, 2.6856e-24 ** 2]),
        )

    def test_CE2(self):

        freqs = np.array([3.0002, 3577.8])

        ce2 = sc.Sn_CE2(freqs)

        self.assertEqual(
            ce2,
            fs.FrequencySeries(freqs, [3.7939e-20 ** 2, 1.4944e-24 ** 2]),
        )

    def test_aLIGO(self):

        freqs = np.array([9, 4.3015530306322887e02])

        aLIGO = sc.Sn_aLIGO(freqs)

        self.assertEqual(
            aLIGO,
            fs.FrequencySeries(
                freqs,
                [1.7370722680197635e-21 ** 2, 3.8885269176411187e-24 ** 2],
            ),
        )

    def test_voyager(self):

        freqs = np.array([5.001, 356.92])

        voya = sc.Sn_voyager(freqs)

        self.assertEqual(
            voya,
            fs.FrequencySeries(freqs, [1.7021e-20 ** 2, 9.846e-25 ** 2]),
        )

    def test_KAGRA_D(self):

        freqs = np.array([2.2491, 479.733])

        kagra = sc.Sn_KAGRA_D(freqs)

        self.assertEqual(
            kagra,
            fs.FrequencySeries(freqs, [1.01841e-18 ** 2, 6.78164e-24 ** 2]),
        )

    def test_aLIGO_plus(self):

        freqs = np.array([5, 536.7])

        aligo = sc.Sn_aLIGO_plus(freqs)

        self.assertEqual(
            aligo,
            fs.FrequencySeries(freqs, [1.9995e-20 ** 2, 1.8182e-24 ** 2]),
        )
