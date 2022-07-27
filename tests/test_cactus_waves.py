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

import unittest

import numpy as np
from scipy import signal

from kuibit import cactus_waves as cw
from kuibit import gw_utils as gwu
from kuibit import simdir as sd
from kuibit import timeseries as ts


class TestCactusWaves(unittest.TestCase):
    def setUp(self):
        self.sim = sd.SimDir("tests/tov")
        self.gwdir = cw.GravitationalWavesDir(self.sim)
        self.psi4 = self.gwdir[110.69]
        self.emdir = cw.ElectromagneticWavesDir(self.sim)
        self.phi2 = self.emdir[110.69]

    def test_WavesOneDet(self):

        t1 = np.linspace(0, np.pi, 100)
        t2 = np.linspace(2 * np.pi, 3 * np.pi, 100)
        y1 = np.sin(t1)
        y2 = np.sin(t2)
        ts1 = ts.TimeSeries(t1, y1)
        ts2 = ts.TimeSeries(t2, y2)
        dist1 = 100

        data1 = [(2, 2, ts1), (2, 2, ts2)]
        data2 = [(1, 1, ts1), (1, 0, ts2), (1, -1, ts2)]

        gw = cw.GravitationalWavesOneDet(dist1, data1)
        em = cw.ElectromagneticWavesOneDet(dist1, data2)

        self.assertEqual(gw.l_min, 2)
        self.assertEqual(em.l_min, 1)

        # test copy
        gw_copy = gw.copy()
        em_copy = em.copy()
        self.assertEqual(gw, gw_copy)
        self.assertEqual(em, em_copy)

    def test_get_psi4_lm(self):

        self.assertEqual(
            self.gwdir[110.69].get_psi4_lm(2, 2), self.gwdir[110.69][(2, 2)]
        )

    def test__fixed_frequency_integrated(self):

        # First, we test the FF integration with a function (sin(x))
        # where the threshold frequency is smaller than the real frequency
        # (the period is extremely long).
        # This is a standard integration

        # Several points is better
        # However, to really reach agreement with the analytical result
        # we would need millions of points
        t = np.linspace(0, 2 * np.pi, 19000)
        y = np.sin(t)
        tts = ts.TimeSeries(t, y)
        # Dummy object (FFI is a staticmethod)
        gwdum = cw.GravitationalWavesOneDet(0, [(2, 2, tts)])

        # The integral of sin should be -cos
        # The period of sin(x) is 2 pi, so we pick pcut = 1e10
        integral = gwdum._fixed_frequency_integrated(tts, 1e10)

        np.testing.assert_allclose(integral.t, t)
        np.testing.assert_allclose(integral.y, -np.cos(t), atol=5e-4)

        # The second integral should be sin(x)
        integral2 = gwdum._fixed_frequency_integrated(tts, 1e10, order=2)

        np.testing.assert_allclose(integral2.y, -np.sin(t), atol=5e-4)

        # Now, let's see the opposite case in which the frequency is lower than
        # any frequencies. The output should be the same timeseries we started
        # with, rescaled by 1/(i omega_threshold). Now the factor of 1/i
        # corresponds to a rotation of pi/2, so we expect the output to be the
        # cosine. pcut = 1e-4 -> omega_threshold = 2 pi / pcut = 2 pi * 1e4
        # Hence, the timeseries is divided by 1e-4
        integral3 = gwdum._fixed_frequency_integrated(tts, 1e-4)
        np.testing.assert_allclose(
            integral3.y * 2 * np.pi * 1e4, -np.cos(t), atol=1e-3
        )

        # Check warning for irregularly spaced
        with self.assertWarns(RuntimeWarning):
            tts.t[1] *= 1.01
            gwdum._fixed_frequency_integrated(tts, 1e-4)

    def test_get_strain_lm(self):

        with self.assertRaises(ValueError):
            self.psi4.get_strain_lm(3, 3, 1)

        # Large pcut
        with self.assertRaises(ValueError):
            self.psi4.get_strain_lm(2, 2, 1000)

        # Test window function errors
        with self.assertRaises(ValueError):
            self.psi4.get_strain_lm(2, 2, 0.1, window_function=1)
        # Not implemented
        with self.assertRaises(ValueError):
            self.psi4.get_strain_lm(2, 2, 0.1, window_function="bubu")

        # We do not need to test the FFI, hopefully that is already tested

        # Test window = set verything to 0
        np.testing.assert_allclose(
            self.psi4.get_strain_lm(2, 2, 0.1, window_function=lambda x: 0).y,
            0,
        )

        psi4lm = self.psi4[(2, 2)]
        psi4lm *= self.psi4.dist

        # Test when window is a function
        ham_array = signal.hamming(len(psi4lm))
        ham_psi4lm = psi4lm.copy()
        ham_psi4lm.y *= ham_array
        ffi_ham = self.psi4._fixed_frequency_integrated(
            ham_psi4lm, 0.1, order=2
        ).y

        np.testing.assert_allclose(
            self.psi4.get_strain_lm(
                2, 2, 0.1, window_function=signal.hamming, trim_ends=False
            ).y,
            ffi_ham,
        )

        # Test window is a string, like hamming
        np.testing.assert_allclose(
            self.psi4.get_strain_lm(
                2, 2, 0.1, window_function="hamming", trim_ends=False
            ).y,
            ffi_ham,
        )

        # Test no window
        np.testing.assert_allclose(
            self.psi4.get_strain_lm(2, 2, 0.1, trim_ends=False).y,
            self.psi4._fixed_frequency_integrated(psi4lm, 0.1, order=2).y,
        )

        # Test trim ends
        np.testing.assert_allclose(
            self.psi4.get_strain_lm(2, 2, 0.1, trim_ends=True).y,
            self.psi4._fixed_frequency_integrated(psi4lm, 0.1, order=2)
            .cropped(
                init=self.psi4[(2, 2)].tmin + 0.1,
                end=self.psi4[(2, 2)].tmax - 0.1,
            )
            .y,
        )

    def test_get_strain(self):

        # test l_max too big
        with self.assertRaises(ValueError):
            self.psi4.get_strain(0, 0, 1, l_max=100)

        theta, phi = np.pi / 2, 1
        ym2 = gwu.sYlm(-2, 2, -2, theta, phi)
        ym1 = gwu.sYlm(-2, 2, -1, theta, phi)
        y0 = gwu.sYlm(-2, 2, 0, theta, phi)
        y1 = gwu.sYlm(-2, 2, 1, theta, phi)
        y2 = gwu.sYlm(-2, 2, 2, theta, phi)

        strain = (
            self.psi4.get_strain_lm(2, -2, 0.1).y * ym2
            + self.psi4.get_strain_lm(2, -1, 0.1).y * ym1
            + self.psi4.get_strain_lm(2, 0, 0.1).y * y0
            + self.psi4.get_strain_lm(2, 1, 0.1).y * y1
            + self.psi4.get_strain_lm(2, 2, 0.1).y * y2
        )

        np.testing.assert_allclose(
            strain, self.psi4.get_strain(theta, phi, 0.1).y
        )

    def test_get_observed_strain(self):

        # Let's check with Hanford
        theta_GW, phi_GW = np.pi / 3, 0
        antennas = gwu.antenna_responses_from_sky_localization(
            8, -70, "2015-09-14 09:50:45"
        )
        Fc_H, Fp_H = antennas.hanford

        expected_strain = self.psi4.get_strain(
            theta_GW, phi_GW, 0.1, trim_ends=False
        )
        expected_strain = (
            expected_strain.real() * Fp_H - expected_strain.imag() * Fc_H
        )

        strain = self.psi4.get_observed_strain(
            8,
            -70,
            "2015-09-14 09:50:45",
            theta_GW,
            phi_GW,
            0.1,
            trim_ends=False,
        )

        self.assertEqual(strain.hanford, expected_strain)

    def test_get_power_energy(self):

        psi4lm = self.psi4[(2, 2)]

        psi4lm_int = self.psi4._fixed_frequency_integrated(
            psi4lm, 0.1, order=1
        )

        power_lm = self.psi4.dist**2 / (16 * np.pi) * np.abs(psi4lm_int) ** 2

        self.assertEqual(power_lm, self.psi4.get_power_lm(2, 2, 0.1))
        self.assertEqual(
            power_lm.integrated(), self.psi4.get_energy_lm(2, 2, 0.1)
        )

        # Total power
        total_power = (
            self.psi4.get_power_lm(2, 2, 0.1)
            + self.psi4.get_power_lm(2, 1, 0.1)
            + self.psi4.get_power_lm(2, 0, 0.1)
            + self.psi4.get_power_lm(2, -1, 0.1)
            + self.psi4.get_power_lm(2, -2, 0.1)
        )

        self.assertEqual(total_power, self.psi4.get_total_power(0.1))
        self.assertEqual(
            total_power.integrated(), self.psi4.get_total_energy(0.1)
        )

    def test_get_power_energy_em(self):

        phi2 = self.phi2
        phi2lm = phi2[(2, 2)]

        power_lm = phi2.dist**2 / (4 * np.pi) * np.abs(phi2lm) ** 2

        self.assertEqual(power_lm, phi2.get_power_lm(2, 2))
        self.assertEqual(power_lm.integrated(), phi2.get_energy_lm(2, 2))

        # Total power
        total_power = (
            phi2.get_power_lm(2, 2)
            + phi2.get_power_lm(2, 1)
            + phi2.get_power_lm(2, 0)
            + phi2.get_power_lm(2, -1)
            + phi2.get_power_lm(2, -2)
        )

        self.assertEqual(total_power, phi2.get_total_power())
        self.assertEqual(total_power.integrated(), phi2.get_total_energy())

    def test_get_force_linear_momentum_em(self):

        # Test the z component with the 2,2 mode, so
        # dPz/dt should be r^2 / 4 pi * phi2_22 * conj(phi2_22) * 1 / 3

        # Test of the other components is below (we need more components)

        phi2lm = self.phi2[(2, 2)]

        mult_l, mult_m = 2, 2
        c_lm = mult_m / (mult_l * (mult_l + 1))

        phi2lm_2 = c_lm * np.conj(phi2lm)

        force_lm = (
            self.phi2.dist**2 / (4 * np.pi) * (phi2lm * phi2lm_2).real()
        )

        self.assertEqual(force_lm, self.phi2.get_force_z_lm(2, 2))

        with self.assertRaises(ValueError):
            self.phi2.get_force_lm(2, 2, direction=4)

        self.assertEqual(
            self.phi2.get_force_lm(2, 2, direction=2),
            self.phi2.get_force_z_lm(2, 2),
        )

        self.assertEqual(
            force_lm.integrated(),
            self.phi2.get_linear_momentum_z_lm(2, 2),
        )

        # get_linear_momentum_lm
        self.assertEqual(
            self.phi2.get_linear_momentum_lm(2, 2, direction=2),
            self.phi2.get_linear_momentum_z_lm(2, 2),
        )

        # Total linear momentum
        total_force_z = (
            self.phi2.get_force_z_lm(2, 2)
            + self.phi2.get_force_z_lm(2, 1)
            + self.phi2.get_force_z_lm(2, 0)
            + self.phi2.get_force_z_lm(2, -1)
            + self.phi2.get_force_z_lm(2, -2)
        )

        self.assertEqual(total_force_z, self.phi2.get_total_force_z())
        self.assertEqual(
            total_force_z.integrated(),
            self.phi2.get_total_linear_momentum_z(),
        )

        # total
        self.assertEqual(total_force_z, self.phi2.get_total_force(direction=2))
        self.assertEqual(
            total_force_z.integrated(),
            self.phi2.get_total_linear_momentum(direction=2),
        )

        # Now we want to test a case with l, m = 3, 2, and for which we have
        # l = 2 and l = 4.  We are going to fake these momenta in a copy
        # of self.phi2.

        # We multiply times some numbers so that they are not exactly the same

        dist = self.phi2.dist
        multipoles = [
            (2, 2, 2 * self.phi2[(2, 2)]),
            (3, 1, 3 * self.phi2[(2, 2)]),
            (3, 2, 4 * self.phi2[(2, 2)]),
            (4, 2, 5 * self.phi2[(2, 2)]),
        ]

        phi2_234 = cw.ElectromagneticWavesOneDet(dist, multipoles)

        el, em = 3, 1

        a_lm = np.sqrt((el - em) * (el + em + 1)) / (2 * el * (el + 1))
        b_lmm = (
            1
            / (2 * el)
            * np.sqrt(
                ((el - 1) * (el + 1) * (el - em) * (el - em - 1))
                / ((2 * el - 1) * (2 * el + 1))
            )
        )
        b_l1m1 = (
            1
            / (2 * el + 2)
            * np.sqrt(
                (el * (el + 2) * (el + em + 2) * (el + em + 1))
                / ((2 * el + 1) * (2 * el + 3))
            )
        )

        phi2lm_xy_2 = (
            a_lm * np.conj(phi2_234[3, 2])
            + b_lmm * np.conj(phi2_234[2, 2])
            - b_l1m1 * np.conj(phi2_234[4, 2])
        )

        Pp_lm = (
            phi2_234.dist**2 / (2 * np.pi) * (phi2_234[el, em] * phi2lm_xy_2)
        )

        self.assertEqual(Pp_lm.real(), phi2_234.get_force_x_lm(3, 1))
        self.assertEqual(Pp_lm.imag(), phi2_234.get_force_y_lm(3, 1))

        # There's only one component
        self.assertEqual(
            phi2_234.get_force_x_lm(3, 1),
            phi2_234.get_total_force(direction=0),
        )
        self.assertEqual(
            phi2_234.get_force_y_lm(3, 1),
            phi2_234.get_total_force(direction=1),
        )

        self.assertEqual(
            Pp_lm.real().integrated(),
            phi2_234.get_linear_momentum_x_lm(3, 1),
        )
        self.assertEqual(
            Pp_lm.imag().integrated(),
            phi2_234.get_linear_momentum_y_lm(3, 1),
        )

        self.assertEqual(
            Pp_lm.real().integrated(),
            phi2_234.get_total_linear_momentum_x(),
        )
        self.assertEqual(
            Pp_lm.imag().integrated(),
            phi2_234.get_total_linear_momentum_y(),
        )
        self.assertEqual(
            Pp_lm.imag().integrated(),
            phi2_234.get_total_linear_momentum(direction=1),
        )

        # z direction

        el, em = 3, 2

        c_lm = em / (el * (el + 1))
        d_lm = (
            1
            / el
            * np.sqrt(
                ((el - 1) * (el + 1) * (el - em) * (el + em))
                / ((2 * el - 1) * (2 * el + 1))
            )
        )
        d_l1m = (
            1
            / (el + 1)
            * np.sqrt(
                (el * (el + 2) * (el + 1 - em) * (el + 1 + em))
                / ((2 * el + 1) * (2 * el + 3))
            )
        )

        phi2lm_z_2 = (
            c_lm * np.conj(phi2_234[3, 2])
            + d_lm * np.conj(phi2_234[2, 2])
            + d_l1m * np.conj(phi2_234[4, 2])
        )

        force_z_lm = (
            phi2_234.dist**2
            / (4 * np.pi)
            * (phi2_234[el, em] * phi2lm_z_2).real()
        )

        self.assertEqual(force_z_lm, phi2_234.get_force_z_lm(3, 2))

    def test_get_torque_angular_momentum(self):

        psi4lm = self.psi4[(2, 2)]

        psi4lm_int1 = self.psi4._fixed_frequency_integrated(
            psi4lm, 0.1, order=1
        )
        psi4lm_int2 = self.psi4._fixed_frequency_integrated(
            psi4lm, 0.1, order=2
        )

        torque_lm = (
            self.psi4.dist**2
            / (16 * np.pi)
            * 2
            * (np.conj(psi4lm_int2) * psi4lm_int1).imag()
        )

        self.assertEqual(torque_lm, self.psi4.get_torque_z_lm(2, 2, 0.1))
        self.assertEqual(
            torque_lm.integrated(),
            self.psi4.get_angular_momentum_z_lm(2, 2, 0.1),
        )

        # Test get_torque_lm
        with self.assertRaises(ValueError):
            self.psi4.get_torque_lm(2, 2, 0.1, direction=4)

        self.assertEqual(
            self.psi4.get_torque_lm(2, 2, 0.1, direction=2),
            self.psi4.get_torque_z_lm(2, 2, 0.1),
        )

        with self.assertRaises(NotImplementedError):
            self.psi4.get_torque_x_lm(2, 2, 0.1)

        with self.assertRaises(NotImplementedError):
            self.psi4.get_torque_y_lm(2, 2, 0.1)

        # get_angular_momentum_lm
        self.assertEqual(
            self.psi4.get_angular_momentum_lm(2, 2, 0.1, direction=2),
            self.psi4.get_angular_momentum_z_lm(2, 2, 0.1),
        )

        with self.assertRaises(NotImplementedError):
            self.psi4.get_angular_momentum_x_lm(2, 2, 0.1)

        with self.assertRaises(NotImplementedError):
            self.psi4.get_angular_momentum_y_lm(2, 2, 0.1)

        # Total angular momentum
        total_torque_z = (
            self.psi4.get_torque_z_lm(2, 2, 0.1)
            + self.psi4.get_torque_z_lm(2, 1, 0.1)
            + self.psi4.get_torque_z_lm(2, 0, 0.1)
            + self.psi4.get_torque_z_lm(2, -1, 0.1)
            + self.psi4.get_torque_z_lm(2, -2, 0.1)
        )

        self.assertEqual(total_torque_z, self.psi4.get_total_torque_z(0.1))

        with self.assertRaises(NotImplementedError):
            self.psi4.get_total_torque_x(0.1)

        with self.assertRaises(NotImplementedError):
            self.psi4.get_total_torque_y(0.1)

        self.assertEqual(
            total_torque_z.integrated(),
            self.psi4.get_total_angular_momentum_z(0.1),
        )

        self.assertEqual(
            total_torque_z.integrated(),
            self.psi4.get_total_angular_momentum(0.1, direction=2),
        )

        with self.assertRaises(NotImplementedError):
            self.psi4.get_total_angular_momentum_x(0.1)

        with self.assertRaises(NotImplementedError):
            self.psi4.get_total_angular_momentum_y(0.1)

    def test_get_force_linear_momentum(self):

        # Test the z component with the 2,2 mode, so
        # dPz/dt should be r^2 / 16 pi * int psi4_22 * int conj(psi4_cc) * 2 / 3

        # Test of the other components is below (we need more components)

        psi4lm = self.psi4[(2, 2)]

        mult_l, mult_m = 2, 2

        c_lm = 2 * mult_m / (mult_l * (mult_l + 1))

        psi4lm_int1 = self.psi4._fixed_frequency_integrated(
            psi4lm, 0.1, order=1
        )
        psi4lm_int2 = self.psi4._fixed_frequency_integrated(
            c_lm * np.conj(psi4lm), 0.1, order=1
        )

        force_lm = (
            self.psi4.dist**2
            / (16 * np.pi)
            * (psi4lm_int1 * psi4lm_int2).real()
        )

        self.assertEqual(force_lm, self.psi4.get_force_z_lm(2, 2, 0.1))

        with self.assertRaises(ValueError):
            self.psi4.get_force_lm(2, 2, 0.1, direction=4)

        self.assertEqual(
            self.psi4.get_force_lm(2, 2, 0.1, direction=2),
            self.psi4.get_force_z_lm(2, 2, 0.1),
        )

        self.assertEqual(
            force_lm.integrated(),
            self.psi4.get_linear_momentum_z_lm(2, 2, 0.1),
        )

        # get_linear_momentum_lm
        self.assertEqual(
            self.psi4.get_linear_momentum_lm(2, 2, 0.1, direction=2),
            self.psi4.get_linear_momentum_z_lm(2, 2, 0.1),
        )

        # Total linear momentum
        total_force_z = (
            self.psi4.get_force_z_lm(2, 2, 0.1)
            + self.psi4.get_force_z_lm(2, 1, 0.1)
            + self.psi4.get_force_z_lm(2, 0, 0.1)
            + self.psi4.get_force_z_lm(2, -1, 0.1)
            + self.psi4.get_force_z_lm(2, -2, 0.1)
        )

        self.assertEqual(total_force_z, self.psi4.get_total_force_z(0.1))
        self.assertEqual(
            total_force_z.integrated(),
            self.psi4.get_total_linear_momentum_z(0.1),
        )

        # total
        self.assertEqual(
            total_force_z, self.psi4.get_total_force(0.1, direction=2)
        )
        self.assertEqual(
            total_force_z.integrated(),
            self.psi4.get_total_linear_momentum(0.1, direction=2),
        )

        # Now we want to test a case with l, m = 3, 2, and for which we have
        # l = 2 and l = 4.  We are going to fake these momenta in a copy
        # of self.psi4.

        # We multiply times some numbers so that they are not exactly the same

        dist = self.psi4.dist
        multipoles = [
            (2, 2, 2 * self.psi4[(2, 2)]),
            (3, 1, 3 * self.psi4[(2, 2)]),
            (3, 2, 4 * self.psi4[(2, 2)]),
            (4, 2, 5 * self.psi4[(2, 2)]),
        ]

        psi4_234 = cw.GravitationalWavesOneDet(dist, multipoles)

        el, em = 3, 1

        a_lm = np.sqrt((el - em) * (el + em + 1)) / (el * (el + 1))
        b_lmm = (
            1
            / (2 * el)
            * np.sqrt(
                ((el - 2) * (el + 2) * (el - em) * (el - em - 1))
                / ((2 * el - 1) * (2 * el + 1))
            )
        )
        b_l1m1 = (
            1
            / (2 * el + 2)
            * np.sqrt(
                ((el - 1) * (el + 3) * (el + em + 2) * (el + em + 1))
                / ((2 * el + 1) * (2 * el + 3))
            )
        )

        psi4lm_xy_int1 = psi4_234._fixed_frequency_integrated(
            psi4_234[3, 1], 0.1, order=1
        )
        psi4lm_xy_int2 = psi4_234._fixed_frequency_integrated(
            a_lm * np.conj(psi4_234[3, 2])
            + b_lmm * np.conj(psi4_234[2, 2])
            - b_l1m1 * np.conj(psi4_234[4, 2]),
            0.1,
            order=1,
        )

        Pp_lm = (
            psi4_234.dist**2
            / (8 * np.pi)
            * (psi4lm_xy_int1 * psi4lm_xy_int2)
        )

        self.assertEqual(Pp_lm.real(), psi4_234.get_force_x_lm(3, 1, 0.1))
        self.assertEqual(Pp_lm.imag(), psi4_234.get_force_y_lm(3, 1, 0.1))

        # There's only one component
        self.assertEqual(
            psi4_234.get_force_x_lm(3, 1, 0.1),
            psi4_234.get_total_force(0.1, direction=0),
        )
        self.assertEqual(
            psi4_234.get_force_y_lm(3, 1, 0.1),
            psi4_234.get_total_force(0.1, direction=1),
        )

        self.assertEqual(
            Pp_lm.real().integrated(),
            psi4_234.get_linear_momentum_x_lm(3, 1, 0.1),
        )
        self.assertEqual(
            Pp_lm.imag().integrated(),
            psi4_234.get_linear_momentum_y_lm(3, 1, 0.1),
        )

        self.assertEqual(
            Pp_lm.real().integrated(),
            psi4_234.get_total_linear_momentum_x(0.1),
        )
        self.assertEqual(
            Pp_lm.imag().integrated(),
            psi4_234.get_total_linear_momentum_y(0.1),
        )
        self.assertEqual(
            Pp_lm.imag().integrated(),
            psi4_234.get_total_linear_momentum(0.1, direction=1),
        )

        # z direction

        el, em = 3, 2

        c_lm = 2 * em / (el * (el + 1))
        d_lm = (
            1
            / el
            * np.sqrt(
                ((el - 2) * (el + 2) * (el - em) * (el + em))
                / ((2 * el - 1) * (2 * el + 1))
            )
        )
        d_l1m = (
            1
            / (el + 1)
            * np.sqrt(
                ((el - 1) * (el + 3) * (el + 1 - em) * (el + 1 + em))
                / ((2 * el + 1) * (2 * el + 3))
            )
        )

        psi4lm_z_int1 = psi4_234._fixed_frequency_integrated(
            psi4_234[3, 2], 0.1, order=1
        )
        psi4lm_z_int2 = psi4_234._fixed_frequency_integrated(
            c_lm * np.conj(psi4_234[3, 2])
            + d_lm * np.conj(psi4_234[2, 2])
            + d_l1m * np.conj(psi4_234[4, 2]),
            0.1,
            order=1,
        )

        force_z_lm = (
            psi4_234.dist**2
            / (16 * np.pi)
            * (psi4lm_z_int1 * psi4lm_z_int2).real()
        )

        self.assertEqual(force_z_lm, psi4_234.get_force_z_lm(3, 2, 0.1))

    def test_WavesDir(self):

        # Test the error on wrong input type
        with self.assertRaises(TypeError):
            cw.GravitationalWavesDir(0)

        with self.assertRaises(TypeError):
            cw.ElectromagneticWavesDir(0)

        # Check type
        self.assertTrue(
            isinstance(self.gwdir[110.69], cw.GravitationalWavesOneDet)
        )
        self.assertTrue(
            isinstance(self.emdir[110.69], cw.ElectromagneticWavesOneDet)
        )

    def test_extrapolation(self):

        # Test too many radii for the extrapolation
        with self.assertRaises(RuntimeError):
            self.gwdir._extrapolate_waves_to_infinity(
                [1, 2], [1, 2], [1, 2], 1, order=3
            )

        # Number of radii is not the same as waves
        with self.assertRaises(RuntimeError):
            self.gwdir._extrapolate_waves_to_infinity(
                [1, 2], [1, 2], [1, 2, 3], 1, order=1
            )

        # TODO: Write real tests for these functions. Compare with POWER?

        # NOTE: These tests are not physical tests!

        # Smoke test, we just check that the code returns expected data types
        times = np.linspace(0, 2 * np.pi, 100)
        sins = ts.TimeSeries(times, np.sin(times))
        sins_plus_one = ts.TimeSeries(times, np.sin(times + 1))
        self.gwdir._extrapolate_waves_to_infinity(
            [sins, sins_plus_one],
            np.linspace(-12, -11, 100),
            [10, 11],
            1,
            order=1,
        )

        self.gwdir.extrapolate_strain_lm_to_infinity(
            2, 2, 0.1, [110.69, 110.69], [2791, 2791.1], order=0
        )
        self.gwdir.extrapolate_strain_lm_to_infinity(
            2,
            2,
            0.1,
            [110.69, 110.69],
            [2791, 2791.1],
            order=0,
            extrapolate_amplitude_phase=True,
        )

    def test_empty(self):

        # Test with no wave information
        empty_sim = sd.SimDir("kuibit")
        self.assertIs(empty_sim.gws.l_max, None)
