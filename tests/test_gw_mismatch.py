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

import sys
from importlib import reload
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from kuibit import frequencyseries as fs
from kuibit import gw_mismatch as gwm
from kuibit import gw_utils as gwu
from kuibit import timeseries as ts
from kuibit import unitconv as uc
from kuibit.cactus_waves import GravitationalWavesOneDet


class TestGWMismatch(unittest.TestCase):
    def setUp(self):
        self.num_times = 4000
        self.times = np.linspace(0, 20 * 2 * np.pi, self.num_times)
        self.values1 = np.sin(30 * self.times)
        self.values2 = np.sin(60 * self.times) * np.cos(30 * self.times)

        self.ts1 = ts.TimeSeries(self.times, self.values1)
        self.ts2 = ts.TimeSeries(self.times, self.values2)

    def test_mismatch_from_strains(self):

        # Test with PyCBC.
        fmin = 10
        fmax = 15

        # # PyCBC raises some benign warnings. We ignore them.
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     from pycbc.types import timeseries as pycbcts
        #     from pycbc.types import frequencyseries as pycbcfs
        #     from pycbc.filter import match

        # ts1_pcbc = pycbcts.TimeSeries(self.values1, delta_t=self.ts1.dt)
        # ts2_pcbc = pycbcts.TimeSeries(self.values2, delta_t=self.ts2.dt)
        # pycbc_m, _ = match(
        #     ts1_pcbc,
        #     ts2_pcbc,
        #     psd=None,
        #     low_frequency_cutoff=fmin,
        #     high_frequency_cutoff=fmax,
        # )

        # expected_mism = 1 - pycbc_m
        expected_mism = 0.9596805448739523

        # Test no noise, no antenna pattern, not using Numba
        # We are picking a very narrow time shift to reduce
        # the number of points needed
        o = gwm.mismatch_from_strains(
            self.ts1,
            self.ts2,
            fmin=fmin,  # It's important to have fmin != 0
            fmax=fmax,
            noises=None,
            # We only select the "real" polarization
            # because the signal is real
            antenna_patterns=[(0, 1)],
            num_polarization_shifts=50,
            num_time_shifts=50,
            time_shift_start=-1,
            time_shift_end=1,
            force_numba=False,
        )

        self.assertAlmostEqual(o[0], expected_mism, places=3)

        # Test no noise, no antenna pattern, using Numba

        # Since using numba takes time for compilation, we will only
        # do this test with it
        o_numba = gwm.mismatch_from_strains(
            self.ts1,
            self.ts2,
            fmin=fmin,
            fmax=fmax,
            noises=None,
            antenna_patterns=[(0, 1)],
            num_polarization_shifts=300,
            num_time_shifts=300,
            time_shift_start=-1,
            time_shift_end=1,
            force_numba=True,
        )

        self.assertAlmostEqual(o_numba[0], expected_mism, places=3)

        # Test fixed (differt) antenna pattern. Should not change the result
        # because it goes into the normalization.
        o_ap = gwm.mismatch_from_strains(
            self.ts1,
            self.ts2,
            fmin=fmin,
            fmax=fmax,
            noises=None,
            antenna_patterns=[(0.0, 2)],
            num_polarization_shifts=50,
            num_time_shifts=50,
            time_shift_start=-1,
            time_shift_end=1,
            force_numba=False,
        )

        self.assertAlmostEqual(o_ap[0], o[0])

        # Test fixed (different) noise. Should not change the result. Tests
        # also the resampling of the noise.
        freqs = np.linspace(0, 20 * 2 * np.pi, 10000)
        noise = fs.FrequencySeries(freqs, 2 * np.ones_like(freqs))
        o_noise = gwm.mismatch_from_strains(
            self.ts1,
            self.ts2,
            fmin=fmin,
            fmax=fmax,
            noises=[noise],
            antenna_patterns=[(0, 1)],
            num_polarization_shifts=50,
            num_time_shifts=50,
            time_shift_start=-1,
            time_shift_end=1,
            force_numba=False,
        )

        self.assertAlmostEqual(o_noise[0], o[0])

        # Now, let's try with two noises, but we set the antenna pattern of one
        # of the two to zero. The result should be the same as before.

        o_noise_fake2 = gwm.mismatch_from_strains(
            self.ts1,
            self.ts2,
            fmin=fmin,
            fmax=fmax,
            noises=[noise, noise],
            antenna_patterns=[(0, 1), (0, 0)],
            num_polarization_shifts=50,
            num_time_shifts=50,
            time_shift_start=-1,
            time_shift_end=1,
            force_numba=False,
        )

        self.assertAlmostEqual(o_noise_fake2[0], o[0])

        # Test with a None noise and the antenna pattern that kills the other
        # term.

        o_noise_fake3 = gwm.mismatch_from_strains(
            self.ts1,
            self.ts2,
            fmin=fmin,
            fmax=fmax,
            noises=[None, noise],
            antenna_patterns=[(0, 1), (0, 0)],
            num_polarization_shifts=50,
            num_time_shifts=50,
            time_shift_start=-1,
            time_shift_end=1,
            force_numba=False,
        )

        self.assertAlmostEqual(o_noise_fake3[0], o[0])

        # Test with non trivial noise
        # PyCBC requires the noise to be defined on the same frequencies as the
        # data
        # df_noise = ts1_pcbc.to_frequencyseries().delta_f
        df_noise = 0.00795575771780612
        f_noise = np.array([i * df_noise for i in range(10000)])

        # Funky looking noise
        psd_noise = np.abs(np.sin(300 * f_noise) + 0.5)
        noise2 = fs.FrequencySeries(f_noise, psd_noise)

        # PyCBC
        # noise_pycbc = pycbcfs.FrequencySeries(psd_noise, delta_f=df_noise)
        # pycbc_m_noise, u = match(
        #     ts1_pcbc,
        #     ts2_pcbc,
        #     psd=noise_pycbc,
        #     low_frequency_cutoff=fmin,
        #     high_frequency_cutoff=fmax,
        # )

        # # expected_mism_noise = 1 - pycbc_m_noise
        expected_mism_noise = 0.8654645514638336

        o_noise2 = gwm.mismatch_from_strains(
            self.ts1,
            self.ts2,
            fmin=fmin,
            fmax=fmax,
            noises=[noise2],
            antenna_patterns=[(0, 1)],
            num_polarization_shifts=50,
            num_time_shifts=200,
            time_shift_start=-2.5,
            time_shift_end=2.5,
            force_numba=False,
        )

        self.assertAlmostEqual(o_noise2[0], expected_mism_noise, places=2)

        # Test warning
        with self.assertWarns(Warning):
            gwm.mismatch_from_strains(
                self.ts1,
                self.ts2,
                fmin=fmin,
                fmax=fmax,
                num_polarization_shifts=50,
                num_time_shifts=20,
                time_shift_start=0.1,
                time_shift_end=0.19,
                force_numba=False,
            )

    def test_mismatch(self):

        fmin = 5
        fmax = 15

        # Invalid noise
        with self.assertRaises(TypeError):
            gwm.network_mismatch(
                self.ts1, self.ts2, 8, -70, "2015-09-14 09:50:45", noises=1
            )

        # No noise, three detectors
        antennas = gwu.antenna_responses_from_sky_localization(
            8, -70, "2015-09-14 09:50:45"
        )

        self.assertAlmostEqual(
            gwm.mismatch_from_strains(
                self.ts1,
                self.ts2,
                fmin=fmin,
                fmax=fmax,
                noises=None,
                antenna_patterns=list(antennas),
                num_polarization_shifts=30,
                num_time_shifts=30,
                time_shift_start=-70,
                time_shift_end=70,
            )[0],
            gwm.network_mismatch(
                self.ts1,
                self.ts2,
                8,
                -70,
                "2015-09-14 09:50:45",
                fmin=fmin,
                fmax=fmax,
                noises=None,
                num_polarization_shifts=30,
                num_time_shifts=30,
                time_shift_start=-70,
                time_shift_end=70,
            )[0],
        )

        # Only one active detector

        only_virgo = gwu.Detectors(hanford=-1, livingston=-1, virgo=None)

        self.assertAlmostEqual(
            gwm.mismatch_from_strains(
                self.ts1,
                self.ts2,
                fmin=fmin,
                fmax=fmax,
                noises=None,
                antenna_patterns=[antennas.virgo],
                num_polarization_shifts=30,
                num_time_shifts=30,
                time_shift_start=-70,
                time_shift_end=70,
            )[0],
            gwm.network_mismatch(
                self.ts1,
                self.ts2,
                8,
                -70,
                "2015-09-14 09:50:45",
                fmin=fmin,
                fmax=fmax,
                noises=only_virgo,
                num_polarization_shifts=30,
                num_time_shifts=30,
                time_shift_start=-70,
                time_shift_end=70,
            )[0],
        )

        # Test with a "gw-looking" singal from PyCBC
        #
        # First, we test the overlap by giving num_polarizations,
        # num_time_shifts=1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from pycbc.waveform import get_td_waveform
                from pycbc.types import timeseries as pycbcts
                from pycbc.filter import overlap
                from pycbc.filter import match
            fmin_gw = 50
            fmax_gw = 100
            delta_t = 1 / 4096

            hp1, hc1 = get_td_waveform(
                approximant="IMRPhenomPv2",
                mass1=10,
                mass2=10,
                spin1z=0.9,
                delta_t=delta_t,
                f_lower=40,
            )

            hp2, hc2 = get_td_waveform(
                approximant="IMRPhenomPv2",
                mass1=10,
                mass2=25,
                spin1z=-0.5,
                delta_t=delta_t,
                f_lower=40,
            )

            # PyCBC does not work well with series with different length. So, we
            # crop the longer one to the length of the shorter one. For the choice
            # of paramters, it is 2 that is shorter than 1. 1 starts earlier in the
            # past. However, they have the same frequencies, so we can simply crop
            # away the part we are not interested in.

            time_offset = 2  # Manually computed looking at the times
            hp1 = hp1.crop(time_offset, 0)
            hc1 = hc1.crop(time_offset, 0)

            # We apply the "antenna pattern"
            h1_pycbc = pycbcts.TimeSeries(
                0.33 * hp1 + 0.66 * hc1, delta_t=hp1.delta_t
            )
            h2_pycbc = pycbcts.TimeSeries(
                0.33 * hp2 + 0.66 * hc2, delta_t=hp2.delta_t
            )

            overlap_m = overlap(
                h1_pycbc,
                h2_pycbc,
                psd=None,
                low_frequency_cutoff=fmin_gw,
                high_frequency_cutoff=fmax_gw,
            )

            h1_postcac = ts.TimeSeries(h1_pycbc.sample_times, hp1 - 1j * hc1)
            h2_postcac = ts.TimeSeries(h2_pycbc.sample_times, hp2 - 1j * hc2)

            o = gwm.mismatch_from_strains(
                h1_postcac,
                h2_postcac,
                fmin=fmin_gw,
                fmax=fmax_gw,
                noises=None,
                antenna_patterns=[(0.66, 0.33)],
                num_polarization_shifts=1,
                num_time_shifts=1,
                time_shift_start=0,
                time_shift_end=0,
                force_numba=False,
            )

            self.assertAlmostEqual(1 - o[0], overlap_m, places=2)

            # Now we can test the mismatch
            pycbc_m, _ = match(
                h1_pycbc,
                h2_pycbc,
                psd=None,
                low_frequency_cutoff=fmin_gw,
                high_frequency_cutoff=fmax_gw,
            )

            pycbc_m = 1 - pycbc_m

            mat = gwm.mismatch_from_strains(
                h1_postcac,
                h2_postcac,
                fmin=fmin_gw,
                fmax=fmax_gw,
                noises=None,
                antenna_patterns=[(0.66, 0.33)],
                num_polarization_shifts=100,
                num_time_shifts=800,
                time_shift_start=-0.3,
                time_shift_end=0.3,
                force_numba=False,
            )

            self.assertAlmostEqual(mat[0], pycbc_m, places=2)
        except ImportError:  # pragma: no cover
            pass

    def test_mismatch_from_psi4(self):

        fmin = 5
        fmax = 15

        # The FFI integration of sin(ax) is -cos(ax)/a if pcut is very large

        data1 = [(2, 2, self.ts1)]
        data2 = [(2, 2, self.ts2)]

        psi1 = GravitationalWavesOneDet(100, data1)
        psi2 = GravitationalWavesOneDet(100, data2)

        h1 = psi1.get_strain_lm(2, 2, 1, 0.1, window_function="tukey")
        h2 = psi2.get_strain_lm(2, 2, 3, 0.1, window_function="tukey")

        h1.time_shift(-h1.time_at_maximum())
        h2.time_shift(-h2.time_at_maximum())

        h1.window("tukey", 0.1)
        h2.window("tukey", 0.1)

        h1.zero_pad(16384)
        h2.zero_pad(16384)

        self.assertAlmostEqual(
            gwm.network_mismatch_from_psi4(
                psi1,
                psi2,
                8,
                -70,
                "2015-09-14 09:50:45",
                1,
                3,
                0.1,
                window_function="tukey",
                fmin=fmin,
                fmax=fmax,
                noises=None,
                num_polarization_shifts=50,
                num_time_shifts=50,
                time_shift_start=-200,
                time_shift_end=200,
            )[0],
            gwm.network_mismatch(
                h1,
                h2,
                8,
                -70,
                "2015-09-14 09:50:45",
                fmin=fmin,
                fmax=fmax,
                noises=None,
                num_polarization_shifts=50,
                num_time_shifts=50,
                time_shift_start=-200,
                time_shift_end=200,
            )[0],
        )

        # If we pick a distance that is very close, the effect of the redshift
        # should be negligible, so the mismatch should not change. When we use
        # physical units, we have to rescale the frequencies and the times.
        CU = uc.geom_umass_msun(1)

        self.assertAlmostEqual(
            gwm.network_mismatch_from_psi4(
                psi1,
                psi2,
                8,
                -70,
                "2015-09-14 09:50:45",
                1,
                3,
                0.1,
                mass_scale1_msun=1,
                mass_scale2_msun=1,
                distance1=1e-6,
                distance2=1e-6,
                window_function="tukey",
                fmin=fmin * CU.freq,
                fmax=fmax * CU.freq,
                noises=None,
                time_removed_beginning=0,
                time_to_keep_after_max=1000,
                num_polarization_shifts=50,
                num_time_shifts=50,
                time_shift_start=-200 * CU.time,
                time_shift_end=200 * CU.time,
            )[0],
            gwm.network_mismatch(
                h1,
                h2,
                8,
                -70,
                "2015-09-14 09:50:45",
                fmin=fmin,
                fmax=fmax,
                noises=None,
                num_polarization_shifts=50,
                num_time_shifts=50,
                time_shift_start=-200,
                time_shift_end=200,
            )[0],
        )

        self.assertAlmostEqual(
            gwm.one_detector_mismatch_from_psi4(
                psi1,
                psi2,
                1,
                3,
                0.1,
                window_function="tukey",
                fmin=fmin,
                fmax=fmax,
                noise=None,
                time_removed_beginning=0,
                time_to_keep_after_max=1000,
                num_polarization_shifts=50,
                num_time_shifts=50,
                time_shift_start=-200,
                time_shift_end=200,
            )[0],
            gwm.mismatch_from_strains(
                h1,
                h2,
                fmin=fmin,
                fmax=fmax,
                noises=None,
                antenna_patterns=None,
                num_polarization_shifts=50,
                num_time_shifts=50,
                time_shift_start=-200,
                time_shift_end=200,
            )[0],
        )

    def test_mismatch_without_numba_installed(self):

        # We remove "njit" for the loaded module, so gwm thinks that numba is
        # not available
        del gwm.__dict__["njit"]

        fmin = 10
        fmax = 15
        expected_mism = 0.9596805448739523

        # We try to force numba
        with self.assertWarns(Warning):
            o = gwm.mismatch_from_strains(
                self.ts1,
                self.ts2,
                fmin=fmin,  # It's important to have fmin != 0
                fmax=fmax,
                noises=None,
                # We only select the "real" polarization
                # because the signal is real
                antenna_patterns=[(0, 1)],
                num_polarization_shifts=50,
                num_time_shifts=50,
                time_shift_start=-1,
                time_shift_end=1,
                force_numba=True,
            )

        self.assertAlmostEqual(o[0], expected_mism, places=3)
