#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
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

"""The :py:mod:`~.sensitivity_curves` module contains the power spectral
distribution of the noise of known detectors. The available ones are:

- :py:func:`~.Sn_LISA`
- :py:func:`~.Sn_ET_B`
- :py:func:`~.Sn_CE1`
- :py:func:`~.Sn_CE2`
- :py:func:`~.Sn_aLIGO`
- :py:func:`~.Sn_voyager`
- :py:func:`~.Sn_KAGRA_D`
- :py:func:`~.Sn_aLIGO_plus`

"""

import pkgutil
from io import StringIO

import numpy as np

from kuibit.frequencyseries import FrequencySeries, load_FrequencySeries
from kuibit.unitconv import C_SI


def Sn_LISA(freqs, arms_length=2.5e9):
    """Return the average power spectral density noise for LISA in 1/Hz.

    The expression implemented is Equation (13) in
    https://arxiv.org/abs/1803.01944.

    :param freqs: Frequencies in Hz over which to evaluate the sensitivity curve.
    :type freqs: 1d NumPy array
    :param arms_length: Length of the detector's arms in meters.
    :type arms_length: float

    :returns: LISA sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    def P_acc(f):
        """Equation (11) of 1803.01944"""
        return (
            (3.0e-15) ** 2
            * (1.0 + (0.4e-3 / f) ** 2)
            * (1.0 + (f / 8e-3) ** 4)
        )  # 1/Hz

    def P_OMS(f):
        """Equation (10) of 1803.01944"""
        return (1.5e-11) ** 2 * (1.0 + (2e-3 / f) ** 4)  # 1/Hz

    # Transfer frequency
    f_star = C_SI / (2 * np.pi * arms_length)

    # This is Equation (13)
    Sn = (
        (10.0 / (3 * arms_length**2))
        * (
            P_OMS(freqs)
            + 2
            * (1.0 + np.cos(freqs / f_star) ** 2)
            * P_acc(freqs)
            / ((2 * np.pi * freqs) ** 4)
        )
        * (1.0 + (6.0 / 10.0) * (freqs / f_star) ** 2)
    )

    return FrequencySeries(freqs, Sn)


def Sn_ET_B(freqs):
    """Return the average power spectral density noise for Einstein Telescope
    (variant B) in 1/Hz.

    .. note::

        The data was downloaded from
        https://apps.et-gw.eu/tds/?content=3&r=14323 and has range ``fmin=1``,
        ``fmax=10000`` (Hz).

    :param freqs: Frequencies in Hz over which to evaluate the sensitivity curve.
    :type freqs: 1d NumPy array

    :returns: ET-B sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    # Why is it so difficult to read files in Python packages? :(
    data = pkgutil.get_data("kuibit", "data/ETB.dat").decode("utf8")
    # We convert this data in a StringIO that NumPy can read, we can pass this
    # to load_FrequencySeries, since its backend is np.loadtxt
    #
    # ET distributes Amplitude Spectral Densities
    asd = load_FrequencySeries(StringIO(data))
    psd = asd**2

    # Resample on the requested frequencies. ETB is well-behaved, so it is fine
    # to use splines.
    psd.resample(freqs)

    return psd


def Sn_ET_D(freqs):
    """Return the average power spectral density noise for Einstein Telescope
    (variant D) in 1/Hz.

    .. note::

        The data was downloaded from
        https://apps.et-gw.eu/tds/?content=3&r=14065 and has range ``fmin=1``,
        ``fmax=10000`` (Hz).

    :param freqs: Frequencies in Hz over which to evaluate the sensitivity curve.
    :type freqs: 1d NumPy array

    :returns: ET-D sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    # Why is it so difficult to read files in Python packages? :(
    data = pkgutil.get_data("kuibit", "data/ETD.dat").decode("utf8")
    # We convert this data in a StringIO that NumPy can read, we can pass this
    # to load_FrequencySeries, since its backend is np.loadtxt
    #
    # ET-D has four columns: freq, ET-D-LF, ET-D-HF, ET-D-sum. We only care
    # about the last one
    f, _, _, fft = np.loadtxt(StringIO(data), unpack=True)
    asd = FrequencySeries(f, fft)
    psd = asd**2

    # Resample on the requested frequencies. ETD has some spikes, so it is
    # better to not use splines.
    psd.resample(freqs, piecewise_constant=True)

    return psd


def Sn_CE1(freqs):
    """Return the average power spectral density noise for Einstein Telescope in
    1/Hz.

    .. note::

        The data was downloaded from
        https://cosmicexplorer.org/data/CE1_strain.txt and has range ``fmin=3``,
        ``fmax=10000`` (Hz).

    The resampling to the requested frequencies is done considering the values
    of the nearest neighbors.

    :param freqs: Frequencies in Hz over which to evaluate the sensitivity curve.
    :type freqs: 1d NumPy array

    :returns: CE1 sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    data = pkgutil.get_data("kuibit", "data/CE1.dat").decode("utf8")

    # CE distributes Amplitude Spectral Densities
    asd = load_FrequencySeries(StringIO(data))
    psd = asd**2

    # Resample on the requested frequencies. CE1 has some spikes, so it is
    # better to not use splines.
    psd.resample(freqs, piecewise_constant=True)

    return psd


def Sn_CE2(freqs):
    """Return the average power spectral density noise for Einstein Telescope in
    1/Hz.

    .. note::

        The data was downloaded from
        https://cosmicexplorer.org/data/CE2_strain.txt and has range ``fmin=3``,
        ``fmax=10000`` (Hz).

    The resampling to the requested frequencies is done considering the values
    of the nearest neighbors.

    :param freqs: Frequencies in Hz over which to evaluate the sensitivity curve.
    :type freqs: 1d NumPy array

    :returns: CE2 sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    data = pkgutil.get_data("kuibit", "data/CE1.dat").decode("utf8")

    # CE distributes Amplitude Spectral Densities
    asd = load_FrequencySeries(StringIO(data))
    psd = asd**2

    # Resample on the requested frequencies. CE1 has some spikes, so it is
    # better to not use splines.
    psd.resample(freqs, piecewise_constant=True)

    return psd


def Sn_aLIGO(freqs):
    """Return the average power spectral density noise for advanced LIGO in
    1/Hz. This is the Zero-Detuned-High-Power noise curve.

    .. note::

        The data was downloaded from
        https://dcc.ligo.org/LIGO-T1500293-v11/public and has range ``fmin=9``,
        ``fmax=8192`` (Hz).

    The resampling to the requested frequencies is done considering the values
    of the nearest neighbors.

    :param freqs: Frequencies in Hz over to evaluate the sensitivity curve.
    :type freqs: 1d NumPy array

    :returns: Advanced LIGO Zero-Detuned High-Power sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    data = pkgutil.get_data("kuibit", "data/aLIGO_ZDHP.dat").decode("utf8")

    # aLIGO distributes Amplitude Spectral Densities
    asd = load_FrequencySeries(StringIO(data))
    psd = asd**2

    # Resample on the requested frequencies. CE1 has some spikes, so it is
    # better to not use splines.
    psd.resample(freqs, piecewise_constant=True)

    return psd


def Sn_voyager(freqs):
    """Return the average power spectral density noise for voyager in
    1/Hz.

    .. note::

        The data was downloaded from
        https://dcc.ligo.org/LIGO-T1500293-v11/public and has range ``fmin=5``,
        ``fmax=10000`` (Hz).

    The resampling to the requested frequencies is done considering the values
    of the nearest neighbors.

    :param freqs: Frequencies in Hz over to evaluate the sensitivity curve.
    :type freqs: 1d NumPy array

    :returns: Voyager sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    data = pkgutil.get_data("kuibit", "data/voyager.dat").decode("utf8")

    # Voyager distributes Amplitude Spectral Densities
    asd = load_FrequencySeries(StringIO(data))
    psd = asd**2

    # Resample on the requested frequencies. CE1 has some spikes, so it is
    # better to not use splines.
    psd.resample(freqs, piecewise_constant=True)

    return psd


def Sn_KAGRA_D(freqs):
    """Return the average power spectral density noise for KAGRA in
    1/Hz.


    .. note::

        The data was downloaded from
        https://granite.phys.s.u-tokyo.ac.jp/svn/LCGT/trunk/sensitivity/spectrum/BW2009_VRSED.dat
        and has range ``fmin=1.00231``, ``fmax=10000`` (Hz).

    The resampling to the requested frequencies is done considering the values
    of the nearest neighbors.

    :param freqs: Frequencies in Hz over to evaluate the sensitivity curve.
    :type freqs: 1d NumPy array

    :returns: KAGRA sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    data = pkgutil.get_data("kuibit", "data/KAGRA_VRSED.dat").decode("utf8")

    # KAGRA distributes Amplitude Spectral Densities
    asd = load_FrequencySeries(StringIO(data))
    psd = asd**2

    # Resample on the requested frequencies. CE1 has some spikes, so it is
    # better to not use splines.
    psd.resample(freqs, piecewise_constant=True)

    return psd


def Sn_aLIGO_plus(freqs):
    """Return the average power spectral density noise for Advanced LIGO + in
    1/Hz.

    .. note::

        The data was downloaded from
        https://dcc.ligo.org/public/0149/T1800042/005/AplusDesign.txt and has
        range ``fmin=5``, ``fmax=5000`` (Hz).

    The resampling to the requested frequencies is done considering the values
    of the nearest neighbors.

    :param freqs: Frequencies in Hz over to evaluate the sensitivity curve.
    :type freqs: 1d NumPy array

    :returns: aLIGO+ sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    data = pkgutil.get_data("kuibit", "data/aLIGO_PLUS.dat").decode("utf8")

    # aLIGO distributes Amplitude Spectral Densities
    asd = load_FrequencySeries(StringIO(data))
    psd = asd**2

    # Resample on the requested frequencies. CE1 has some spikes, so it is
    # better to not use splines.
    psd.resample(freqs, piecewise_constant=True)

    return psd
