#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
# sYlm: Copyright (C) Christian Reisswig
# ra_dec_to_theta_phi: Copyright (C) 2020-2021 Yuk Tung Liu
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


"""The module :py:mod:`~.gw_utils` contains convenience functions and
structures to analyze and work with gravitational waves.

First, the ``Detectors`` object is defined. ``Detectors`` is a named tuple
with fields "hanford", "livingston", "virgo". This is used every time we
deal with specific detectors.

The functions provided are:

- :py:func:`~.luminosity_distance_to_redshift`: convert a given luminosity
  distance to a redshift in the ΛCDM cosmology.
- :py:func:`~.sYlm`: return the spin-weighted spherical harmonics at a given
  angle.
- :py:func:`~.ra_dec_to_theta_phi`: convert right ascension and declination to
  spherical coordinates.
- :py:func:`~.antenna_responses`: compute the antenna responses of a given
  angle.
- :py:func:`~.antenna_responses_from_sky_localization`: compute the antenna
  responses for known detectors at a given sky localization.

- :py:func:`~.signal_to_noise_ratio_from_strain`: compute the signal to noise
  for a given signal and a given noise curve.

"""

import datetime
from collections import namedtuple

import numpy as np
from scipy import integrate, optimize

import kuibit.timeseries as ts
import kuibit.unitconv as uc

# This is just a convenience to avoid having to remember the order of
# the output (and for easy of extension)
# One can access the output as detectos.hanford, or
# detectors['hanford']
Detectors = namedtuple("Detectors", "hanford livingston virgo")


def luminosity_distance_to_redshift(
    luminosity_distance,
    Omega_m=0.309,
    Omega_L=0.691,
    initial_guess=0.1,
):
    r"""Compute redshift from luminosity distance in Mpc assuming the ΛCDM cosmology.

    This function is useful to correctly reproduce observed signals
    from cosmological sources (e.g., binary black holes far away).

    The redshift is computed via root-finding, so an initial guess is needed.

    :param luminosity_distance: Luminosity distance in megaparsec.
    :type luminosity_distance: float
    :param Omega_m: :math:`\Omega_m` (matter) cosmological parameter.
    :type Omega_m: float
    :param Omega_L: :math:`\Omega_m` (dark energy) cosmological parameter.
    :type Omega_L: float
    :param initial_guess: Initial guess to the redshift for the
                          root-finding routine.
    :type initial_guess: float

    :returns z: Redshift
    :rtype z: float

    """
    distance_in_m = luminosity_distance * uc.MEGAPARSEC_SI  # m
    H0 = uc.H0_SI  # 1/s
    c = uc.C_SI  # m/s

    def function_to_root_find(z):
        def z_to_DL(z):
            def DL_integral(z):
                return 1 / np.sqrt(Omega_m * (1 + z) ** 3 + Omega_L)

            return c / H0 * (1 + z) * integrate.quad(DL_integral, 0, z)[0]

        return np.abs(distance_in_m - z_to_DL(z))

    redshift = optimize.root(function_to_root_find, initial_guess)

    if redshift["status"] != 1:
        raise RuntimeError("Conversion between distance and redshift failed!")

    return redshift["x"][0]


def sYlm(ss, ll, mm, theta, phi):
    """Compute spin-weighted spherical harmonics at the angles ``theta`` and ``phi``.

    When ``ss = 0``, these are spherical harmonics.

    :param ss: Spin weight.
    :type ss: int
    :param ll: :math:`l` multipolar number.
    :type ll: int
    :param mm: :math:`m` multipolar number.
    :type mm: int
    :param theta: Meridional angle.
    :type theta: float
    :param phi: Azimuthal angle.
    :type phi: float

    :returns sYlm: Spin-weighted spherical harmonic evaluated at
                  ``theta`` and ``phi``.
    :rtype sYlm: float

    """
    # Code by Christian Reisswig

    # Recursion function for spin-weighted spherical harmonics
    def s_lambda_lm(local_s, local_l, local_m, x):
        # Coefficient function for spin-weighted spherical harmonics
        def sYlm_Cslm(local_s, local_l, local_m):
            return np.sqrt(
                local_l
                * local_l
                * (4.0 * local_l * local_l - 1.0)
                / (
                    (local_l * local_l - local_m * local_m)
                    * (local_l * local_l - local_s * local_s)
                )
            )

        Pm = np.power(-0.5, local_m)

        if local_m != local_s:
            Pm = Pm * np.power(1.0 + x, (local_m - local_s) * 1.0 / 2)
        if local_m != -local_s:
            Pm = Pm * np.power(1.0 - x, (local_m + local_s) * 1.0 / 2)

        Pm = Pm * np.sqrt(
            np.math.factorial(2 * local_m + 1)
            * 1.0
            / (
                4.0
                * np.pi
                * np.math.factorial(local_m + local_s)
                * np.math.factorial(local_m - local_s)
            )
        )

        if local_l == local_m:
            return Pm

        Pm1 = (
            (x + local_s * 1.0 / (local_m + 1))
            * sYlm_Cslm(local_s, local_m + 1, local_m)
            * Pm
        )

        if local_l == local_m + 1:
            return Pm1

        for n in range(local_m + 2, local_l + 1):
            Pn = (x + local_s * local_m * 1.0 / (n * (n - 1.0))) * sYlm_Cslm(
                local_s, n, local_m
            ) * Pm1 - sYlm_Cslm(local_s, n, local_m) * 1.0 / sYlm_Cslm(
                local_s, n - 1, local_m
            ) * Pm
            Pm = Pm1
            Pm1 = Pn

        return Pn

    Pm = 1.0

    mult_l = ll
    mult_m = mm
    mult_s = ss

    if mult_l < 0:
        return 0
    if abs(mult_m) > mult_l or mult_l < abs(mult_s):
        return 0

    if abs(mm) < abs(ss):
        mult_s = mm
        mult_m = ss
        if (mult_m + mult_s) % 2:
            Pm = -Pm

    if mult_m < 0:
        mult_s = -mult_s
        mult_m = -mult_m
        if (mult_m + mult_s) % 2:
            Pm = -Pm

    result = Pm * s_lambda_lm(mult_s, mult_l, mult_m, np.cos(theta))

    return complex(result * np.cos(mm * phi), result * np.sin(mm * phi))


def ra_dec_to_theta_phi(right_ascension, declination, time_utc):
    """Compute the spherical angles ``theta`` and ``phi`` for Hanford, Livingston
    and Virgo for a given source localization.

    ``utc_time`` has to have the following formatting: ``%Y-%m-%d %H:%M``,
    (eg ``2015-09-14 09:50:45``)

    :param right_ascension: Right ascension of the source in degrees.
    :type right_ascension: float
    :param declination: Declination of the source in degrees.
    :type declination: float
    :param time_utc: UTC time of the event.
    :type declination: str

    :returns spherical coordinates: ``Theta``, ``phi`` for the different detectors.
    :rtype: namedtuple with fields hanford, livingston, and virgo

    """
    # NOTE: This function can make use of some cleanup and double-checking
    # This code is based on the one written by Yuk Tung Liu.

    # RA and DEC of the GW source:
    alpha_hours = right_ascension
    delta_degrees = declination

    # degrees to radian
    deg_to_rad = np.pi / 180.0

    alpha = alpha_hours * 15.0 * deg_to_rad
    delta = delta_degrees * deg_to_rad

    # LIGO detector geometry from
    # http://www.ligo.org/scientists/GW100916/GW100916-geometry.html
    # and also
    # http://www.ligo.org/scientists/GRB051103/GRB051103-geometry.php
    # The two pages are consistent.

    # LIGO Hanford (H) detector geometry
    # Latitude, Longitude and azimuth of the x-arm
    # Note: Longitude is measured positively westward from Greenwich
    # Azimuth is measured from the South point, turning positive to the West.
    lat_H = (46 + 27.0 / 60 + 19.0 / 3600) * deg_to_rad
    long_H = (119 + 24.0 / 60 + 28.0 / 3600) * deg_to_rad
    xazi_H = (180 - 36) * deg_to_rad

    # LIGO Livingston (L) detector geometry
    # Latitude, Longitude and azimuth of the x-arm
    lat_L = (30 + 33.0 / 60 + 46.0 / 3600) * deg_to_rad
    long_L = (90 + 46.0 / 60 + 27.0 / 3600) * deg_to_rad
    xazi_L = (90 - 18) * deg_to_rad

    # Virgo detector geometry
    # https://arxiv.org/pdf/1706.09505.pdf (Table 3.1)
    lat_V = (43 + 37.0 / 60 + 53.0 / 3600) * deg_to_rad
    long_V = -(10 + 30.0 / 60 + 16.0 / 3600) * deg_to_rad
    xazi_V = -(180 - 19) * deg_to_rad

    # Time of detection of GW150914: 2015-09-14 09:50:45 UTC
    # The Greenwich sidereal time theta_G is calculated by
    # the formula at https://en.wikipedia.org/wiki/Sidereal_time

    base_date = datetime.datetime.strptime(
        "2000-01-01 12:00:00", "%Y-%m-%d %H:%M:%S"
    )
    date = datetime.datetime.strptime(time_utc, "%Y-%m-%d %H:%M:%S")
    # Days between DATE and 2000-01-01 12:00
    D = (date - base_date).total_seconds() / 86400
    theta_G = 18.697374558 + 24.06570982441908 * D
    # theta_G mod 24h
    theta_G = theta_G - np.floor(theta_G / 24.0) * 24
    theta_G = theta_G * 15 * deg_to_rad

    # Equatorial -> horizontal using the formulae at
    # https://en.wikipedia.org/wiki/Celestial_coordinate_system

    # Hour angles
    h_H = theta_G - long_H - alpha
    h_L = theta_G - long_L - alpha
    h_V = theta_G - long_V - alpha

    # Zenith distance theta and azimuth A
    theta_H = np.arccos(
        np.sin(lat_H) * np.sin(delta)
        + np.cos(lat_H) * np.cos(delta) * np.cos(h_H)
    )
    theta_L = np.arccos(
        np.sin(lat_L) * np.sin(delta)
        + np.cos(lat_L) * np.cos(delta) * np.cos(h_L)
    )
    theta_V = np.arccos(
        np.sin(lat_V) * np.sin(delta)
        + np.cos(lat_V) * np.cos(delta) * np.cos(h_V)
    )

    A_H = np.arctan2(
        np.cos(delta) * np.sin(h_H),
        np.cos(delta) * np.cos(h_H) * np.sin(lat_H)
        - np.sin(delta) * np.cos(lat_H),
    )
    phi_H = xazi_H - A_H

    A_L = np.arctan2(
        np.cos(delta) * np.sin(h_L),
        np.cos(delta) * np.cos(h_L) * np.sin(lat_L)
        - np.sin(delta) * np.cos(lat_L),
    )
    phi_L = xazi_L - A_L

    A_V = np.arctan2(
        np.cos(delta) * np.sin(h_V),
        np.cos(delta) * np.cos(h_V) * np.sin(lat_V)
        - np.sin(delta) * np.cos(lat_V),
    )
    phi_V = xazi_V - A_V

    coords = Detectors(
        hanford=(theta_H, phi_H),
        livingston=(theta_L, phi_L),
        virgo=(theta_V, phi_V),
    )

    return coords


def antenna_responses(theta, phi, polarization=0):
    """Return the antenna response pattern of a detector on the z = 0 plane
    with the arms on the x and y directions for a given localization defined
    by the spherical angles ``theta`` and ``phi``.

    :param theta: Meridional angle.
    :type theta: float
    :param phi: Azimuthal angle.
    :type phi: float
    :param polarization: Polarization angle of the wave.
    :type polarization: float

    :returns: Antenna response for cross and plus polarizations (in this order).
    :rtype: tuple of floats.

    """
    # http://research.physics.illinois.edu/cta/movies/bhbh_sim/wavestrain.html

    Fp = 0.5 * (1 + np.cos(theta) * np.cos(theta)) * np.cos(2 * phi) * np.cos(
        2 * polarization
    ) - np.cos(theta) * np.sin(2 * phi) * np.sin(2 * polarization)
    Fc = 0.5 * (1 + np.cos(theta) * np.cos(theta)) * np.cos(2 * phi) * np.sin(
        2 * polarization
    ) + np.cos(theta) * np.sin(2 * phi) * np.cos(2 * polarization)

    return (Fc, Fp)


def antenna_responses_from_sky_localization(
    right_ascension, declination, time_utc, polarization=0
):
    """Return the antenna responses for Hanford, Livingston and Virgo for a
    given source.

    See,
    http://research.physics.illinois.edu/cta/movies/bhbh_sim/wavestrain.html.

    ``utc_time`` has to have the following formatting: ``%Y-%m-%d %H:%M``,
    (eg ``2015-09-14 09:50:45``)

    :param right_ascension: Right ascension of the source in degrees
    :type right_ascension: float
    :param declination: Declination of the source in degrees
    :type declination: float
    :param time_utc: UTC time of the event
    :type declination: str
    :param polarization: Polarization of the wave
    :type polarization: float

    :returns antenna_pattern: Cross and plus antenna pattern for the different
                             interferometers.
    :rtype: namedtuple with fields hanford, livingston, and virgo

    """

    coords = ra_dec_to_theta_phi(right_ascension, declination, time_utc)

    Fc_H, Fp_H = antenna_responses(*coords.hanford, polarization)
    Fc_L, Fp_L = antenna_responses(*coords.livingston, polarization)
    Fc_V, Fp_V = antenna_responses(*coords.virgo, polarization)

    antenna = Detectors(
        hanford=(Fc_H, Fp_H),
        livingston=(Fc_L, Fp_L),
        virgo=(Fc_V, Fp_V),
    )

    return antenna


# Extrapolation of GWs at infinity.
# Follows what done in the NRAR collaboration (1307.5307)
# This is what we are going to do:

# 1. Assume that the spacetime is almost Schwarzschild with mass M.
# 2. Choose a set of retarded times u_i where GWs have to be evaluated
# 3. Compute the coordinate times t_i that correspond to the retarded
#    times u_i at the radius r. This uses tortoise coordinates.
# 4. Interpolate the waveforms at the coordinate times corresponding to
#    the retarded times u_i for different extraction radii.
#    Now our waves are so that they are evaluate at different t_i but at
#    the same u_i.
# 5. We do this process for a bunch of extraction radii (rex1, rex2, rex3, ...)
#    So, we should have wave1, wave2, wave3, with all the same number of
#    points that correspond to the same retarded time.
# 6. For each element in u_i, fit all the waves (wave1, wave2, ...) with a
#    polynomial of the form a_n/r^n.


def Schwarzschild_radius_to_tortoise(radii, mass):
    """Transform radial coordinates ``radii`` to tortoise coordinates assuming mass
    ``mass``.

    Equation (26) in 1307.5307.

    :param radii: Radius in Schwarzschild coordinates.
    :type radii: float or 1D NumPy array
    :param mass: ADM mass.
    :type mass: float

    :returns: Tortoise radii.
    :rtype: float or 1D NumPy array

    """
    return radii + 2 * mass * np.log(radii / (2 * mass) - 1)


def retarded_times_to_coordinate_times(retarded_times, radii, mass):
    """Compute the coordinate times corresponding to the retarded times at the
      coordinate radii.

    First, the tortoise radius is computed from ``radii``, then the
    coordinate times are computed with :math:`t = u + r_{tortoise}(radii)`.

    This function is used to extrapolate gravitational waves to infinity.

    :param retarded_times: Retarded times.
    :type retarded_times: float or 1D NumPy array
    :param radii: Radii of evaluation.
    :type radii: float or 1D NumPy array
    :param mass: ADM mass (needed to compute tortoise radius).
    :type mass: float

    :returns: Coordinate times corresponding to the given retarded times
              and evaluation radii.
    :rtype: float or 1D NumPy array

    """
    return retarded_times + Schwarzschild_radius_to_tortoise(radii, mass)


def _coordinate_times_to_retarded_times(coordinate_times, radii, mass):
    """Compute the coordinate times corresponding to the retarded times
      at the given coordinate radii.

    This function is used to extrapolate gravitational waves to infinity.

    :param retarded_times: Coordinate times.
    :type retarded_times: float or 1D NumPy array
    :param radii: Radii (it can be just one)
    :type radii: float or 1D NumPy array
    :param mass: ADM mass
    :type mass: float

    :returns: Retarded times corresponding to the given coordinate times
              and evaluation radii.
    :rtype: float or 1D NumPy array

    """
    return coordinate_times - Schwarzschild_radius_to_tortoise(radii, mass)


def signal_to_noise_ratio_from_strain(
    h, *args, noise=None, fmin=0, fmax=np.inf, window_function=None, **kwargs
):
    r"""Return the signal to noise ratio given a strain and a power spectal density
    distribution for a detector.

    If ``window_function`` is not None, the window will be applied to the signal.
    All the unknown arguments are passed to the window function.

    The SNR is computed as :math:`sqrt of 4 \int_fmin^fmax |\tilde{h} f|^2 / Sn(f) d f`
    (equation from 1408.0740)

    :param h: Gravitational-wave strain.
    :type h: :py:class:`~.TimeSeries`
    :param noise: Power spectral density of the noise of the detector.
    :type noise: :py:class:`~.FrequencySeries`
    :param fmin: Minimum frequency over which to compute the SNR.
    :type fmin: float
    :param fmax: Maximum frequency over which to compute the SNR.
    :type fmax: float
    :param window_function: If not None, apply ``window_function`` to the
                            series before computing the strain.
    :type window_function: callable, str, or None
    :param args, kwargs: All the additional parameters are passed to
                         the window function.

    :returns: Signal-to-noise ratio.
    :rtype: float

    """
    if not isinstance(h, ts.TimeSeries):
        raise TypeError("Strain has to be a TimeSeries")
    # First, we window to avoid spectral leakage
    h_win = h.windowed(window_function, *args, **kwargs)
    # Then, we take the Fourier transform
    h_fft = h_win.to_FrequencySeries()
    # The SNR is just the inner product of h_fft with itself
    return np.sqrt(
        h_fft.inner_product(h_fft, noises=noise, fmin=fmin, fmax=fmax)
    )
