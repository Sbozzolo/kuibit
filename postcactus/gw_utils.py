#!/usr/bin/env python3
"""Convenience functions to analyze and manipulate gravitational waves.

"""

import datetime
from collections import namedtuple
import numpy as np
import postcactus.unitconv as uc
from scipy import optimize
from scipy import integrate


def luminosity_distance_to_redshift(luminosity_distance,
                                    Omega_m=0.309,
                                    Omega_L=0.691,
                                    initial_guess=0.1):
    r"""Compute redshift from luminosity distance in Mpc assuming
    the LCMD cosmology.

    This function is useful to correctly reproduce observed signals
    from cosmological sources (e.g., binary black holes far away).

    The redshift is computed via root-finding, so an initial guess
    is needed.

    :param luminosity_distance: Luminosity distance in Mpc
    :type luminosity_distance: float
    :param Omega_m: :math:`\Omega_m` (matter) cosmological parameter
    :type Omega_m: float
    :param Omega_L: :math:`\Omega_m` (dark energy) cosmological parameter
    :type Omega_L: float
    :param initial_guess: Initial guess to the redshift for the
                          root-finding routine
    :type initial_guess: float

    :rvalue z: Redshift
    :rtype z: float

    """
    distance_in_m = luminosity_distance * uc.MEGAPARSEC_SI  # m
    H0 = uc.H0_SI  # 1/s
    c = uc.C_SI  # m/s

    def DL_integral(z):
        return 1 / np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

    def z_to_DL(z):
        return c / H0 * (1 + z) * integrate.quad(DL_integral, 0, z)[0]

    def function_to_root_find(z):
        return np.abs(distance_in_m - z_to_DL(z))

    redshift = optimize.root(function_to_root_find, initial_guess)

    if (redshift['status'] != 1):
        raise RuntimeError("Conversion between distance and redshift failed!")

    return redshift['x'][0]


def sYlm(ss, ll, mm, theta, phi):
    """Compute spin-weighted spherical harmonics at the angles theta and phi.

    When ss = 0, results are spherical harmonics.

    :param ss: Spin weight
    :type ss: int
    :param ll: l multipolar number
    :type ll: int
    :param mm: l multipolar number
    :type mm: int
    :param theta: Meridional angle
    :type theta: float
    :param phi: Azimuthal angle
    :type phi: float

    :rvalue sYlm: Spin-weighted spherical harmonics
    :rtype sYlm: float

    """
    # Code by Christian Reisswig

    # Coefficient function for spin-weighted spherical harmonics
    def sYlm_Cslm(local_s, local_l, local_m):
        return np.sqrt(local_l * local_l * (4.0 * local_l * local_l - 1.0) /
                       ((local_l * local_l - local_m * local_m) *
                        (local_l * local_l - local_s * local_s)))

    # Recursion function for spin-weighted spherical harmonics
    def s_lambda_lm(local_s, local_l, local_m, x):

        Pm = np.power(-0.5, local_m)

        if (local_m != local_s):
            Pm = Pm * np.power(1.0 + x, (local_m - local_s) * 1.0 / 2)
        if (local_m != -local_s):
            Pm = Pm * np.power(1.0 - x, (local_m + local_s) * 1.0 / 2)

        Pm = Pm * np.sqrt(
            np.math.factorial(2 * local_m + 1) * 1.0 /
            (4.0 * np.pi * np.math.factorial(local_m + local_s) *
             np.math.factorial(local_m - local_s)))

        if (local_l == local_m):
            return Pm

        Pm1 = (x + local_s * 1.0 /
               (local_m + 1)) * sYlm_Cslm(local_s, local_m + 1, local_m) * Pm

        if (local_l == local_m + 1):
            return Pm1

        for n in range(local_m + 2, local_l + 1):
            Pn = ((x + local_s*local_m * 1.0 / (n * (n-1.0)))
                  * sYlm_Cslm(local_s, n, local_m) * Pm1 -
                  sYlm_Cslm(local_s, n, local_m) * 1.0 /
                  sYlm_Cslm(local_s, n-1, local_m) * Pm)
            Pm = Pm1
            Pm1 = Pn

        return Pn

    Pm = 1.0

    mult_l = ll
    mult_m = mm
    mult_s = ss

    if (mult_l < 0):
        return 0
    if (abs(mult_m) > mult_l or mult_l < abs(mult_s)):
        return 0

    if (abs(mm) < abs(ss)):
        mult_s = mm
        mult_m = ss
        if ((mult_m + mult_s) % 2):
            Pm = -Pm

    if (mult_m < 0):
        mult_s = -mult_s
        mult_m = -mult_m
        if ((mult_m + mult_s) % 2):
            Pm = -Pm

    result = Pm * s_lambda_lm(mult_s, mult_l, mult_m, np.cos(theta))

    return complex(result * np.cos(mm * phi), result * np.sin(mm * phi))


def antenna_responses(right_ascension, declination, time_utc, polarization=0):
    """Return the antenna responses for Hanford and Livingston for a given
    source.

    See, research.physics.illinois.edu/cta/movies/bhbh_sim/wavestrain.html,
    considering that there is typo in the sign of the second cosine in the F
    cross.

    utc_time has to have the following formatting: %Y-%m-%d %H:%M,
    eg 2015-09-14 09:50:45

    Return values are plus and cross responses for Hanford, Livingston, and
    Virgo.

    :param right_ascension: Right ascension of the source in degrees
    :type right_ascension: float
    :param declination: Declination of the source in degrees
    :type declination: float
    :param time_utc: UTC time of the event
    :type declination: str
    :param polarization: Polarization of the wave
    :type polarization: float

    :rvalue antenna_pattern: Cross and plus antenna pattern for the different
    interferometers
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
    lat_V = (43 + 37.0/60 + 53.0/3600)*deg_to_rad
    long_V = -(10 + 30.0/60 + 16.0/3600)*deg_to_rad
    xazi_V = -(180-19)*deg_to_rad

    # Time of detection of GW150914: 2015-09-14 09:50:45 UTC
    # The Greenwich sidereal time theta_G is calculated by
    # the formula at https://en.wikipedia.org/wiki/Sidereal_time

    base_posix_date = datetime.datetime.strptime(
        "2000-01-01 12:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
    posix_date = datetime.datetime.strptime(time_utc,
                                            "%Y-%m-%d %H:%M:%S").timestamp()
    # Days between DATE and 2000-01-01 12:00
    D = (posix_date - base_posix_date) / 86400
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
        np.sin(lat_H) * np.sin(delta) +
        np.cos(lat_H) * np.cos(delta) * np.cos(h_H))
    theta_L = np.arccos(
        np.sin(lat_L) * np.sin(delta) +
        np.cos(lat_L) * np.cos(delta) * np.cos(h_L))
    theta_V = np.arccos(
        np.sin(lat_V) * np.sin(delta) +
        np.cos(lat_V) * np.cos(delta) * np.cos(h_V))

    A_H = np.arctan2(
        np.cos(delta) * np.sin(h_H),
        np.cos(delta) * np.cos(h_H) * np.sin(lat_H) -
        np.sin(delta) * np.cos(lat_H))
    phi_H = xazi_H - A_H

    A_L = np.arctan2(
        np.cos(delta) * np.sin(h_L),
        np.cos(delta) * np.cos(h_L) * np.sin(lat_L) -
        np.sin(delta) * np.cos(lat_L))
    phi_L = xazi_L - A_L

    A_V = np.arctan2(
        np.cos(delta) * np.sin(h_V),
        np.cos(delta) * np.cos(h_V) * np.sin(lat_V) -
        np.sin(delta) * np.cos(lat_V))
    phi_V = xazi_V - A_V

    # Equations from the website with corrected typo in the sign of F_cross
    # (It should be + in front of the cosine)
    Fp_H = (0.5 * (1 + np.cos(theta_H) * np.cos(theta_H))
            * np.cos(2 * phi_H) * np.cos(2 * polarization)
            - np.cos(theta_H) * np.sin(2 * phi_H) * np.sin(2 * polarization))
    Fc_H = (0.5 * (1 + np.cos(theta_H) * np.cos(theta_H))
            * np.cos(2 * phi_H) * np.sin(2 * polarization)
            + np.cos(theta_H) * np.sin(2 * phi_H) * np.cos(2 * polarization))

    Fp_L = (0.5 * (1 + np.cos(theta_L) * np.cos(theta_L))
            * np.cos(2 * phi_L) * np.cos(2 * polarization)
            - np.cos(theta_L) * np.sin(2 * phi_L) * np.sin(2 * polarization))
    Fc_L = (0.5 * (1 + np.cos(theta_L) * np.cos(theta_L))
            * np.cos(2 * phi_L) * np.sin(2 * polarization)
            + np.cos(theta_L) * np.sin(2 * phi_L) * np.cos(2 * polarization))

    Fp_V = (0.5 * (1 + np.cos(theta_V) * np.cos(theta_V))
            * np.cos(2 * phi_V) * np.cos(2 * polarization)
            - np.cos(theta_V) * np.sin(2 * phi_V) * np.sin(2 * polarization))
    Fc_V = (0.5 * (1 + np.cos(theta_V) * np.cos(theta_V))
            * np.cos(2 * phi_V) * np.sin(2 * polarization)
            + np.cos(theta_V) * np.sin(2 * phi_V) * np.cos(2 * polarization))


    # This is just a convenience to avoid having to remember the order of
    # the output
    # One can access the output as antenna_LIGO.hanford, or
    # antenna_LIGO['hanford']
    AntennaResponse = namedtuple("AntennaResponse", "hanford livingston virgo")
    antenna = AntennaResponse(hanford=(Fc_H, Fp_H),
                              livingston=(Fc_L, Fp_L),
                              virgo=(Fc_V, Fp_V))

    return antenna
