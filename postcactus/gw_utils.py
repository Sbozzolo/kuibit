#!/usr/bin/env python3
"""Convenience functions to analyze and manipulate gravitational waves.

"""

import postcactus.unitconv as uc
from scipy import optimize
from scipy import integrate
import numpy as np


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
