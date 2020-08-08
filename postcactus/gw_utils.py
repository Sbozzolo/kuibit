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
        return c / H0 * (1 + z) * integrate.quad(DL_integral, 0,
                                                 z)[0]

    def function_to_root_find(z):
        return np.abs(distance_in_m - z_to_DL(z))

    redshift = optimize.root(function_to_root_find, initial_guess)

    if (redshift['status'] != 1):
        raise RuntimeError("Conversion between distance and redshift failed!")

    return redshift['x'][0]
