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

"""The :py:mod:`~.sensitivity_curves` module has functions to compute the power
spectral noise densities for known detectors.

"""

import numpy as np

from postcactus.frequencyseries import FrequencySeries
from postcactus.unitconv import C_SI


def Sn_LISA(freqs, arms_length=2.5e9):
    """Return the average power spectral density noise for LISA in 1/Hz.

    Equation (13) in 1803.01944

    :param freqs: Frequencies in Hz over to evaluate the sensitivity curve.
    :type freqs: 1d numpy array
    :param arms_length: Length of the detector's arms in meters.
    :type arms_length: float

    :returns: LISA sensitivity curve in 1/Hz.
    :rtype: :py:class:`~.FrequencySeries`

    """
    freqs = np.asarray(freqs)

    def P_acc(f):
        """Equation (11)"""
        return (
            (3.0e-15) ** 2
            * (1.0 + (0.4e-3 / f) ** 2)
            * (1.0 + (f / 8e-3) ** 4)
        )  # 1/Hz

    def P_OMS(f):
        """Equation (10)"""
        return (1.5e-11) ** 2 * (1.0 + (2e-3 / f) ** 4)  # 1/Hz

    # Transfer frequency
    f_star = C_SI / (2 * np.pi * arms_length)

    # Equation (13)
    Sn = (
        (10.0 / (3 * arms_length ** 2))
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
