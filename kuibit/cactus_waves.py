#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
#
# Inspired by code originally developed by Wolfgang Kastaun. See, GitHub,
# wokast/PyCactus/PostCactus/cactus_multipoles.py
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

"""The :py:mod:`~.cactus_waves` module provides classes to access gravitational
and electromagnetic wave signals computed using Weyl scalars.

The classes in this module specialize the ones in :py:mod:`~.cactus_multipoles`
to deal with gravitational and electromagnetic wave data.

There classes defined in this module are:

- :py:class`~.GravitationalWavesOneDet` and :py:class`~.ElectromagneticWavesOneDet`,
  which extend :py:class:`~.MultipoleOneDet` adding methods to compute quantities like
  strain, or energy lost.
- :py:class`~.WavesDir` (derived from :py:class`~.MultipoleAllDets`), from which we
  derive :py:class`~.GravitationalWavesDir` and :py:class`~.ElectromagneticWavesDir`,
  which organize the available data in terms of extraction radii.

"""

import warnings

import numpy as np

from kuibit import cactus_multipoles as mp
from kuibit import gw_utils, simdir
from kuibit import timeseries as ts
from kuibit.gw_utils import Detectors


class GravitationalWavesOneDet(mp.MultipoleOneDet):
    """This class represents is an abstract class to represent multipole
    signals from Weyl scalars available at a given distance.

    To check if component is available, use the operator "in". You can iterate
    over all the availble components with a for loop.

    This class is derived from :py:class:`~.MultipoleOneDet`, so it shares most
    of the features, while expanding with methods specific for gravitational
    waves (e.g, to compute the strain).

    This class is not intended to be initialized directly.

    """

    def __init__(self, dist, data):
        """Constructor.

        :param dist: Radius of the spherical surface.
        :type dist: float
        :param data: List of tuples with the two multipolar numbers and
                     the data as :py:class:`~.TimeSeries`.
        :type data: list of tuple ``(l, m, timeseries)``

        """

        super().__init__(dist, data, 2)

    # staticmethod means that this function will be allocated by python only
    # once, since it doesn't depend on the detail of the instance
    @staticmethod
    def _fixed_frequency_integrated(timeseries, pcut, order=1):
        r"""Return a new timeseries that is the one obtained with the method of
        the fixed frequency integration from the input ``timeseries``.

        ``pcut`` is the longest physical period in the system, corresponding to
        ``omega_threshold``,  the lowest physical frequency.

        ``order`` is order of integration (how many integrations).

        The fixed frequency integration algorithm is the following:

        The Fourier transform of f(t) is

        :math:`F[f](omega) = \int_-inf^inf e^-i omega t f(t) dt`

        The the Fourier transform of the integral of f(t) is
        :math:`F[f](omega) / i omega`

        In the FFI method we replace this with
        :math:`F[f](omega) / i omega`               if omega > omega_threshold
        :math:`F[f](omega) / i omega_threshold`     otherwise

        (Equation (27) in [arxiv:1006.1632])

        We can perform multiple integrations (needed for example to go from
        psi4 to h) by raising everything to the power of the order of
        integration:
        (due to the convolution theorem)

        :math:`F[f](omega) / (i omega)**order`             if omega > omega_threshold
        :math:`F[f](omega) / (i omega_threshold)**order`   otherwise

        Than, we take the inverse Fourier transform.

        :param timeseries: :py:class:`~TimeSeries` that has to be integrated.
        :type timeseries: :py:class:`~TimeSeries`
        :param pcut: Period associated with the threshold frequency
                     ``omega_0 = 2 * pi / pcut``
        :type pcut: float
        :param order: Number of integrations.
        :type order: int

        :returns: :py:class:`~TimeSeries` integrated with the fixed-frequency integration
                  method
        :rtype: :py:class:`~TimeSeries`

        """

        if not timeseries.is_regularly_sampled():
            warnings.warn(
                "Timeseries not regularly sampled. Resampling.",
                RuntimeWarning,
            )
            integrand = timeseries.regular_resampled()
        else:
            integrand = timeseries

        fft = np.fft.fft(integrand.y)
        omega = np.fft.fftfreq(len(integrand), d=integrand.dt) * (2 * np.pi)

        omega_abs = np.abs(omega)
        omega_threshold = 2 * np.pi / pcut

        # np.where(omega_abs > omega_threshold, omega_abs, omega_threshold)
        # means: return omega_abs when omega_abs > omega_threshold, otherwise
        # return omega_threshold
        ffi_omega = np.where(
            omega_abs > omega_threshold, omega_abs, omega_threshold
        )

        # np.sign(omega) / (ffi_omega) is omega when omega_abs > omega_thres
        # this is a convient way to group together positive and negative omega
        integration_factor = (np.sign(omega) / (1j * ffi_omega)) ** int(order)

        # Now, inverse fft
        integrated_y = np.fft.ifft(fft * integration_factor)

        return ts.TimeSeries(integrand.t, integrated_y)

    # This function is only for convenience
    def get_psi4_lm(self, mult_l, mult_m):
        r"""Return the multipolar components l and m of Psi4.

        :param mult_l:     Multipole component l.
        :type mult_l:      int
        :param mult_m:     Multipole component m.
        :type mult_m:      int

        :returns: :math:`\Psi_4^{lm}`
        :rtype: complex :py:class:`~.TimeSeries`
        """
        return self[(mult_l, mult_m)]

    def get_strain_lm(
        self,
        mult_l,
        mult_m,
        pcut,
        *args,
        window_function=None,
        trim_ends=True,
        **kwargs,
    ):
        r"""Return the strain associated to the multipolar component (l, m).

        The strain returned is multiplied by the distance.
        The strain is extracted from the Weyl Scalar using the formula

        .. math::

             h_+^{lm}(r,t)
             -     i h_\times^{lm}(r,t) = \int_{-\infty}^t \mathrm{d}u
                    \int_{-\infty}^u \mathrm{d}v\, \Psi_4^{lm}(r,v)

        The return value is the complex :py:class:`~.TimeSeries` ``(r * h_plus - i r * h_cross)``.

        It is always important to have a function that goes smoothly to zero
        before taking Fourier transform (to avoid spectral leakage and
        aliasing). You can pass the ``window_function`` to apply as a parameter.
        If ``window_function`` is None, no tapering is performed. If
        ``window_function`` is a function, it has to be a function that takes as
        first argument the length of the array and returns a new array with the
        same length that is to be multiplied to the data (this is how SciPy's
        windows work) If ``window_function`` is a string, use the method with
        corresponding name from the :py:class:`~.TimeSeries` class. You must
        only provide the name (e.g, 'tukey' will call 'tukey_windowed').
        Optional arguments to the window function can be passed directly to this
        function.

        ``pcut`` is the period associated to the angular velocity that enters in
        the fixed frequency integration (``omega_th = 2 pi / pcut``). In
        general, a wise choise is to pick the longest physical period in the
        signal.

        Optionally, remove part of the output signal at both the beginning and
        the end. If ``trim_ends`` is True, ``pcut`` is removed. This is because
        those parts of the signal are typically not very accurate.

        :param mult_l: Multipolar component l.
        :type mult_l: int
        :param mult_m: Multipolar component m.
        :type mult_m: int
        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float
        :param window_function: If not None, apply window_function to the
                                series before computing the strain.
        :type window_function: callable, str, or None
        :param trim_ends: If True, a portion of the resulting strain is removed
                          at both the initial and final times. The amount removed
                          is equal to pcut.
        :type trim_ends: bool

        :returns: :math:`r (h^+ - i rh^\times)`
        :rtype: :py:class:`~.TimeSeries`

        """
        if (mult_l, mult_m) not in self.available_lm:
            raise ValueError(f"l = {mult_l}, m = {mult_m} not available")

        psi4lm = self[(mult_l, mult_m)]

        # If pcut is too large, the result will likely be inaccurate
        if psi4lm.time_length < 2 * pcut:
            raise ValueError("pcut too large for timeseries")

        if callable(window_function):
            integrand = psi4lm.windowed(window_function, *args, **kwargs)
        elif isinstance(window_function, str):
            window_function_method = f"{window_function}_windowed"
            if not hasattr(psi4lm, window_function_method):
                raise ValueError(f"Window {window_function} not implemented")
            window_function_callable = getattr(psi4lm, window_function_method)

            # This returns a new TimeSeries
            integrand = window_function_callable(*args, **kwargs)
        elif window_function is None:
            integrand = psi4lm
        else:
            raise ValueError("Unknown window function")

        strain = self._fixed_frequency_integrated(integrand, pcut, order=2)

        if trim_ends:
            strain.crop(strain.tmin + pcut, strain.tmax - pcut)

        # The return value is rh not just h (the strain)
        # h_plus - i h_cross
        return strain * self.dist

    def get_strain(
        self,
        theta,
        phi,
        pcut,
        *args,
        window_function=None,
        l_max=None,
        trim_ends=True,
        **kwargs,
    ):
        r"""Return the strain accounting for all the multipoles and the spin
        weighted spherical harmonics.

        This is computed as:

        .. math::

             h_+(r,t)
             -     i h_\times(r,t) = \sum_{l=2}^{l=l_{\mathrm{max}}}
             \sum_{m=-l}^{m=l} h(r, t)^{lm} {}_{-2}Y_{lm}(\theta, \phi)

        :param theta: Meridional observation angle.
        :type theta: float
        :param phi: Azimuthal observation angle.
        :type phi: float
        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float
        :param window_function: If not None, apply window_function to the
                                series before computing the strain.
        :type window_function: callable, str, or None
        :param trim_ends: If True, a portion of the resulting strain is removed
                          at both the initial and final times. The amount removed
                          is equal to ``pcut``.
        :type trim_ends: bool
        :param l_max: Ignore multipoles with ``l > l_max``
        :type l_max: int

        :returns: :math:`r (h^+ - i rh^\times)`
        :rtype: :py:class:`~.TimeSeries`
        """

        # Here we use the BaseClass method total_function_on_available_lm
        # This function loops over all the available (l, m) (with l < l_max)
        # and invokes a function that takes as arguments the timeseries
        # of the multipole component, l, m, r, and potentially others.
        # Then, it accumulates all the results, and return the sum.

        # This is a closure with theta, phi, pcut, and window_function and
        # trim_ends
        def compute_strain(_1, mult_l, mult_m, _2):
            return gw_utils.sYlm(
                -2, mult_l, mult_m, theta, phi
            ) * self.get_strain_lm(
                mult_l,
                mult_m,
                pcut,
                *args,
                window_function=window_function,
                trim_ends=trim_ends,
                **kwargs,
            )

        return self.total_function_on_available_lm(compute_strain, l_max=l_max)

    def get_observed_strain(
        self,
        right_ascension,
        declination,
        time_utc,
        theta_gw,
        phi_gw,
        pcut,
        *args,
        window_function=None,
        polarization=0,
        l_max=None,
        trim_ends=True,
        **kwargs,
    ):
        r"""Return the strain accounting for all the multipoles and the spin
        weighted spherical harmonics as observed by Hanford, Livingston and
        Virgo.

        .. math::

             h_+(r,t)
             -     i h_\times(r,t) = \sum_{l=2}^{l=l_{\mathrm{max}}}
             \sum_{m=-l}^{m=l} h(r, t)^{lm} {}_{-2}Y_{lm}(\theta, \phi)

        Here \theta and \phi are the arguments ``theta_gw`` and ``phi_gw``.

        Then, for each detector

        .. math::

             h(r,t) = F_\times h_\times(theta_gw, phi_gw) + F_+ h_+(theta_gw, phi_gw)

        :param right_ascension: Right ascension of the source in degrees.
        :type right_ascension: float
        :param declination: Declination of the source in degrees.
        :type declination: float
        :param time_utc: UTC time of the event
        :type declination: str
        :param theta_gw, phi_gw: Spherical coordinates of the observer
                                 from the binary's frame, taking the angular
                                 momentum of the binary to
                                 point along the z-axis.
        :type theta_gw, phi_gw: floats
        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float
        :param window_function: If not None, apply ``window_function`` to the
                                series before computing the strain.
        :type window_function: callable, str, or None
        :param trim_ends: If True, a portion of the resulting strain is removed
                          at both the initial and final times. The amount removed
                          is equal to pcut.
        :type trim_ends: bool
        :param l_max: Ignore multipoles with l > l_max
        :type l_max: int

        :returns: :math:`r (h^+ - i rh^\times)`
        :rtype: Detectors of :py:class:`~.TimeSeries`

        """

        # Detectors contains three fields, one for each detector
        antennas = gw_utils.antenna_responses_from_sky_localization(
            right_ascension, declination, time_utc, polarization
        )

        # We collect all the strains in a list, then we convert it in a
        # nameduples Detectors
        strains = []

        # Loop over the detectors in Detectors
        # antennas and coords are namedtuples Detectors
        for (Fc, Fp) in antennas:
            strain = self.get_strain(
                theta_gw,
                phi_gw,
                pcut,
                *args,
                window_function=window_function,
                l_max=l_max,
                trim_ends=trim_ends,
                **kwargs,
            )
            # What have that:
            # strain.real = hp
            # strain.imag = -hc
            strains.append(strain.real() * Fp - strain.imag() * Fc)

        return Detectors(*strains)

    def get_power_lm(self, mult_l, mult_m, pcut):
        r"""Return the instantaneous power in the mode (l, m).

        This is computed with Eq. (9.139) in Baumgarte Shapiro.

        .. math::

        \frac{dE}{\dt}(r, t) = \frac{r^2}{16 \pi}
        \sum_{l=2}^{l=l_{\mathrm{max}}} \sum_{m=-l}^{m=l}
        \psi^{lm}_4(\theta, \phi, r, t)

        :param mult_l: l multipole moment.
        :type mult_t: int
        :param mult_m: m multipole moment.
        :type mult_m: int
        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float

        :returns: Instantaneous power in the mode ``(l, m)`` as a function of
                  time.
        :rtype: :py:class:`~TimeSeries`
        """
        psi4_int = self._fixed_frequency_integrated(
            self[(mult_l, mult_m)], pcut, order=1
        )
        return self.dist ** 2 / (16 * np.pi) * np.abs(psi4_int) ** 2

    def get_energy_lm(self, mult_l, mult_m, pcut):
        """Return the cumulative energy lost in the mode (l, m).

        :param mult_l: l multipole moment.
        :type mult_t: int
        :param mult_m: m multipole moment.
        :type mult_m: int
        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float

        :returns: Energy lost up to the time t in the mode ``(l, m)`` as a function of
                  time.
        :rtype: :py:class:`~TimeSeries`
        """
        return self.get_power_lm(mult_l, mult_m, pcut).integrated()

    def get_total_power(self, pcut, l_max=None):
        """Return the total power in all the modes up to ``l_max``.

        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float
        :param l_max: Ignore multipoles with l > l_max
        :type l_max: int

        :returns: Instantaneous total power in the modes up to  ``l_max``
                  as a function of time.
        :rtype: :py:class:`~TimeSeries`

        """

        def powlm(_1, mult_l, mult_m, _2):
            return self.get_power_lm(mult_l, mult_m, pcut)

        return self.total_function_on_available_lm(powlm, l_max=l_max)

    def get_total_energy(self, pcut, l_max=None):
        """Return the cumulative energy lost in all the modes up to ``l_max``.

        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float
        :param l_max: Ignore multipoles with l > l_max
        :type l_max: int

        :returns: Cumulative total energy lost up to time in the modes up to  ``l_max``
                  as a function of time.
        :rtype: :py:class:`~TimeSeries`

        """
        return self.get_total_power(pcut, l_max).integrated()

    def get_torque_z_lm(self, mult_l, mult_m, pcut):
        """Return the instantaneous torque in the z axis in the mode (l, m).

        This is computed with Eq. (9.140) in Baumgarte Shapiro (or 9.137)

        :param mult_l: l multipole moment.
        :type mult_t: int
        :param mult_m: m multipole moment.
        :type mult_m: int
        :returns: Instantaneous total torque in the z direction in the mode (l, m)
                  as a function of time.
        :rtype: :py:class:`~TimeSeries`

        """
        # This is what we are going to implement
        # The foruma is dJ/dt = r**2/16pi m (dot(A)B - dot(B)*A)
        # where ddot(A) = psi4.real and ddot(B) = -psi4.imag
        # So, A - i B = \int\int psi4, and
        # dot(A) - i dot(B) = \int psi4
        #
        # Considering that:
        # (A - i B) (dot(A) - i dot(B)) =
        # = A dot(A) + B dot(B) - i A dot(B) - i B dot(A)
        #
        # To get the integrand for the angular momentum we can evaluate
        # (A + i B) (dot(A) - i dot(B)) =
        # = A dot(A) - B dot(B) - i A dot(B) + i B dot(A)
        # and take the imaginary part. So,
        # torque = - Im(conj(\int\int psi4) * \int psi4)

        psi4_int1 = self._fixed_frequency_integrated(
            self[(mult_l, mult_m)], pcut
        )
        # We need to integrate twice
        psi4_int2 = self._fixed_frequency_integrated(
            self[(mult_l, mult_m)], pcut, order=2
        )
        return (
            self.dist ** 2
            / (16 * np.pi)
            * mult_m
            * (psi4_int1 * np.conj(psi4_int2)).imag()
        )

    def get_angular_momentum_z_lm(self, mult_l, mult_m, pcut):
        """Return the cumulative angular momentum lost in the mode (l, m).

        :param mult_l: l multipole moment.
        :type mult_t: int
        :param mult_m: l multipole moment.
        :type mult_m: int
        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float

        :returns: Cumulative angular momentum in the z direction in the mode (l, m)
                  as a function of time.
        :rtype: :py:class:`~TimeSeries`

        """
        return self.get_torque_z_lm(mult_l, mult_m, pcut).integrated()

    def get_total_torque_z(self, pcut, l_max=None):
        """Return the total torque z in all the modes up to ``l_max``.

        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float
        :param l_max: Ignore multipoles with l > l_max
        :type l_max: int

        :returns: Instantaneous total torque up to time in the modes up to  ``l_max``
                  as a function of time.
        :rtype: :py:class:`~TimeSeries`
        """

        def torqzlm(_1, mult_l, mult_m, _2):
            return self.get_torque_z_lm(mult_l, mult_m, pcut)

        return self.total_function_on_available_lm(torqzlm, l_max=l_max)

    def get_total_angular_momentum_z(self, pcut, l_max=None):
        """Return the cumulative angular momentum lost in all the modes up to
        ``l_max``.

        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float
        :param l_max: Ignore multipoles with l > l_max
        :type l_max: int

        :returns: Cumulative angular momentum up to time in the modes up to  ``l_max``
                  as a function of time.
        :rtype: :py:class:`~TimeSeries`
        """
        return self.get_total_torque_z(pcut, l_max=l_max).integrated()

    def get_force_z_lm(self, mult_l, mult_m, pcut):
        r"""Return the instantaneous linear momentum along the z direction
        lost in the mode (l, m).

        This is computed with Eq. (3.15) in Ruiz 2008.

        :param mult_l: l multipole moment.
        :type mult_t: int
        :param mult_m: m multipole moment.
        :type mult_m: int
        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float

        :returns: Instantaneous force along the z axis in the mode ``(l, m)``
                  as a function of time.
        :rtype: :py:class:`~TimeSeries`
        """

        # Needed only for Px and Py

        # def a_lm(el, em):
        #     return np.sqrt((el - em) * (el + em + 1)) / (el * (el + 1))

        # def b_lm(el, em):
        #     return 1 / (2 * el) * np.sqrt(((el - 2) * (el + 2) * (el + em) * (el + em - 1))/((2 * el - 1) * (2 * el + 1)))

        def c_lm(el, em):
            num = 2 * em
            den = el * (el + 1)
            return num / den

        def d_lm(el, em):
            num = np.sqrt((el - 2) * (el + 2) * (el - em) * (el + em))
            den = el * np.sqrt((2 * el - 1) * (2 * el + 1))
            return num / den

        # Pz_int1 = \int psi4_lm
        # Pz_int2 = \int [(c_lm * conj(psi4_lm)
        #                + d_lm * conj(psi_(l-1)m)
        #                + d_(l+1)m * conj(psi_(l+1)m))]
        #         = \int conj(A + B + C)
        #
        # (We move the conjugate outside given that all the coefficients
        #  are real)
        Pz_int1 = self._fixed_frequency_integrated(
            self[(mult_l, mult_m)], pcut, order=1
        )

        A = c_lm(mult_l, mult_m) * self[(mult_l, mult_m)]
        if (mult_l - 1, mult_m) in self:
            B = d_lm(mult_l, mult_m) * self[(mult_l - 1, mult_m)]
        else:
            B = ts.TimeSeries(A.t, np.zeros_like(A.y))

        if (mult_l + 1, mult_m) in self:
            C = d_lm(mult_l + 1, mult_m) * self[(mult_l + 1, mult_m)]
        else:
            C = ts.TimeSeries(A.t, np.zeros_like(A.y))

        Pz_int2 = self._fixed_frequency_integrated(
            np.conj(A + B + C), pcut, order=1
        )

        # The imaginary part is zero, so we make the signal real
        return self.dist ** 2 / (16 * np.pi) * (Pz_int1 * Pz_int2).real()

    def get_linear_momentum_z_lm(self, mult_l, mult_m, pcut):
        """Return the cumulative linear momentum lost along the z direction
        in the mode (l, m).

        :param mult_l: l multipole moment.
        :type mult_t: int
        :param mult_m: m multipole moment.
        :type mult_m: int
        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float

        :returns: Linear momentum along the z direction lost up to the time t
                  in the mode ``(l, m)`` as a function of time.
        :rtype: :py:class:`~TimeSeries`
        """
        return self.get_force_z_lm(mult_l, mult_m, pcut).integrated()

    def get_total_force_z(self, pcut, l_max=None):
        """Return the total force along the z direction in all the modes
        up to ``l_max``.

        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float
        :param l_max: Ignore multipoles with l > l_max
        :type l_max: int

        :returns: Instantaneous total force along the z direction in the
                  modes up to  ``l_max`` as a function of time.
        :rtype: :py:class:`~TimeSeries`

        """

        def forclm(_1, mult_l, mult_m, _2):
            return self.get_force_z_lm(mult_l, mult_m, pcut)

        return self.total_function_on_available_lm(forclm, l_max=l_max)

    def get_total_linear_momentum_z(self, pcut, l_max=None):
        """Return the cumulative linear momentum lost in all the modes up to ``l_max``.

         :param pcut: Period that enters the fixed-frequency integration.
                      Typically, the longest physical period in the signal.
         :type pcut: float
         :param l_max: Ignore multipoles with l > l_max
         :type l_max: int

        :returns: Cumulative total linear momentum along the z direction lost
                  up to time in the modes up to  ``l_max`` as a function of time.
         :rtype: :py:class:`~TimeSeries`

        """
        return self.get_total_force_z(pcut, l_max).integrated()


class ElectromagneticWavesOneDet(mp.MultipoleOneDet):
    """Electromagnetic waves computed with the Newman-Penrose approach, using Phi2.

    (These are useful when studying charged black holes, for instance)

    """

    def __init__(self, dist, data):

        super().__init__(dist, data, 1)

    def get_power_lm(self, mult_l, mult_m):
        """Return the instantaneous power in the mode (l, m).

        This is computed with Eq. 2.23 in 1311.6483

        :param mult_l: l multipole moment.
        :type mult_t: int
        :param mult_m: m multipole moment.
        :type mult_m: int
        :returns: Instantaneous power in the mode ``(l, m)`` as a function of
                  time.
        :rtype: :py:class:`~TimeSeries`
        """
        return (
            self.dist ** 2 / (4 * np.pi) * np.abs(self[(mult_l, mult_m)]) ** 2
        )

    def get_energy_lm(self, mult_l, mult_m):
        """Return the cumulative energy lost in the mode (l, m).

        :param mult_l: l multipole moment.
        :type mult_t: int
        :param mult_m: m multipole moment.
        :type mult_m: int

        :returns: Energy lost up to the time t in the mode ``(l, m)`` as a function of
                  time.
        :rtype: :py:class:`~TimeSeries`
        """
        return self.get_power_lm(mult_l, mult_m).integrated()

    def get_total_power(self, l_max=None):
        """Return the total power in all the modes up to ``l_max``.

        :param l_max: Ignore multipoles with l > l_max
        :type l_max: int

        :returns: Instantaneous total power in the modes up to  ``l_max``
                  as a function of time.
        :rtype: :py:class:`~TimeSeries`
        """

        def powlm(_1, mult_l, mult_m, _2):
            return self.get_power_lm(mult_l, mult_m)

        return self.total_function_on_available_lm(powlm, l_max=l_max)

    def get_total_energy(self, l_max=None):
        """Return the cumulative energy lost in all the modes up to l_max.

        :param l_max: Ignore multipoles with l > l_max
        :type l_max: int

        :returns: Cumulative total energy lost up to time in the modes up to  ``l_max``
                  as a function of time.
        :rtype: :py:class:`~TimeSeries`

        """
        return self.get_total_power(l_max=l_max).integrated()

    # Angular momentum is not computed by any public thorns.


class WavesDir(mp.MultipoleAllDets):
    """This class provides acces gravitational-wave data at different radii.

    It is based on :py:class:`~.MultipoleAllDets` with the difference that
    takes as input :py:class:`~.SimDir`. Objects inside
    :py:class:`~.MultipoleAllDets` are redefined as
    :py:class:`~.GravitationalWavesDet`.

    This class is an abstract class meant to be derived to describe gravitational waves
        and electromagnetic waves.

    """

    def __init__(self, sd, l_min, var, derived_type_one_det):
        """Constructor.

        ``derived_type_one_det`` is the type that the values of this
        dictionary-like object has to have.

        :param sd: Simulation directory.
        :type sd: :py:class:`~.SimDir`
        :param l_min: Minimum value of ``l`` to consider.
        :type l_min: int
        :param var: Name of the variable that has be consider
        :type var: str (Psi4 or Phi2)
        :param derived_type_one_det: Class of the derived object that
                                     has to be initialized.
        :type derived_type_one_det: class

        """
        if not isinstance(sd, simdir.SimDir):
            raise TypeError("Input is not SimDir")

        # This module is morally equivalent to mp.MultipoleAllDets because "it
        # is indexed by radius". However, it is the main point of access to GW
        # data, so we keep naming consistent and call it "Dir" and let it have
        # it interface with a SimDir.

        data = []

        # We have to collect data only if var is available
        if var in sd.multipoles:

            psi4_mpalldets = sd.multipoles[var]

            # Now we have to prepare the data for the constructor of the base class
            # The data has format:
            # (multipole_l, multipole_m, extraction_radius, timeseries)
            for radius, det in psi4_mpalldets._dets.items():
                for mult_l, mult_m, tts in det:
                    if mult_l >= l_min:
                        data.append((mult_l, mult_m, radius, tts))

        super().__init__(data)

        # Next step is to change the type of the objects from MultipoleOneDet
        # to GravitationalWaveOneDet (or ElectromagneticWavesDir).
        #
        # To do this, we redefine the objects by instantiating new ones with
        # the same data
        for dist, det in self._dets.items():
            self._dets[dist] = derived_type_one_det(det.dist, det.data)


class GravitationalWavesDir(WavesDir):
    """This class provides access gravitational-wave data at different radii as
     computed from the Psi4 Weyl scalar.

    This is dictionary-like objects with keys the extraction radii and values
    the corresponding :py:class:`~.GravitationalWavesOneDet`.

    """

    def __init__(self, sd):
        super().__init__(sd, 2, "Psi4", GravitationalWavesOneDet)

    @staticmethod
    def _extrapolate_waves_to_infinity(waves, times, radii, mass=1, order=2):
        """Extrapolate ``waves`` to infinity and evaluate the result on the given
        ``times``.

        We follow what described in 1307.5307.

        In practice, ``waves`` is a list of strains at the extraction radii ``radii``.

        :param waves: Waves that have to be extrapolated (strains computed at
                      different radii).
        :type waves: list of :py:class:`~.TimeSeries`
        :param times: Times at which the waves have to be evaluated
        :type times: float or 1D NumPy array
        :param radii: Extraction radii. It has to be that radii[i] correspond to
                      waves[i].
        :type radii: float or 1D NumPy array
        :param mass: ADM mass of the system.
        :type mass: float
        :param order: Order of the extrapolation.
        :type order: int

        :returns: Waves evaluated at the retarded times.
        :rtype: List of :py:class:`~.TimeSeries`

        """

        # Follows what done in the NRAR collaboration (1307.5307)
        # This is what we are going to do:

        # 1. Assume that the spacetime is almost Schwarzschild with mass mass.
        # 2. Choose a set of retarded times u_i where GWs have to be evaluated
        # 3. Compute the coordinate times t_i that correspond to the retarded
        #    times u_i at the radius r. This uses tortoise coordinates.
        # 4. Interpolate the waveforms at the coordinate times corresponding to
        #    the retarded times u_i for different extraction radii.
        #    Now our waves are so that they are evaluated at different t_i but
        #    at the same u_i.
        # 5. We do this process for a bunch of extraction radii
        #    (rex1, rex2, rex3, ...)
        #    So, we should have wave1, wave2, wave3, with all the same number
        #    of points that correspond to the same retarded time.
        # 6. For each element in u_i, fit all the waves (wave1, wave2, ...)
        #    with a polynomial of the form a_n/r^n.

        if order >= len(radii):
            raise RuntimeError(
                "Order too high for the number of extraction radii provided"
            )

        if len(radii) != len(waves):
            raise RuntimeError(
                "Numbers of extraction radii and waves do not agree"
            )
        # Make sure radii is an array
        radii = np.array(radii)

        # First, we resample the waves so that they have all the same retarded
        # times.
        #
        # Take the timeseries w, and return a timeseries evaluated at
        # coordinate times that correspond to the retarded times ui at the
        # coordinate extraction radius rex. mass is the ADM mass (needed to
        # compute tortoise radius).
        waves_retarded = [
            w.resampled(
                gw_utils.retarded_times_to_coordinate_times(times, r, mass)
            )
            for w, r in zip(waves, radii)
        ]

        # We perform the fit in ir=1/r instead of r
        # So, technically, we are fitting sum^p_n=0 a_n * ir^n
        inverse_radii = 1.0 / radii

        # waves_matrix is a table. Each line is a different time, each column
        # is a different extraction radius. We will polyfit on every fixed line
        # to get the extrapolated waves.
        waves_matrix = np.vstack([w.y for w in waves_retarded]).T

        # Polyfit returns coefficient ordered from the highest to the lowest
        # This is why we take the [-1]
        extrapolated_wave = [
            np.polyfit(inverse_radii, waves_at_t, order)[-1]
            for waves_at_t in waves_matrix
        ]

        return ts.TimeSeries(times, extrapolated_wave)

    def extrapolate_strain_lm_to_infinity(
        self,
        mult_l,
        mult_m,
        pcut,
        detectors_distances,
        retarded_times,
        *args,
        window_function=None,
        trim_ends=True,
        mass=1,
        order=2,
        extrapolate_amplitude_phase=False,
        **kwargs,
    ):
        """Extrapolate strains to spatial infinity with the method described in 1307.5307.

        If ``extrapolate_amplitude_phase`` is True, extrapolate amplitude and
        phase of the complex strain instead of real and imaginary parts.

        TODO: Test this function!

        [Equation (29)]

        :param mult_l: Multipolar component l.
        :type mult_l: int
        :param mult_m: Multipolar component m.
        :type mult_m: int
        :param pcut: Period that enters the fixed-frequency integration.
                     Typically, the longest physical period in the signal.
        :type pcut: float
        :param retarded_times: Times at which the waves have to be evaluated
        :type retarded_times: float or 1D NumPy array
        :param detectors_distances: Extraction radii
        :type detectors_distances: 1D NumPy array
        :param mass: ADM mass of the system
        :type mass: float
        :param order: Order of the extrapolation.
        :type order: int
        :param window_function: If not None, apply window_function to the
                                series before computing the strain.
        :type window_function: callable, str, or None
        :param trim_ends: If True, a portion of the resulting strain is removed
                          at both the initial and final times. The amount removed
                          is equal to pcut.
        :type trim_ends: bool
        :param extrapolate_amplitude_phase: If True, extrapolate phase and amplitude, if
                                            False, extrapolate real and imaginary parts.
        :type extrapolate_amplitude_phase: int

        :returns: Waves evaluated at the retarded times.
        :rtype: List of :py:class:`~.TimeSeries`

        """

        dists = np.sort(detectors_distances)

        # Strains are in the form r * h_+ - i r * h_cross
        strains = [
            self[dist].get_strain_lm(
                mult_l,
                mult_m,
                pcut,
                *args,
                window_function=window_function,
                trim_ends=trim_ends,
                **kwargs,
            )
            for dist in dists
        ]

        # Resample the waves to have all the same retarded times
        strains_resampled = [
            strain.resampled(
                gw_utils.retarded_times_to_coordinate_times(
                    retarded_times, dist, mass
                )
            )
            for strain, dist in zip(strains, dists)
        ]

        if not extrapolate_amplitude_phase:
            extrapolated = self._extrapolate_waves_to_infinity(
                strains_resampled,
                retarded_times,
                dists,
                mass,
                order=order,
            )
        else:
            strains_amplitudes = [s.abs() for s in strains_resampled]
            strains_phases = [s.unfolded_phase() for s in strains_resampled]

            extrapolated_amp = self._extrapolate_waves_to_infinity(
                strains_amplitudes,
                retarded_times,
                dists,
                mass,
                order=order,
            )
            extrapolated_phase = self._extrapolate_waves_to_infinity(
                strains_phases,
                retarded_times,
                dists,
                mass,
                order=order,
            )

            extrapolated = extrapolated_amp * np.exp(1j * extrapolated_phase)

        return extrapolated


class ElectromagneticWavesDir(WavesDir):
    """This class provides access electromagnetic-wave data at different radii as
     computed from the Phi2 Weyl scalar.

    This is dictionary-like objects with keys the extraction radii and values
    the corresponding :py:class:`~.ElectromagneticWavesOneDet`.

    """

    def __init__(self, sd):
        super().__init__(sd, 1, "Phi2", ElectromagneticWavesOneDet)
