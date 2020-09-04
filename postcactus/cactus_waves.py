#!/usr/bin/env python3

"""The :py:mod:`~.cactus_waves` module provides classes to access gravitational
and electromagnetic wave signals computed using Weyl scalars.

"""


import warnings

import numpy as np

from postcactus import simdir
from postcactus import cactus_multipoles as mp
from postcactus import timeseries as ts


class GravitationalWavesOneDet(mp.MultipoleOneDet):
    """This class represents is an abstract class to represent multipole
    signals from Weyl scalars available at a given distance. To check if
    component is available, use the operator "in". You can iterate over all
    the availble components with a for loop.

    This class is derived from :py:class:`~.MultipoleOneDet`, so it shares most
    of the features, while expanding with methods specific for gravitational
    waves (e.g, to compute the strain).

    This class is not intended to be initialized directly.

    """

    def __init__(self, dist, data):

        super().__init__(dist, data, 2)

    # staticmethod means that this function will be allocated by python only
    # once, since it doesn't depend on the detail of the instance
    @staticmethod
    def _fixed_frequency_integrated(timeseries, pcut, order=1):
        r"""Return a new timeseries that is the one obtained with the method of
        the fixed frequency integration from the input timeseries.

        pcut is the longest physical period in the system (omega_threshold is
        the lowest physical frequency).

        omega is an angular velocity.

        The Fourier transform of f(t) is

        F[f](omega) = \int_-inf^inf e^-i omega t f(t) dt

        The the Fourier transform of the integral of f(t) is
        F[f](omega) / i omega

        In the FFI method we replace this with
        F[f](omega) / i omega               if omega > omega_thereshold
        F[f](omega) / i omega_threshold     otherwise

        (Equation (27) in [arxiv:1006.1632])

        We can perform multiple integrations (needed for example to go from
        psi4 to h) by raising everything to the power of the order of
        integration:
        (due to the convolution theorem)

        F[f](omega) / (i omega)**order             if omega > omega_thereshold
        F[f](omega) / (i omega_threshold)**order   otherwise

        Than, we take the antitransform.

        It is important to window the signal before FFI!
        It is also recommended to cut the boundaries.

        :param timeseries: Timeseries that has to be integrated
        :type timeseries: :py:mod:`~TimeSeries`
        :param pcut: Period associated with the threshold frequency
                     ``omega_0 = 2 * pi / pcut``
        :type pcut: float
        :param order:
        :type order: int

        """
        if (not timeseries.is_regularly_sampled()):
            warnings.warn("Timeseries not regularly sampled. Resampling.",
                          RuntimeWarning)
            integrand = timeseries.regular_resampled()
        else:
            integrand = timeseries

        fft = np.fft.fft(integrand.y)
        omega = np.fft.fftfreq(len(integrand),
                               d=integrand.dt) * (2*np.pi)

        omega_abs = np.abs(omega)
        omega_threshold = 2 * np.pi / pcut

        # np.where(omega_abs > omega_threshold, omega_abs, omega_threshold)
        # means: return omega_abs when omega_abs > omega_threshold, otherwise
        # return omega_threshold
        ffi_omega = np.where(omega_abs > omega_threshold,
                             omega_abs, omega_threshold)

        # np.sign(omega) / (ffi_omega) is omega when omega_abs > omega_thres
        # this is a convient way to group together positive and negative omega
        integration_factor = (np.sign(omega)
                              / (1j * ffi_omega + 1e-100))**int(order)

        # Now, inverse fft
        integrated_y = np.fft.ifft(fft * integration_factor)

        return ts.TimeSeries(integrand.t, integrated_y)

    def get_strain_lm(self, mult_l, mult_m, pcut, *args, window_function=None,
                      trim_ends=True, **kwargs):
        r"""Return the strain associated to the multipolar component (l, m).

        The strain returned is multiplied by the distance.

        The strain is extracted from the Weyl Scalar using the formula

        .. math::

        h_+^{lm}(r,t)
       -     i h_\times^{lm}(r,t) = \int_{-\infty}^t \mathrm{d}u
                    \int_{-\infty}^u \mathrm{d}v\, \Psi_4^{lm}(r,v)

        The return value is complex timeseries (r * h_plus + i r * h_cross).

        It is always important to have a function that goes smoothly to zero
        before taking Fourier transform (to avoid spectral leakage and
        aliasing). You can pass the window function to apply as a paramter.
        If window_function is None, no tapering is performed.
        If window_function is a function, it has to be a function that takes
        as first argument the length of the array and returns a new array
        with the same length that is to be multiplied to the data (this is
        how SciPy's windows work)
        If window_function is a string, use the method with corresponding
        name from the TimeSeries class. You must only provide the name
        (e.g, 'tukey' will call 'tukey_windowed').
        Optional arguments to the window function can be passed directly to
        this function.

        pcut is the period associated to the angular velocity that enters in
        the fixed frequency integration (omega_th = 2 pi / pcut). In general,
        a wise choise is to pick the longest physical period in the signal.

        Optionally, remove part of the output signal at both the beginning and
        the end. If trim_ends is True, pcut is removed. This is because those
        parts of the signal are typically not very accurate.

        :param mult_l: Multipolar component l
        :type mult_l: int
        :param mult_m: Multipolar component m
        :type mult_m: int
        :param pcut: Period that enters the fixed-frequency integration.
        Typically, the longest physical period in the signal.
        :type pcut: float
        :param window_function: If not None, apply window_function to the
        series before computing the strain.
        :type window_function: callable, str, or None
        :param trim_ends: If True, a portion of the resulting strain is removed
        at both the initial and final times. The amount removed is equal to
        pcut.
        :type trim_ends: bool

        :returns: :math:`r (h^+ - i rh^\times)`
        :rtype: :py:class:`~.TimeSeries`

        """
        if ((mult_l, mult_m) not in self.available_lm):
            raise ValueError(f"l = {mult_l}, m = {mult_m} not available")

        psi4lm = self[(mult_l, mult_m)]

        # If pcut is too large, the result will likely be inaccurate
        if (psi4lm.time_length < 2 * pcut):
            raise ValueError("pcut too large for timeseries")

        if (callable(window_function)):
            integrand = psi4lm.windowed(window_function, *args, **kwargs)
        elif (isinstance(window_function, str)):
            window_function_method = f"{window_function}_windowed"
            if (not hasattr(psi4lm, window_function_method)):
                raise ValueError(f"Window {window_function} not implemented")
            window_function_callable = getattr(psi4lm, window_function_method)

            # This returns a new TimeSeries
            integrand = window_function_callable(*args, **kwargs)
        elif (window_function is None):
            integrand = psi4lm
        else:
            raise ValueError("Unknown window function")

        strain = self._fixed_frequency_integrated(integrand, pcut, order=2)

        if (trim_ends):
            strain.crop(strain.tmin + pcut, strain.tmax - pcut)

        # The return value is rh not just h (the strain)
        # h_plus - i h_cross
        return strain * self.dist


class ElectromagneticWavesOneDet(mp.MultipoleOneDet):
    """These are electromagnetic waves computed with the Newman-Penrose
    approach, using Phi2.

    (These are useful when studying charged black holes, for instance)

    """

    def __init__(self, dist, data):

        super().__init__(dist, data, 1)


class GravitationalWavesDir(mp.MultipoleAllDets):
    """This class provides acces gravitational-wave data at different radii.

    It is based on :py:class:`~.MultipoleAllDets` with the difference that
    takes as input :py:class:`~.SimDir`. Objects inside
    :py:class:`~.MultipoleAllDets` are redefined as
    :py:class:`~.GravitationalWavesDet`.
    """

    def __init__(self, sd):
        if (not isinstance(sd, simdir.SimDir)):
            raise TypeError("Input is not SimDir")

        # NOTE: There is significant code duplication between this function and
        #       the init of ElectromagneticWavesDir. However, we allow for this
        #       duplication to avoid introducing a third intermediate class
        l_min = 2

        # This module is morally equivalent to mp.MultipoleAllDets because "it
        # is indexed by radius". However, it is the main point of access to GW
        # data, so we keep naming consistent and call it "Dir" and let it have
        # it interface with a SimDir.
        psi4_mpalldets = sd.multipoles['Psi4']

        # Now we have to prepare the data for the constructor of the base class
        # The data has format:
        # (multipole_l, multipole_m, extraction_radius, timeseries)
        data = []
        for radius, det in psi4_mpalldets._dets.items():
            for mult_l, mult_m, tts in det:
                if (mult_l >= l_min):
                    data.append((mult_l, mult_m, radius, tts))

        super().__init__(data)

        # Next step is to change the type of the objects from MultipoleOneDet
        # to GravitationalWaveOneDet.
        #
        # To do this, we redefine the objects by instantiating new ones with
        # the same data
        for r, det in self._dets.items():
            self._dets[r] = GravitationalWavesOneDet(det.dist,
                                                     det.data)


class ElectromagneticWavesDir(mp.MultipoleAllDets):
    """This class provides acces electromagnetic-wave data at different radii.

    Electromagnetic waves are computed from the Phi2 Weyl scalar.

    """

    def __init__(self, sd):

        if (not isinstance(sd, simdir.SimDir)):
            raise TypeError("Input is not SimDir")

        # NOTE: See comments in init of ElectromagneticWavesDir
        l_min = 1

        psi4_mpalldets = sd.multipoles['Phi2']
        data = []
        for radius, det in psi4_mpalldets._dets.items():
            for mult_l, mult_m, tts in det:
                if (mult_l >= l_min):
                    data.append((mult_l, mult_m, radius, tts))

        super().__init__(data)
        for r, det in self._dets.items():
            self._dets[r] = ElectromagneticWavesOneDet(det.dist,
                                                       det.data)
