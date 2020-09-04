#!/usr/bin/env python3

"""The :py:mod:`~.cactus_waves` module provides classes to access gravitational
and electromagnetic wave signals computed using Weyl scalars.

"""


import warnings

from postcactus import simdir
from postcactus import cactus_multipoles as mp


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
        """Return a new timeseries that is the one obtained with the method of
        the fixed frequency integration from the input timeseries.

        :param timeseries: Timeseries that has to be integrated
        :type timeseries: :py:mod:`~TimeSeries`
        :param pcut: Period associated with the threshold frequency
                     ``omega_0 = 2 * pi / pcut``
        :type pcut: float
        :param order:
        :type order: int

        """
        if (not timeseries.is_regularly_sampled()):
            warnings.warning("Timeseries not regularly sampled. Resampling.",
                             RuntimeWarning)
            integrand = timeseries.regularly_sampled()
        else:
            integrand = timeseries

#       regts = ts.regular_sample()
#       t,z   = regts.t, regts.y
#       if (w0 != 0):
#         p     = 2*math.pi/w0
#         eps   = p / (t[-1]-t[0])
#         if (eps>0.3):
#           raise RuntimeError("FFI: waveform too short")
#       else:
#         w0 = 1e-20                  # This practically disable FFI when w0 = 0
#       #
#       if taper:
#         pw = planck_window(eps)
#         z  *= pw(len(z))
#       #
#       dt    = t[1]-t[0]
#       zt    = np.fft.fft(z)
#       w     = np.fft.fftfreq(len(t), d=dt) * (2*math.pi)
#       wa    = np.abs(w)
#       # np.where(wa>w0, wa, w0) means:
#       # return wa when wa > w0, otherwise return w0
#       # This is the FFI integration [arxiv:1006.1632]
#       fac1  = -1j * np.sign(w) / np.where(wa>w0, wa, w0)
#       faco  = fac1**int(order)
#       ztf   = zt * faco
#       zf    = np.fft.ifft(ztf)
#       g     = timeseries.TimeSeries(t, zf)
#       if cut:
#         g.clip(tmin=g.tmin()+p, tmax=g.tmax()-p)
#       #
#       return g


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
            for mult_l, mult_m, ts in det:
                if (mult_l >= l_min):
                    data.append((mult_l, mult_m, radius, ts))

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
            for mult_l, mult_m, ts in det:
                if (mult_l >= l_min):
                    data.append((mult_l, mult_m, radius, ts))

        super().__init__(data)
        for r, det in self._dets.items():
            self._dets[r] = ElectromagneticWavesOneDet(det.dist,
                                                       det.data)
