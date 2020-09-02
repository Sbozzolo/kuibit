#!/usr/bin/env python3

"""The :py:mod:`~.cactus_waves` module provides classes to access gravitational
and electromagnetic wave signals computed using Weyl scalars.

"""

from postcactus import cactus_multipoles as mp
from postcactus import simdir


class GravitationalWavesOneDet(mp.MultipoleOneDet):
    """This class represents is an abstract class to represent multipole
    signals from Weyl scalars available at a given distance. To check if
    component is available, use the operator "in". You can iterate over all
    the availble components with a for loop.

    This class is derived from :py:class:`~.MultipoleOneDet`, so it shares most of
    the features, while expanding with methods specific for gravitational waves
    (e.g, to compute the strain).

    This class is not intended to be initialized directly.

    """

    def __init__(self, dist, data):

        super().__init__(dist, data, 2)


class ElectromagneticWavesOneDet(mp.MultipoleOneDet):
    """These are electromagnetic waves computed with the Newman-Penrose approach,
    using Phi2.

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

        # Next step is to change the type of the objects from MultipoleOneDet to
        # GravitationalWaveOneDet.
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
