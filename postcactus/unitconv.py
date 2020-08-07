#!/usr/bin/env python3
"""This module provides a class Units representing unit sytems or unit
conversions.

Units can be used to convert from geometrized units to SI.

For example, assuming a that we are using geometrized units with G = c = M = 1,
where M = 65 M_sun, we can defined the object CU = geom_umass(65 * M_SUN_SI).
CU knows how to convert geometrized quantities to SI: for instance, to convert
the length variable d from geometrized units to SI, just multiply d times
CU.length. Similarly for all the other quantities.

The following natural constants are defined
* C_SI          Vacuum speed of light
* G_SI          Gravitational constant
* M_SOL_SI      Solar mass
* M_SUN_SI      Solar mass
* EV_SI         Electron volt
* MEV_SI        1e6 eV
* UAMU_SI       Unified atomic mass unit
* KB_SI         Boltzmann constant [J/K]
* M_ELECTRON_SI Electron mass
* M_PROTON_SI   Proton mass
* M_NEUTRON_SI  Neutron mass
* LIGHTYEAR_SI  Lightyear
* MEGAPARSEC_SI Parsec
* PARSEC_SI     Parsec
* N_AVOGADRO    Avogadro number [1/mol]
* H_SI          Planck constant
* HBAR_SI       H_SI / (2 pi)
* H0_SI         Hubble constant [1/s]

"""
from scipy import constants


class Units:
    """Class representing unit conversion. The unit system is specified by
length, time, and mass units, from which derived units are computed.

This class can be used to convert units from one system to another.

For example, define CGS = Units(1.0e-2, 1.0, 1.0e-3).
If the length d is in SI, then d * CGS.length will be in CGS.

The main use of Units in PostCactus is to provide a way to convert from
geometrized units to physical units (see, unitconv.geom_umass).
    """

    def __init__(self, ulength, utime, umass):
        """Create a unit system based on length unit ulength, time unit utime,
        and mass unit umass.
        """
        # NOTE: If you add any quantity here, modify the unitconv.rst doc!

        self.length = float(ulength)
        self.time = float(utime)
        self.mass = float(umass)
        self.freq = 1.0 / self.time
        self.velocity = self.length / self.time
        self.accel = self.velocity / self.time
        self.force = self.accel * self.mass
        self.area = self.length * self.length
        self.volume = self.area * self.length
        self.density = self.mass / self.volume
        self.pressure = self.force / self.area
        self.power = self.force * self.velocity
        self.energy = self.force * self.length
        self.energy_density = self.energy / self.volume
        self.angular_moment = self.energy * self.time
        self.moment_inertia = self.mass * self.area

        # TODO: All the electromagnetic quantites are missing. Add them.


# Why do we re-define units here when we have scipy.constants? Two reasons:
# 1. We want to stress that the base units are SI
# 2. scipy.constants does not have astronomycal constants
#    (we could use astropy.constants, but astropy is too big of a dependency)

# NOTE: If you add any constant here, modify the unitconv.rst doc!

# The following constants are all given in SI units
C_SI = constants.speed_of_light          # Speed of light in vacuum
G_SI = constants.gravitational_constant  # Gravitational constant

M_SOL_SI = 1.988_435e30                  # Solar mass
M_SUN_SI = M_SOL_SI                      # Alias for solar mass

PARSEC_SI = constants.parsec             # Parsec
MEGAPARSEC_SI = 1e6 * constants.parsec   # Megaparsec
GIGAPARSEC_SI = 1e9 * constants.parsec   # Gigaparsec

LIGHTYEAR_SI = constants.light_year      # Lightyear

H0_SI = 2.192_711_267e-18                # Hubble constant [1/s]


def geom_ulength(ulength):
    """Create a geometric unit system, expressed in SI, where the length unit
    is given by ulength, expressed in SI units as well.
    """
    return Units(ulength, ulength / C_SI, ulength * (C_SI**2) / G_SI)


def geom_umass(umass):
    """Create a geometric unit system, expressed in SI, where the mass unit
    is given by umass, expressed in SI units as well.
    """
    return geom_ulength(umass * G_SI / (C_SI**2))


def geom_umass_msun(umass):
    """Create a geometric unit system, expressed in SI, where the mass unit
    is given by umass, expressed in SI units as well.
    """
    return geom_umass(umass * M_SOL_SI)
