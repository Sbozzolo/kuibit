Gravitational-wave utilities
==============================

:ref:`gw_utils_ref:Reference on postcactus.gw_utils`


luminosity_distance_to_redshift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute the redshift starting from a given luminosity distance in Megaparsec.
The assumed cosmology is LCDM with no radiation (only matter and dark energy)
and the default values are the ones provided by the Planck mission.

The computation is based on a numerical inversion, which requires an initial
guess. The default one (0.1) should work in most scenarios, but it should be
changed in case of failure.

sYlm
^^^^

Compute the spin-weighted spherical harmonics using recursion relationships. The
angles are defined as :math:`\theta` the meridional angle and :math:`\phi` the
azimulathal one.
