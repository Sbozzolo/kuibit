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

antenna_responses
^^^^^^^^^^^^^^^^^

Compute the antenna pattern :math:`F` for Hanford, Livingston, and the Virgo
interferometers for a given source localization (as right ascension and
declination in degrees, and the UTC time). The antenna pattern is used to
compute the strain measured by a detector with the formula. The output of this
:py:meth:`antenna_responses_from_sky_localization` is a ``namedtuple`` with
attributes ``hanford``, ``livingston``, and ``virgo``, each containing a
standard tuple with the responses of that detector for the cross and plus
polarizations.

.. code-block:: python

   # GW150914
   antenna = antenna_responses_from_sky_localization(8, -70, "2015-09-14 09:50:45")
   (Fc_H, Fp_H) = antenna_LIGO.hanford  # or antenna_LIGO['hanford']

   # Alternatively, unpack everything
   ((Fc_H, Fp_H), (Fc_L, Fp_L), (Fc_V, Fp_V)) = antenna

If you are working with a single generic detector, you can use
:py:meth:`antenna_responses` which takes the spherical angles with respect to
a detector on the :math:`z=0` plane and with arms on the two other directions.
