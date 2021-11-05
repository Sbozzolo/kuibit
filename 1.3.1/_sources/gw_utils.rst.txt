Gravitational-wave utilities
==============================

The :py:mod:`~.gw_utils` module contains functions that are useful in connecting
numerical relativity simulation with gravitational-waves observations.


:ref:`gw_utils_ref:Reference on kuibit.gw_utils`

Detectors
^^^^^^^^^^^^^^^^^

:py:mod:`~.gw_utils` defines a new data type, ``Detectors`` that can contain
quantities that are specific to the each of operating gravitational-wave
detector (Hanford, Livingston, Virgo). A ``Detectors`` object ``det`` has three
attributes: ``det.hanford``, ``det.livingston``, and ``det.virgo``. You can
access the fields in ``det`` as shown, or you can use the index notation, e.g.,
``det[0]`` (which is det.hanford). The fields are (by convention) in
alphabetical order. Using ``Detectors`` is very convinent because it allows us
to forget about the order of the fields, while guaranteeing that an order exist.

Technically, ``Detectors`` is a `namedtuple
<https://docs.python.org/3/library/collections.html>`_. You can think of it as
an ordered and immutable dictionary.

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
:py:meth:`antenna_responses_from_sky_localization` is a ``Detectors``, a
``namedtuple`` with attributes ``hanford``, ``livingston``, and ``virgo``, each
containing a standard tuple with the responses of that detector for the cross
and plus polarizations.

.. code-block:: python

   # GW150914
   antenna = antenna_responses_from_sky_localization(8, -70, "2015-09-14 09:50:45")
   (Fc_H, Fp_H) = antenna_LIGO.hanford  # or antenna_LIGO['hanford']

   # Alternatively, unpack everything
   ((Fc_H, Fp_H), (Fc_L, Fp_L), (Fc_V, Fp_V)) = antenna

If you are working with a single generic detector, you can use
:py:meth:`antenna_responses` which takes the spherical angles with respect to
a detector on the :math:`z=0` plane and with arms on the two other directions.

signal_to_noise_ratio_from_strain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:meth:`signal_to_noise_ratio_from_strain` takes a strain, a noise curve, and
two boundary frequencies and return the signal-to-noise ratio as

.. :math:

   `\rho^2 = 4 \int_{f_{\mathrm{min}}}^{f_{\mathrm{max}}}\frac{\|\tilde{h}\|^2}{S_n(f)}df`
