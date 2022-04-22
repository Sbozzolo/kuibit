Sensitivity (power spectral noise) curves of known detectors
============================================================

The module :py:mod:`~.sensitivity_curves`
(:ref:`sensitivity_curves_ref:Reference on kuibit.sensitivity_curves`)
contains functions that return :py:class:`~.FrequencySeries` with known detector
noise curves on given frequencies.

The routines are named ``Sn_*``, with ``*`` being a detector name (for example,
``Sn_LISA``) and take a ordered numpy array with the frequencies where the noise
has to be evaluated. In some cases, additional parameters can be passed as well.
The frequencies supplied have to be in Hertz, and the output is a
:py:class:`~.FrequencySeries` with the power spectral noise density (in 1/Hz).

The output of the functions in :py:mod:`~.sensitivity_curves` is ready to be
used elsewhere in ``kuibit`` where noises are required.

Available detectors are:
- Advanced LIGO (:py:meth:`~.Sn_aLIGO`)
- Advanced LIGO Plus (:py:meth:`~.Sn_aLIGO_plus`)
- Voyager (:py:meth:`~.Sn_voyager`)
- Cosmic Explorer 1 and 2 (:py:meth:`~.Sn_CE1`, :py:meth:`~.Sn_CE2`)
- Einstein Telescope (:py:meth:`~.Sn_ET_B`)
- KAGRA (:py:meth:`~.Sn_KAGRA_D`)
- LISA (:py:meth:`~.Sn_LISA`)
