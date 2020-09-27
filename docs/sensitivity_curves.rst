Sensitivity (power spectral noise) curves of known detectors
============================================================

The module :py:mod:`~.sensitivity_curves` (:ref:`series_ref:Reference on
postcactus.sensitivity_curves`) contains functions that return
:py:class:`~.FrequencySeries` with known detector noise curves on given
frequencies.

The routines are named ``Sn_*``, with ``*`` being a detector name (for example,
``Sn_LISA``) and take a ordered numpy array with the frequencies where the noise
has to be evaluated. In some cases, additional parameters can be passed as well.
The frequencies supplied have to be in Hertz, and the output is a
:py:class:`~.FrequencySeries` with the power spectral noise density (in 1/Hz).

The output of the functions in :py:mod:`~.sensitivity_curves` is ready to be
used elsewhere in ``PostCactus`` where noises are required.
