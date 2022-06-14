Mismatch between gravitational waves
====================================

``kuibit`` has a module to compute the mismatch between two gravitational
waves (currently, only for the :math:`l=2`, :math:`m=2` mode). See,
:ref:`gw_mismatch_ref:Reference on kuibit.gw_mismatch`, for a comprehensive
reference.

.. warning::

   You have to read carefully and understand everything in this page to use and
   interpret the results from :py:mod:`~.gw_mismatch`. In some cases, misusing
   the functions will lead to incorrect results without raising any error. The
   code has a lot of comments, you are encouraged to read them.

Overlap and mismatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`h_1` and :math:`h_2` gravitational-wave strains, one can compute
the overlap between the two as

.. math::

   \mathrm{overlap}(h_1, h_2) = \frac{(h_1, h_2)}{\sqrt{(h_1, h_1)(h_2, h_2)}}

where

.. math::

   (h_1, h_2) = 4 \Re \int_{f_min}^{f_max} \frac{\tilde{h}_1(f) \tilde{h}_2^*(f)}{S_n(f)} df

is the inner product between :math:`h_1` and :math:`h_2`, the tildas indicate
Fourier transform, and :math:`S_n(f)` is the power spectral density (typically
in units of one over Hertz). In case multiple detectors are considered, the
inner product is the sum of the inner products of each detector (with their own
spectral noise density).

From the overlap, one computes the mismatch between two waves. The mismatch
(also known as unfaithfulness) is the overlap marginalized over some unknown
quantities (more on this later). If :math:`h_1` is an observed signal and
:math:`h_2` a template, the numerical value of mismatch is related to the
signal-to-noise ratio needed to experimentally distinguish :math:`h_1` from
:math:`h_2`.

Typically, the mismatch is defined as

.. math::

   \textrm{mismatch}(h_1, h_2) = \max_{\phi, t, \psi} \textrm{overlap}(h_1, h_2)

where the max is taken over polarization angles, phase and time shifts. Again,
if multiple detectors are considered, the overlap has to be computed with the
network inner product (sum of all the inner products). If only the :math:`l =
2`, :math:`m = 2` mode is considered, phase and polarization shifts are
degenerate, so one can consider only one of the two. In :py:mod:`~.gw_mismatch`,
we implement the mismatch computation for a network of detectors restricting to
the :math:`l = 2`, :math:`m = 2`.

.. note::

   The approach implemented here is probably not the best. One can compute the
   overlap directly form :math:`\Psi_4`, since the fixed frequency integration
   already returns the Fourier transform. This would avoid taking an additional
   Fourier transform, and would avoid all the problems with windowing and
   zero-padding. If I had to write this code again, I would follow this
   approach. Hopefully, the work was not completely useless because the current
   method is more easily generalizable to cases that are not :math:`l = 2`,
   :math:`m = 2`.


network_mismatch_from_psi4
^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:meth:`~.network_mismatch_from_psi4` is the function the most general
interface and the one you will be most likely to use if you are interested in
working with actual LIGO-Virgo data.

Before we start, it is important to stress that computing the mismatch is an
optimization problem. As it is always the case, it is difficult to determine if
the value found is a local maximum or the absolute one. ``PostCacuts``
implements a simple grid search. This is an inefficient but robust method.
However, to work well, it is important to provide reasonable limits. For the
polarization, we search from 0 to :math:`2 \pi`, and we take as input the
extremes of the search for time shifts. You should start providing physical
values, inspect the result, and refine your search. In should also expand the
bounds to make sure that you are localizing the absolute maximum.

.. warning::

   ``kuibit`` has no way to determine if the maximum found is the absolute
   one. It is your job to set the limits of the search in a meaningful way.

To make up for the algorithmic inefficiency, ``kuibit`` optionally uses
`numba <https://numba.pydata.org/>`_ to speed up the search. Using numba enables
high-resolution searches that would not be possible otherwise. Numba compiles
the main mismatch function (:py:meth:`~_mismatch_core_numerical`) to machine
code to achieve native performances. Numba requires a substantial overhead to do
this, so for small searches it is not convenient to use it. Therefore,
``kuibit`` activates numba only when the size of the parameter space is
larger than 2500 elements. If you want to use numba with fewer elements, you can
set ``force_numba`` to ``True``. This may be faster in some cases (for example,
for very long arrays). To use numba, make sure that the package is available (it
is installed among the ``extras``).


The limits of the serach are specified by the paramters ``time_shift_start`` and
``time_shift_end`` for time shifts, and they are always from 0 to :math:`2 \pi`
for polarization shifts. The number of points inspected is ``num_time_shifts``
and ``num_polarization_shifts``.

:py:meth:`~.network_mismatch_from_psi4` takes the two :math:`\Psi_4` and compute
the strains from there. Hence, you have to provide all the quantities you would
need for computing the strains: ``pcut``, ``window_function``, and the arguments
to the window function. :math:`\Psi_4` are passed as
:py:class:`~.GravitationalWavesOneDet`, as they are found in
:py:class:`~.SimDir`, when a radius is specified for gravitational waves.
``window_function`` can be ``None``, a string (indicating one of the buil-in
windows), or a function that implements a custom-window.

Since the operation requires taking Fourier transform, we provide ways to
pre-process the strain signals. First, the window function that you provide to
compute the strain from :math:`\Psi_4` will be used to window also the strain.
Second, the signal is zero padded so that it has a total of ``num_zero_pad``
points. ``num_zero_pad`` is not the number of zeros added: it is the final
length of the signal.

An important quantity you may want to provide is the noise curve associated to
the detectors. For this :py:meth:`~.network_mismatch_from_psi4` takes a
paramters, ``noises``. This can be None, in which case the mismatch will be
computed with no noise. If ``noises`` is not None, then, it has to be a
``Detectors`` object (:ref:`gw_utils:Detectors`) with each entry being a
:py:class:`~.FrequencySeries` with the noise power spectral densities. At the
moment, ``Detectors`` are set to work with the LIGO and Virgo interferometers.
In case you want to disable one of the detectors, set the entry to -1 (see
example below). You can also set one entry to ``None`` so that its contribution
is computed without noise.

In case you want to remove part of the signal from the comparison, you can use
the two paramters ``time_to_keep_after_max`` and ``time_removed_beginning``. The
first sets how much signal to keep after the peak, everything else after that is
removed. The second controls how much signal to remove at the very beginning.
They are always provided in computational units. You may need to set
``trim_ends=False`` if you want to have finer control on how much signal to
consider. For a meaningful comparison, it is important that the time limits are
set properly, if they are not, the window function may produce incorrect results
(because the two series are windowed in physically different ways). Visualize
your data to make sure that the comparison is meaningful!

Typically, we perform simulations in some geometrized units, but we want to
compare signals using actual noise (in physical units). For this, you can
provide the mass scales in solar masses of the two signals. The waves are
assumed to be in geometrized units in which ``M=1``. If you provide the mass
scales, they are converted in waveform with ``M=mass_scale_msun * M_sun``.
Additionally, if you provide a mass scale, you can provide a distance in
megaparsec. The signal will be redshifted according to the cosmological redshift
corresponding to that distance (assuming standard LCDM). Moreover, you have to
provide the sky localization of the event with the paramters
``right_ascension``, ``declination``, and ``time_utc``. In case you want to work
with ``theta`` and ``phi``, you should use the
:py:meth:`~.mismatch_from_strains` function.

A (roughly) complete example would look like:

.. code-block:: python

    mass_scale = 65
    CU = unitconv.geom_umass_msun(mass_scale)

    pcut1 = 120
    pcut2 = 140

    fmin = 20
    fmax = 512

    rex = 110  # Extraction radius

    psi1 = simdir.SimDir("folder1").gws[rex]
    psi2 = simdir.SimDir("folder2").gws[rex]

    distance = 500  # Mpc

    noise_hanford = load_FrequencySeries("ligo.dat", usecols=(0, 1))
    noise_livingston = load_FrequencySeries("ligo.dat", usecols=(0, 2))

    # -1 disables Virgo
    noises = Detectors(virgo=-1,
                       hanford=noise_hanford,
                       livingston=noise_livingston)

   return gw_mismatch.network_mismatch_from_psi4(psi1,
                                                 psi2,
                                                 8,
                                                 -70,
                                                 "2015-09-14 09:50:45",
                                                 pcut1,
                                                 pcut2,
                                                 0.125,  # tukey alpha
                                                 noises=noises,
                                                 trim_ends=False,
                                                 window_function='tukey',
                                                 mass_scale1_msun=mass_scale,
                                                 mass_scale2_msun=mass_scale,
                                                 distance1=distance,
                                                 distance2=distance,
                                                 fmin=fmin,
                                                 fmax=fmax,
                                                 num_time_shifts=1000,
                                                 num_zero_pad=2**18,
                                                 num_polarization_shifts=1000,
                                                 time_shift_start=-10 * CU.time,
                                                 time_shift_end=10 * CU.time,
                                                 time_to_keep_after_max=400,
                                                 time_removed_beginning=200)


In case you want to compute the optimal mismatch considering only one detector,
you can use the function :py:meth:`~.one_detector_mismatch_from_psi4`, which is
similar to :py:meth:`~.network_mismatch_from_psi4` but considers only one
detector.

mismatch_from_strains
^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:meth:`~.mismatch_from_strains` implements a more low-level interface to
compute the mismatch between the waveforms. Internally, this is what is used by
:py:meth:`~.network_mismatch_from_psi4`.

With :py:meth:`~.mismatch_from_strains` you are responsible of providing valid
strain data ``h1`` and ``h2``, as well as ``antenna_patterns`` and ``noises``.
Here, ``antenna_patterns`` and ``noises`` are lists where the corresponding
index represents the same detector.

If you want to learn how the mismatch computation works, read the comments in
the code of this function.
