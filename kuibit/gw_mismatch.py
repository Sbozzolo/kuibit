#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

"""The :py:mod:`~.gw_mismatch` module functions to compute the mismatch between
two waves using a simple grid search for phase and time shifts. Since no
polarization shifts are performed, this is only relevant to the gravitational
wave 2,2 mode.

The two main interfaces are :py:func:`~.network_mismatch_from_psi4` (when
computing the network mismatch starting from psi4 and the sky localization) and
:py:func:`~.mismatch_from_strains` (when computing the mismatch from the
strains).

    ..warning::

        Make sure to understand what is going on if you are using this module.
        You should read the code and comments in the code.

"""
from contextlib import contextmanager
from warnings import warn

import numpy as np

# What is this? This is numba!
#
# numba is a JITter (JIT = Just In Time). The following code is
# compiled at runtime. The compiled code is instead, and it is much
# faster.
#
# At the moment, fft is not supported, so the full power of numba
# cannot be achieved.

# TODO (FUTURE): Update when numba supports FFTs

# We have to put this here. See numba issue #4456
# We always try to import numba because we need objtmode
try:
    from numba import njit
    from numba import objmode as numba_objmode
except ImportError:  # pragma: no cover
    pass

from kuibit import frequencyseries as fs
from kuibit import gw_utils as gwu
from kuibit import unitconv


def _mismatch_core_numerical(
    h1_c_fft,
    h1_p_fft,
    h2_t,
    delta_t,
    frequencies,
    frequency_mask,
    noises,
    antenna_patterns,
    polarization_shifts,
    time_shifts,
):
    """Compute the maximum overlap between ``h1_fft`` (frequency domain, cross and
    plus) and ``h2_t`` (time domain). This function requires very specific
    pre-processing and should never be used directly. All the details are in the
    comments.

    This can be optinally "numba-ified" to increase the speed (if numba is
    available). The input and output values and standard NumPy objects.

    :param h1_c_fft: Fourier transform of the cross polarization of the first
                     strain (it will not be modified). It has to be defined
                     only over the frequencies of interest.
    :type h1_c_fft: 1D complex NumPy array
    :param h1_p_fft: Fourier transform of the plus polarization of the first
                     strain (it will not be modified). It has to be defined
                     only over the frequencies of interest.
    :type h1_p_fft: 1D complex NumPy array
    :param h2_t: Timeseries of the second strain. It will be modified with time
    and polarization shifts. It has to be pre-processed so that is defined over
    the same times as h1_t :type h2_t: 1D complex NumPy array.
    :type h2_t: 1D complex NumPy array
    :param delta_t: Timestep.
    :type delta_t: float
    :param frequencies: Frequencies where we want to compute the integral (ie,
    from fmin to fmax).
    :type frequencies: 1d NumPy array

    :param frequency_mask: What frequencies we should keep from the unfiltered
    ones (we start from 0 to 1/dt, which ones are in frequencies). Technically
    we can compute this in this function, but it is easier and faster to just
    provide it.
    :type frequency_mask: 1d NumPy array of bools

    :param noises: Power spectral density of the noise, defined on the correct
    frequencies.
    :type noises: tuple of 1d NumPy arrays

    :param antenna_patterns: Fc, Fp for all the detectors. It has to be ordered
    in the same way as ``noises``.
    :type antenna_patterns: tuple of tuples

    :param polarization_shifts: Polarization shifts that will be applied in the
    search for the maximum.
    :type polarization_shifts: 1d NumPy array

    :param time_shifts: Time shifts that will be applied in the search for
    the maximum.
    :type time_shifts: 1d NumPy array

    """

    # Here we are going to compute the overlaps (more or less). Since all the
    # series are evenly spaced in the same frequency range, we can forget about
    # the measure of integration (it simplifies in the formula for the
    # overlap). Similarly, we can also drop the 4 in the inner product because
    # it simplifies. So, we just need to integrate h * h^*

    # 7. Prepare the array of the overlaps which we have to maximise

    overlaps = np.zeros((len(polarization_shifts), len(time_shifts)))

    # Convenience alias (will use later)
    omega = 2j * np.pi * frequencies

    # 8. Now we fill the array

    for index_p, p_shift in enumerate(polarization_shifts):
        # 9. Apply polarization shift

        h2_t_pshifted = h2_t * np.exp(1j * np.pi * p_shift)

        # 10. Split in plus and cross polarizations
        h2_p_t_pshifted = h2_t_pshifted.real
        h2_c_t_pshifted = -h2_t_pshifted.imag

        # 11. Now we have to Fourier transform and make sure that it matches
        #     the frequency range of the other arrays.

        # Numba does not support fft yet, so we have to go to object mode
        # This context manager does not affect anything when we run without
        # numba
        with objmode(  # skipcq PYL-E0602  # noqa F821
            h2_p_fft_pshifted="complex128[:]",
            h2_c_fft_pshifted="complex128[:]",
        ):
            h2_p_fft_pshifted = np.fft.fft(h2_p_t_pshifted)
            # We work with shifted frequencies (and ffts)
            h2_p_fft_pshifted = np.fft.fftshift(h2_p_fft_pshifted)

            h2_c_fft_pshifted = np.fft.fft(h2_c_t_pshifted)
            # We work with shifted frequencies (and ffts)
            h2_c_fft_pshifted = np.fft.fftshift(h2_c_fft_pshifted)

        # Remove negative frequencies, and those outside the range (fmin, fmax)
        h2_p_fft_pshifted = h2_p_fft_pshifted[frequency_mask]
        h2_c_fft_pshifted = h2_c_fft_pshifted[frequency_mask]

        # Normalize
        h2_p_fft_pshifted *= delta_t
        h2_c_fft_pshifted *= delta_t

        for index_t, t_shift in enumerate(time_shifts):
            # 12. We implement time shift is implemented as phase shift in
            #     Fourier space. We do this while computing the integral of
            #     the the inner product (we will normalize later).
            #
            #     (We also have to include the antenna factors and the noise).

            inner_product = np.zeros_like(h1_p_fft)

            for noise, antenna_pattern in zip(noises, antenna_patterns):
                Fc, Fp = antenna_pattern

                numerator = Fp * h1_p_fft + Fc * h1_c_fft
                numerator *= (
                    Fp * h2_p_fft_pshifted + Fc * h2_c_fft_pshifted
                ).conj()

                inner_product += numerator * np.exp(omega * t_shift) / noise

            overlaps[index_p][index_t] = np.sum(inner_product).real

    overlaps = np.abs(overlaps)

    return (np.amax(overlaps), overlaps.argmax())


def mismatch_from_strains(
    h1,
    h2,
    fmin=0,
    fmax=np.inf,
    noises=None,
    antenna_patterns=None,
    num_polarization_shifts=100,
    num_time_shifts=100,
    time_shift_start=-5,
    time_shift_end=5,
    force_numba=False,
):
    r"""Compute the network-mismatch between ``h1`` and ``h2`` by maximizing the
    overlap over time and polarization shifts.

    Network here means that the inner product is computed for N detectors, as
    provided by the lists antenna_patterns and noises. Noises and antenna
    patterns have to be properly ordered: ``noises[i]`` has to correspond to
    ``antenna_pattern[i]``.

    See :ref:`gw_mismatch:Overlap and mismatch` for formulas and details.

    The mismatch is computed by maximizing over time and polarization shifts.
    Polarization shifts and are around the 2pi, time shifts are specified by
    time_shift_start and time_shift_end. If num_time_shifts is 1, then no time
    shift is performed. For times, we make sure that we always have to zero
    timeshift. All the transformations are done in h2.

    This computation is a maximisation, which is very expensive. So, we have a
    very fast core function called _mismatch_core_numerical to do all the hard
    work. This function is compiled to native code by numba, resulting to
    enormous speed-up.There is an overhead in calling numba. Hence, by default
    we do not always use numba. We use it only when the number
    num_polarization_shifts * num_time_shifts is greater than 500*500. You can
    force using numba passing the keyword argument force_numba=True.

    We do not perform phase shifts here, so this function makes sense only
    for the (2,2) mode.

    h1 and h2 have to be already pre-processed for Fourier transform, so you
    should window them and zero pad as needed.

    :param h1: First strain.
    :type h1: :py:class:`~.TimeSeries`
    :param h2: Second strain (the one that will be modified).
    :type h2: :py:class:`~.TimeSeries`
    :param fmin: Lower limit of the integration.
    :type fmin: float
    :param fmax: Higher limit of the integration.
    :type fmax: float
    :param noises: Power spectral density of the noise for all the detectors.
                   If None, a uniform noise is applied.
    :type noises: list of :py:class:`~.FrequencySeries`, or None
    :param antenna_patterns: Fc, Fp for all the detectors. It has to be ordered
                             in the same way as noises. If None, a uniform antenna
                             pattern is applied.
    :type antenna_patterns: list of tuples, or None
    :param num_polarization_shifts: How many points to divide the range
                                    (0, 2 pi) in the polarization shift.
    :type num_polarization_shifts: int
    :param num_time_shifts: How many points to divide the range
                            (time_shift_start, time_shift_end) in the time shift.
    :type num_time_shifts: int
    :param time_shift_start: Minimum time shift applied. Search will be done
                             linearly up to time_shift_end.
    :type time_shift_start: float
    :param time_shift_end: Largest value of time shift applied.
    :type time_shift_end: float
    :param force_numba: Use numba irrespectively of the size of the input.
    :type force_numba: bool

    """

    # In kuibit, we have beautiful collection of classes to represent
    # different data types (TimeSeries, FrequencySeries, ...).
    # However, from great abstraction comes great performance penalities.
    # Using these classes is too slow for expensive operations.
    # The reason for this are (at least):
    # 1. large number of function calls (expensive in Python)
    # 2. several redundant operations
    # 3. several checks that we can guarantee will be passed
    # ...
    # Computing the mismatch is a numerical operation, we should be able
    # to crunch numbers at the speed of light (ie, as fast as C). For this,
    # we use numba and we break apart all our abstractions to expose only
    # the data as NumPy arrays. In this function we pre-process the
    # FrequencySeries so that we can feed _mismatch_core_numerical with
    # what we need. _mismatch_core_numerical takes only standard NumPy
    # objects (arrays, tuples, and floats) and return the mismatch and
    # the phase/time shifts needed for it.
    #
    # An important step will be to guarantee that everything (the series and
    # the noise) is defined over the same frequency range.
    #
    # What we are doing is:
    # 1. Prepare the arrays for the shifts that have to be performed

    polarization_shifts = np.linspace(0, 2 * np.pi, num_polarization_shifts)

    # We make sure that we always have to zero timeshift.
    time_shifts = np.append(
        [0],
        np.linspace(time_shift_start, time_shift_end, num_time_shifts - 1),
    )

    # 2. We resample h1 and h2 to a common timeseries (linearly spaced). This
    #    guarantees that their Fourier transform will be defined over the same
    #    frequencies. To avoid throwing away signal, we resample the two series
    #    to the union of their times, setting them to zero where they were not
    #    defined, and choosing as number of points the smallest number of
    #    points between the two series.

    (smallest_len, largest_tmax, smallest_tmin) = (
        min(len(h1), len(h2)),
        max(h1.tmax, h2.tmax),
        min(h1.tmin, h2.tmin),
    )

    union_times = np.linspace(smallest_tmin, largest_tmax, smallest_len)

    # ext=1 sets zero where the series is not defined
    h1_res, h2_res = (
        h1.resampled(union_times, ext=1),
        h2.resampled(union_times, ext=1),
    )

    # 3. We take the Fourier transform of the two polarizations. In doing this,
    #    we also make sure that the arrays are complex.This is because we will
    #    be doing complex operations (e.g. phase-shift), so, we will always
    #    deal with complex series. However, enforcing that they are complex
    #    since the beginning makes bookeeping easier for the Fourier transform,
    #    as the operation behaves differently for real and imaginary data.
    #
    #    We crop h1_res to the requested frequencies and we only take the
    #    positive ones. We will resample the noise to match h1_res.

    h1_p_res = h1_res.real()
    h1_c_res = -h1_res.imag()

    h1_p_res.y = h1_p_res.y.astype("complex128")
    h1_c_res.y = h1_c_res.y.astype("complex128")

    h1f_p_res = h1_p_res.to_FrequencySeries()
    h1f_p_res.band_pass(fmin, fmax)
    h1f_p_res.negative_frequencies_remove()

    h1f_c_res = h1_c_res.to_FrequencySeries()
    h1f_c_res.band_pass(fmin, fmax)
    h1f_c_res.negative_frequencies_remove()

    # 3. Then, we resample the noise to have be defined on the same frequencies
    #    as h1f. We will only need to take care of the h2. If the noise is
    #    None, we prepare a unweighted noise (ones everywhere).
    #
    #    The problem with resampling noises is that PSD curves have often
    #    strong discontinuities, which are not correctly captured by the
    #    splines. Therefore, instead of using cubic splines, here we prefer
    #    using a piecewise constant approximation. Since the noise has
    #    typically a lot of points, this should be a better approximation than
    #    having large jumps. kuibit does not have this option, so we use
    #    directly SciPy's interp1d.

    if noises is not None:
        # With this, we can guarantee that everything has the same domain.
        # If there's a None entry, we fill it with a constant noise.
        noises_res = []
        for noise in noises:
            noises_res.append(
                fs.FrequencySeries(h1f_p_res.f, np.ones_like(h1f_p_res.fft))
            )
            if noise is not None:
                # TODO: Now the Series class has a function for this kind of
                #       resampling. Use that.
                #
                # We start with a FrequencySeries of ones, and we overwrite the
                # fft attribute
                noises_res[-1] = noises[-1].resampled(
                    h1f_p_res.f, piecewise_constant=True
                )
    else:
        # Here we prepare a noise that is made by ones everywhere. This is what
        # happens internally when noises is None. However, here we do it
        # explicitly because we are going to pass it to the numba function.
        noises_res = [
            fs.FrequencySeries(h1f_p_res.f, np.ones_like(h1f_p_res.fft))
        ]

    # 4. We use the linearity of the Fourier transform to apply the antenna
    #    pattern. (This is why we have to carry around the two polarization
    #    seperatebly). We have to compute tilde(h_1) * tilde(h_2).conj().
    #    But h_i = Fp h_p + Fc h_c. So, for linearity
    #    tilde(h_1) = Fp tilde(h_p) + Fc tilde(h_c). Similarly with h_2.
    #    Therefore, we have to prepare the antenna patterns for each detector.

    # This case is "we have 3 noise curves, but we don't care about the antenna
    # response". So we have to have 3 antenna patterns.
    if antenna_patterns is None:
        antenna_patterns = [(1 / 2, 1 / 2)] * len(noises_res)

    # This case is "we have N detectors, but we don't care about the actual
    # noise curve". So we have to have N noises. Before, we set noises =
    # [ones], so we duplicate that.
    #
    # If both noises and antenna_patterns are None, we will have a single
    # element in the noises list, which is what we expect.
    if noises is None:
        noises_res *= len(antenna_patterns)

    # Numba doesn't support lists, so we generate a tuple of arrays
    antenna_patterns = tuple(antenna_patterns)
    noises = tuple(n.fft for n in noises_res)

    # 5. Now, we have to prepare a frequency mask. This is an array of bools
    #    that indicates which frequencies in h2 should be used. This is because
    #    we are taking the Fourier transform in _mismatch_core_numerical, but
    #    we need to make sure that we considering only positive frequencies
    #    from fmin to fmax.

    all_frequencies = np.fft.fftfreq(len(h2_res.t), d=h2_res.dt)
    shifted_frequencies = np.fft.fftshift(all_frequencies)

    frequency_mask = np.array([f in h1f_p_res.f for f in shifted_frequencies])

    # 6. Finally we can call the numerical routine which will return the
    #    un-normalized mismatch and the shifts required. We will Fourier
    #    transform h2 in there. We must do that because we have to perform
    #    the polarization shifts in the time domain.

    frequencies = h1f_p_res.f  # from fmin to fmax

    use_numba = (
        force_numba or num_polarization_shifts * num_time_shifts >= 500 * 500
    )

    if use_numba and "njit" not in globals():
        if force_numba:
            warn("numba not available, ignoring force_numba")
        use_numba = False

    if use_numba:
        globals()["objmode"] = numba_objmode
        _core_function = njit(_mismatch_core_numerical)
    else:
        # HACK: Now we have to do something dirty. _mismatch_core_numerical
        #       calls numba.objmode to perform FFTs, but when numba is not
        #       available, objmode is unkown. Hence, we have to provide a dummy
        #       objmode that does nothing. As long as numba doesn't support
        #       FFTs natively, that code has to be here. However, cannot put in
        #       _mismatch_core_numerical because numba wouldn't be able to
        #       compile the function.
        @contextmanager
        def nullcontext(*args, **kwargs):
            yield None

        # We override objmode in the gobal scope with nullcontext
        globals()["objmode"] = nullcontext

        _core_function = _mismatch_core_numerical

    (unnormalized_max_overlap, index_max) = _core_function(
        h1f_c_res.fft,
        h1f_p_res.fft,
        h2_res.y,
        h2_res.dt,
        frequencies,
        frequency_mask,
        noises,
        antenna_patterns,
        polarization_shifts,
        time_shifts,
    )

    # 12. The normalization is constant. Again, we do not include df or the
    #     factor of 4.

    h2_p_res = h2_res.real()
    h2_c_res = -h2_res.imag()

    # Transform to complex
    h2_p_res.y = h2_p_res.y.astype("complex128")
    h2_c_res.y = h2_c_res.y.astype("complex128")

    h2f_p_res = h2_p_res.to_FrequencySeries()
    h2f_p_res.band_pass(fmin, fmax)
    h2f_p_res.negative_frequencies_remove()

    h2f_c_res = h2_c_res.to_FrequencySeries()
    h2f_c_res.band_pass(fmin, fmax)
    h2f_c_res.negative_frequencies_remove()

    inner11 = fs.FrequencySeries(h1f_p_res.f, np.zeros_like(h1f_p_res.f))
    inner22 = fs.FrequencySeries(h2f_p_res.f, np.zeros_like(h2f_p_res.f))

    for noise, antenna_pattern in zip(noises_res, antenna_patterns):
        Fc, Fp = antenna_pattern

        numerator11 = Fp * h1f_p_res + Fc * h1f_c_res
        numerator11 *= (Fp * h1f_p_res + Fc * h1f_c_res).conjugate()

        inner11 += numerator11 / noise

        numerator22 = Fp * h2f_p_res + Fc * h2f_c_res
        numerator22 *= (Fp * h2f_p_res + Fc * h2f_c_res).conjugate()

        inner22 += numerator22 / noise

    inner11 = np.sum(inner11.fft).real
    inner22 = np.sum(inner22.fft).real

    norm = np.sqrt(inner11 * inner22)

    # Values that maximise the overlap

    # pylint: disable=unbalanced-tuple-unpacking
    (p_index, t_index) = np.unravel_index(
        index_max, (num_polarization_shifts, num_time_shifts)
    )

    # Check t_index is close to the boundary and emit warning
    # We have to check for t_index = 0 because we always put the tshift=0 there
    if (not 0.05 < t_index / num_time_shifts < 0.95) and t_index != 0:
        warn("Maximum of overlap near the boundary of the time shift interval")

    p_shift_max = polarization_shifts[p_index]
    t_shift_max = time_shifts[t_index]

    return 1 - unnormalized_max_overlap / norm, (
        p_shift_max,
        t_shift_max,
    )


def network_mismatch(
    h1,
    h2,
    right_ascension,
    declination,
    time_utc,
    fmin=0,
    fmax=np.inf,
    noises=None,
    num_polarization_shifts=200,
    num_time_shifts=1000,
    time_shift_start=-20,
    time_shift_end=20,
    force_numba=False,
):
    """Compute network mismatch between strains h1 and h2.

    This is a wrapper around :py:meth:`~.mismatch_from_strains` (read the
    docstring for more information) that prepares the correct antenna patterns
    from the sky localization of the event. Moreover, it makes sure that noises
    (that has to be a gw_utils.Detectors object, or None) is correctly ordered.

    :param h1: First strain.
    :type h1: :py:class:`~.TimeSeries`
    :param h2: Second strain (the one that will be modified).
    :type h2: :py:class:`~.TimeSeries`
    :param right_ascension: Right ascension of the source in the sky.
    :type right_ascension: float
    :param declination: Declination of the source in the sky.
    :type declination: float
    :param time_utc: Time UTC of the event.
    :type time_utc: float
    :param fmin: Lower limit of the integration.
    :type fmin: float
    :param fmax: Higher limit of the integration.
    :type fmax: float
    :param noises: Power spectral density of the noise for all the detectors.
    :type noises: :py:class:`~.Detector`, or None
    :param num_polarization_shifts: How many points to divide the range
                                    (0, 2 pi) in the polarization shift.
    :type num_polarization_shifts: int
    :param num_time_shifts: How many points to divide the range
                            (time_shift_start, time_shift_end) in the
                            time shift.
    :type num_time_shifts: int
    :param time_shift_start: Minimum time shift applied. Search will be done
                             linearly up to time_shift_end.
    :type time_shift_start: float
    :param time_shift_end: Largest value of time shift applied.
    :type time_shift_end: float
    :param force_numba: Use numba irrespectively of the size of the input.
    :type force_numba: bool

    """

    antenna_patterns = gwu.antenna_responses_from_sky_localization(
        right_ascension, declination, time_utc
    )

    # Transform Detectors to lists (they are already properly ordered)
    if noises is not None:
        if isinstance(noises, gwu.Detectors):
            # We select thos antennas for which the corresponding noise is not
            # -1
            antenna_patterns = [
                ap
                for ap, noise in zip(antenna_patterns, noises)
                if noise != -1
            ]
            # We remove all the noises that are -1. This modifies the list.
            noises = [noise for noise in noises if noise != -1]
        else:
            raise TypeError("noises has to be None or of type Detectors")
    else:
        # All three detectors
        antenna_patterns = list(antenna_patterns)

    return mismatch_from_strains(
        h1,
        h2,
        fmin,
        fmax,
        noises=noises,
        antenna_patterns=antenna_patterns,
        num_polarization_shifts=num_polarization_shifts,
        num_time_shifts=num_time_shifts,
        time_shift_start=time_shift_start,
        time_shift_end=time_shift_end,
        force_numba=force_numba,
    )


def _strains_from_psi4(
    psi1,
    psi2,
    pcut1,
    pcut2,
    *args,
    window_function=None,
    align_at_peak=True,
    trim_ends=False,
    mass_scale1_msun=None,
    mass_scale2_msun=None,
    distance1=None,
    distance2=None,
    num_zero_pad=16384,
    time_to_keep_after_max=None,
    time_removed_beginning=None,
    **kwargs
):
    r"""Extract the strains from (2, 2) mode of psi1 and psi2 (which have to be
    :py:class:`~.GravitationalWavesOneDet`, as from :py:class:`~.SimDir` when
    a radius is specified). In doing this, use the fixed-frequency integration
    method with pcut1 and pcut2 and with the provided window function.

    Then, possibly align the signal to peak, transform it to physical units,
    and crop it as requested. Send this to the :py:meth:`~.network_mismatch`
    function.

    :param psi1: :math:`\Psi_4` for the first wave.
    :type psi1: :py:class:`~.GravitationalWavesOneDet`
    :param psi2: :math:`\Psi_4` for the second wave (the one that will be
                 modified).
    :type psi2: :py:class:`~.GravitationalWavesOneDet`

    :param mass_scale1_msun: If not None, the signal h1 is converted from
    computational units to physical units assuming that
    M = mass_scale1_msun.
    :type mass_scale1_msun: float or None

    :param mass_scale2_msun: If not None, the signal h2 is converted from
    computational units to physical units assuming that
    M = mass_scale2_msun.
    :type mass_scale2_msun: float or None

    :param pcut1: Period associated with the threshold frequency
                 ``omega_0 = 2 * pi / pcut`` for the fixed frequency
                 integration of :math:`\Psi_4`
    :type pcut1: float

    :param pcut2: Period associated with the threshold frequency
                 ``omega_0 = 2 * pi / pcut`` for the fixed frequency
                 integration of :math:`\Psi_4`
    :type pcut2: float

    :param window_function: If not None, apply window_function to the
    series before computing the strain.
    :type window_function: callable, str, or None

    :param trim_ends: If True, a portion of the resulting strain is removed
    at both the initial and final times. The amount removed is equal to
    pcut.
    :type trim_ends: bool

    :param align_at_peak: Time-shifts the strain so that they have both the
    maximum amplitude at t=0.
    :type align_at_peak: bool

    :param num_zero_pad: How many points do the timeseries have to have
    before Fourier transforms are taken? This is not the number of zeros
    added (at the end) of the timeseries, this is the total number. If the
    series already have that length, no zeros will be added.
    :type num_zero_pad: int

    :param time_removed_beginning: Remove this amount from the beginning
    of the strain signals before computing the mismatch. If None, nothing
    is removed. This is in computational units regardless of the value of
    mass_scale.
    :type time_removed_beginning: float or None

    :param time_to_keep_after_max: If not None, remove all the signal that comes
    after t_max + time_to_keep_after_max, where t_max is the time at which the
    signal peaks. This is in computational units regardless of the value of
    mass_scale. :type time_to_keep_after_max: float or None

    :param *args: All the other arguments are passed to the window
     function.
    :type *args: anything

    """

    # First, we compute the strains. If we just look at the (2,2) mode, we
    # don't need to multiply the spin weighted spherical harmonics, since it is
    # a normalization factor.
    h1 = psi1.get_strain_lm(
        2,
        2,
        pcut1,
        *args,
        window_function=window_function,
        trim_ends=trim_ends,
        **kwargs
    )

    h2 = psi2.get_strain_lm(
        2,
        2,
        pcut2,
        *args,
        window_function=window_function,
        trim_ends=trim_ends,
        **kwargs
    )

    # Align the waves at the peak
    if align_at_peak:
        h1.align_at_maximum()
        h2.align_at_maximum()

    # Now, we convert to physical units
    if mass_scale1_msun is not None:
        CU1 = unitconv.geom_umass_msun(mass_scale1_msun)
        h1.time_unit_change(CU1.time, inverse=True)
        if distance1 is not None:
            distance1_SI = distance1 * unitconv.MEGAPARSEC_SI
            # Remember, h is actually r * h!
            #
            # We will work with rh (in physical units). This does not change the
            # result because r is just a constant.
            h1 *= CU1.length / distance1_SI
            redshift1 = gwu.luminosity_distance_to_redshift(distance1)
            h1.redshift(redshift1)

    if mass_scale2_msun is not None:
        CU2 = unitconv.geom_umass_msun(mass_scale2_msun)
        h2.time_unit_change(CU2.time, inverse=True)
        if distance2 is not None:
            distance2_SI = distance2 * unitconv.MEGAPARSEC_SI
            # Remember, h is actually r * h!
            #
            # We will work with rh (in physical units). This does not change the
            # result because r is just a constant.
            h2 *= CU2.length / distance2_SI
            redshift2 = gwu.luminosity_distance_to_redshift(distance2)
            h2.redshift(redshift2)

    # This keeps into account if we are using geometrized units or physical
    time_factor1 = CU1.time if mass_scale1_msun is not None else 1
    time_factor2 = CU2.time if mass_scale2_msun is not None else 1

    if time_removed_beginning is not None:
        h1.initial_time_remove(time_removed_beginning * time_factor1)
        h2.initial_time_remove(time_removed_beginning * time_factor2)

    if time_to_keep_after_max is not None:
        h1.crop(end=time_to_keep_after_max * time_factor1)
        h2.crop(end=time_to_keep_after_max * time_factor2)

    if window_function is not None:
        # Next, we window and zero-pad
        h1.window(window_function, *args, **kwargs)
        h2.window(window_function, *args, **kwargs)

    h1.zero_pad(num_zero_pad)
    h2.zero_pad(num_zero_pad)

    return h1, h2


def one_detector_mismatch_from_psi4(
    psi1,
    psi2,
    pcut1,
    pcut2,
    *args,
    window_function=None,
    align_at_peak=True,
    trim_ends=False,
    mass_scale1_msun=None,
    mass_scale2_msun=None,
    distance1=None,
    distance2=None,
    fmin=0,
    fmax=np.inf,
    noise=None,
    num_zero_pad=16384,
    num_time_shifts=100,
    num_polarization_shifts=100,
    time_shift_start=-50,
    time_shift_end=50,
    force_numba=False,
    time_to_keep_after_max=None,
    time_removed_beginning=None,
    **kwargs
):
    r"""Compute the mismatch between strains from psi1 and psi2.

    This function is complex, read :ref:`gw_mismatch:Overlap and mismatch` for
    formulas and details.

    :param psi1: :math:`\Psi_4` for the first wave.
    :type psi1: :py:class:`~.GravitationalWavesOneDet`
    :param psi2: :math:`\Psi_4` for the second wave (the one that will be
                 modified).
    :type psi2: :py:class:`~.GravitationalWavesOneDet`
    :param right_ascension: Right ascension of the source in the sky.
    :type right_ascension: float
    :param declination: Declination of the source in the sky.
    :type declination: float
    :param mass_scale1_msun: If not None, the signal h1 is converted from
                             computational units to physical units assuming that
                             M = mass_scale1_msun.
    :type mass_scale1_msun: float or None
    :param mass_scale2_msun: If not None, the signal h2 is converted from
                             computational units to physical units assuming that
                             M = mass_scale2_msun.
    :type mass_scale2_msun: float or None
    :param time_utc: Time UTC of the event.
    :type time_utc: float
    :param pcut1: Period associated with the threshold frequency
                 ``omega_0 = 2 * pi / pcut`` for the fixed frequency
                 integration of :math:`\Psi_4`
    :type pcut1: float
    :param pcut2: Period associated with the threshold frequency
                 ``omega_0 = 2 * pi / pcut`` for the fixed frequency
                 integration of :math:`\Psi_4`
    :type pcut2: float
    :param window_function: If not None, apply window_function to the
                            series before computing the strain.
    :type window_function: callable, str, or None
    :param trim_ends: If True, a portion of the resulting strain is removed
                      at both the initial and final times. The amount removed
                      is equal to pcut.
    :type trim_ends: bool
    :param align_at_peak: Time-shifts the strain so that they have both the
                          maximum amplitude at ``t=0``.
    :type align_at_peak: bool
    :param fmin: Lower limit of the integration.
    :type fmin: float
    :param fmax: Higher limit of the integration.
    :type fmax: float
    :param noises: Power spectral density of the noise for all the
                   detectors. If None, a uniform noise is applied.
    :type noises: :py:class:`~.Detectors`, or None
    :param num_zero_pad: How many points do the timeseries have to have
                         before Fourier transforms are taken? This is not the
                         number of zeros added (at the end) of the timeseries,
                         this is the total number. If the series already have
                         that length, no zeros will be added.
    :type num_zero_pad: int
    :param num_polarization_shifts: How many points to divide the range
                                    (0, 2 pi) in the polarization shift.
    :type num_polarization_shifts: int

    :param num_time_shifts: How many points to divide the range
                            (time_shift_start, time_shift_end) in the time shift.
    :type num_time_shifts: int
    :param time_shift_start: Minimum time shift applied. Search will be
                             done linearly up to time_shift_end.
    :type time_shift_start: float
    :param time_shift_end: Largest value of time shift applied.
    :type time_shift_end: float
    :param time_removed_beginning: Remove this amount from the beginning
                                   of the strain signals before computing
                                   the mismatch. If None, nothing is removed.
    :type time_removed_beginning: float or None
    :param time_to_keep_after_max: If not None, remove all the signal that
                                   comes after t_max + time_to_keep_after_max,
                                   where t_max is the time at which the signal
                                   peaks.
    :type time_to_keep_after_max: float or None
    :param force_numba: Use numba irrespectively of the size of the input.
    :type force_numba: bool
    :param args: All the other arguments are passed to the window
                  function.
    :type args: anything

    """
    h1, h2 = _strains_from_psi4(
        psi1,
        psi2,
        pcut1,
        pcut2,
        *args,
        window_function=window_function,
        align_at_peak=align_at_peak,
        trim_ends=trim_ends,
        mass_scale1_msun=mass_scale1_msun,
        mass_scale2_msun=mass_scale2_msun,
        distance1=distance1,
        distance2=distance2,
        num_zero_pad=num_zero_pad,
        time_to_keep_after_max=time_to_keep_after_max,
        time_removed_beginning=time_removed_beginning,
        **kwargs
    )

    return mismatch_from_strains(
        h1,
        h2,
        fmin,
        fmax,
        noises=[noise],
        antenna_patterns=None,
        num_polarization_shifts=num_polarization_shifts,
        num_time_shifts=num_time_shifts,
        time_shift_start=time_shift_start,
        time_shift_end=time_shift_end,
        force_numba=force_numba,
    )


def network_mismatch_from_psi4(
    psi1,
    psi2,
    right_ascension,
    declination,
    time_utc,
    pcut1,
    pcut2,
    *args,
    window_function=None,
    align_at_peak=True,
    trim_ends=False,
    mass_scale1_msun=None,
    mass_scale2_msun=None,
    distance1=None,
    distance2=None,
    fmin=0,
    fmax=np.inf,
    noises=None,
    num_zero_pad=16384,
    num_time_shifts=100,
    num_polarization_shifts=100,
    time_shift_start=-50,
    time_shift_end=50,
    force_numba=False,
    time_to_keep_after_max=None,
    time_removed_beginning=None,
    **kwargs
):
    r"""Compute the mismatch between strains from psi1 and psi2.

    This function is complex, read :ref:`gw_mismatch:Overlap and mismatch` for
    formulas and details.

    :param psi1: :math:`\Psi_4` for the first wave.
    :type psi1: :py:class:`~.GravitationalWavesOneDet`
    :param psi2: :math:`\Psi_4` for the second wave (the one that will be
                 modified).
    :type psi2: :py:class:`~.GravitationalWavesOneDet`
    :param right_ascension: Right ascension of the source in the sky.
    :type right_ascension: float
    :param declination: Declination of the source in the sky.
    :type declination: float
    :param mass_scale1_msun: If not None, the signal h1 is converted from
                             computational units to physical units assuming that
                             ``M = mass_scale1_msun``.
    :type mass_scale1_msun: float or None
    :param mass_scale2_msun: If not None, the signal h2 is converted from
                             computational units to physical units assuming that
                             M = mass_scale2_msun.
    :type mass_scale2_msun: float or None
    :param time_utc: Time UTC of the event.
    :type time_utc: float
    :param pcut1: Period associated with the threshold frequency
                  ``omega_0 = 2 * pi / pcut`` for the fixed frequency
                  integration of :math:`\Psi_4`
    :type pcut1: float
    :param pcut2: Period associated with the threshold frequency
                  ``omega_0 = 2 * pi / pcut`` for the fixed frequency
                  integration of :math:`\Psi_4`
    :type pcut2: float
    :param window_function: If not None, apply window_function to the
                            series before computing the strain.
    :type window_function: callable, str, or None
    :param trim_ends: If True, a portion of the resulting strain is removed
                      at both the initial and final times. The amount
                      removed is equal to pcut.
    :type trim_ends: bool
    :param align_at_peak: Time-shifts the strain so that they have both the
                          maximum amplitude at ``t=0``.
    :type align_at_peak: bool
    :param fmin: Lower limit of the integration.
    :type fmin: float
    :param fmax: Higher limit of the integration.
    :type fmax: float
    :param noises: Power spectral density of the noise for all the
                   detectors. If None, a uniform noise is applied.
    :type noises: :py:class:`~.Detectors`, or None
    :param num_zero_pad: How many points do the timeseries have to have
                         before Fourier transforms are taken? This is not
                         the number of zeros added (at the end) of the timeseries,
                         this is the total number. If the series already have that
                         length, no zeros will be added.
    :type num_zero_pad: int
    :param num_polarization_shifts: How many points to divide the range
                                    (0, 2 pi) in the polarization shift.
    :type num_polarization_shifts: int
    :param num_time_shifts: How many points to divide the range
                            (time_shift_start, time_shift_end) in the time shift.
    :type num_time_shifts: int
    :param time_shift_start: Minimum time shift applied. Search will be
                             done linearly up to time_shift_end.
    :type time_shift_start: float
    :param time_shift_end: Largest value of time shift applied.
    :type time_shift_end: float
    :param time_removed_beginning: Remove this amount from the beginning
                                   of the strain signals before computing the
                                   mismatch. If None, nothing is removed.
    :type time_removed_beginning: float or None
    :param time_to_keep_after_max: If not None, remove all the signal that
                                   comes after t_max + time_to_keep_after_max,
                                   where t_max is the time at which the signal peaks.
    :type time_to_keep_after_max: float or None
    :param force_numba: Use numba irrespectively of the size of the input.
    :type force_numba: bool
    :param args: All the other arguments are passed to the window
                  function.
    :type args: anything

    """
    h1, h2 = _strains_from_psi4(
        psi1,
        psi2,
        pcut1,
        pcut2,
        *args,
        window_function=window_function,
        align_at_peak=align_at_peak,
        trim_ends=trim_ends,
        mass_scale1_msun=mass_scale1_msun,
        mass_scale2_msun=mass_scale2_msun,
        distance1=distance1,
        distance2=distance2,
        num_zero_pad=num_zero_pad,
        time_to_keep_after_max=time_to_keep_after_max,
        time_removed_beginning=time_removed_beginning,
        **kwargs
    )

    return network_mismatch(
        h1,
        h2,
        right_ascension,
        declination,
        time_utc,
        fmin=fmin,
        fmax=fmax,
        noises=noises,
        num_polarization_shifts=num_polarization_shifts,
        num_time_shifts=num_time_shifts,
        time_shift_start=time_shift_start,
        time_shift_end=time_shift_end,
        force_numba=force_numba,
    )
