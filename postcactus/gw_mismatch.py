#!/usr/bin/env python3

# Copyright (C) 2020 Gabriele Bozzola
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
two waves.

"""

import numpy as np
from warnings import warn

# What is this? This is numba!
# numba is a JITter (JIT = Just In Time). The following code is compiled at
# runtime. The compiled code is instead, and it is much faster.
#
# At the moment, fft is not supported, so the full power of numba cannot be
# achived.
from numba import njit, objmode

from postcactus import unitconv
from postcactus import frequencyseries as fs
from postcactus import gw_utils as gwu


def _mismatch_core_numerical(
    h1_c_fft,
    h1_p_fft,
    h2_t,
    frequencies,
    frequency_mask,
    noises,
    antenna_patterns,
    polarization_shifts,
    time_shifts,
):
    """Compute the maximum overlap between h1_fft (frequency domain, cross and
    plus) and h2_t (time domain). This function requires very specific
    pre-processing and should never be used directly.

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
        with objmode(
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
    r"""Compute the netowrk-mismatch between h1 and h2 by maximising the
    overlap over time and polarization shifts.

    All the transformations are done in h2.

    This is done by maximizing over time and phase shifts. Phase shifts and are
    around the 2pi, time shifts are specified by time_shift_start and
    time_shift_end. If num_time_shifts is 1, then no time shift is performed.

    Noises has to be a list of FrequencySeries with the PSD of the noise.
    Antenna patterns has to be the computed antenna pattern corresponding to
    the noise. Noises and antenna patterns have to be properly ordered:
    noises[i] has to correspond to antenna_pattern[i].

    For times, we make sure that we always have to zero timeshift.

    NO phase shifts here!

    There is an overhead in calling numba. Hence, by default we do not always
    use numba. We use it only when the number
    num_polarization_shifts * num_time_shifts is greater than 500*500. You can
    force using numba passing the keyword argument force_numba=True.

    This function is not meant to be used directly.

    """

    # In PostCactus, we have beautiful collection of classes to represent
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
    # the data as numpy arrays. In this function we pre-process the
    # FrequencySeries so that we can feed _mismatch_core_numerical with
    # what we need. _mismatch_core_numerical takes only standard numpy
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
        [0], np.linspace(time_shift_start, time_shift_end, num_time_shifts - 1)
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

    # 3. Then, the noise is typically defined on several frequencies (many more
    #    than the series), so we can downsample it to be the same as h1f.
    #    Hence, h1f and the noises will be defined on the same frequencies. We
    #    will only need to take care of the h2. If the noise is None, we
    #    prepare a unweighted noise (ones everywhere).

    if noises is not None:
        # With this, we can guarantee that everything has the same domain.
        # If there's a None entry, we fill it with a constant noise.
        noises_res = [
            n.resampled(h1f_p_res.f)
            if n is not None
            else fs.FrequencySeries(h1f_p_res.f, np.ones_like(h1f_p_res.fft))
            for n in noises
        ]
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

    if use_numba:
        _core_function = njit(_mismatch_core_numerical)
    else:
        _core_function = _mismatch_core_numerical

    (unnormalized_max_overlap, index_max) = _core_function(
        h1f_c_res.fft,
        h1f_p_res.fft,
        h2_res.y,
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

    return 1 - unnormalized_max_overlap / norm, (p_shift_max, t_shift_max)


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
    """h1 and h2 have to be already pre-processed for Fourier transform.

    At the moment, this mismatch makes sense only for the l=2, m=2 mode.
    (No maximisation over phis is done)

    Noises has to be a gw_utils.Detectors object, or None
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
    trim_ends=True,
    mass_scale1=None,
    mass_scale2=None,
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
    time_removed_after_max=None,
    initial_time_removed=None,
    **kwargs
):

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
    if mass_scale1 is not None:
        CU1 = unitconv.geom_umass_msun(mass_scale1)
        h1.time_unit_change(CU1.time, inverse=True)
        if distance1 is not None:
            distance1_SI = distance1 * unitconv.MEGAPARSEC_SI
            # Remember, h is actually r * h!
            h1 *= CU1.length / distance1_SI
            redshift1 = gwu.luminosity_distance_to_redshift(distance1)
            h1.redshift(redshift1)

    if mass_scale2 is not None:
        CU2 = unitconv.geom_umass_msun(mass_scale2)
        h2.time_unit_change(CU2.time, inverse=True)
        if distance2 is not None:
            distance2_SI = distance2 * unitconv.MEGAPARSEC_SI
            # Remember, h is actually r * h!
            h2 *= CU2.length / distance2_SI
            redshift2 = gwu.luminosity_distance_to_redshift(distance2)
            h2.redshift(redshift2)

    if initial_time_removed is not None:
        h1.initial_time_remove(initial_time_removed)
        h2.initial_time_remove(initial_time_removed)

    if time_removed_after_max is not None:
        h1.crop(end=time_removed_after_max)
        h2.crop(end=time_removed_after_max)

    if window_function is not None:
        # Next, we window and zero-pad
        h1.window(window_function, *args, **kwargs)
        h2.window(window_function, *args, **kwargs)

    h1.zero_pad(num_zero_pad)
    h2.zero_pad(num_zero_pad)

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
