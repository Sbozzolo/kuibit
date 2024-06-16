#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
#
# Inspired by code originally developed by Wolfgang Kastaun. This file may
# contain algorithms and/or structures first implemented in
# GitHub:wokast/PyCactus/PostCactus/fourier_util.py
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


"""The :py:mod:`~.frequencyseries` module provides a representation of
frequency series.

:py:class:`~.FrequencySeries` can be evenly or unevenly sampled, real or
complex. They support all the mathematical operations and operators you may
expect, and have additional methods, which include ones for taking derivatives,
integrals, apply windows, smooth the signal, take inverse Fourier transform, and
more. Most of these methods are available in two flavors: those that return a
new :py:class:`~.FrequencySeries`, and those which modify the object in place. The
latter have names with imperative verbs.

As in :py:class:`~.TimeSeries`, :py:class:`~.FrequencySeries` are derived from
the :py:class:`~.BaseSeries`, which in turn is derived from the abstract class
:py:class:`~.BaseNumerical`. Some of the capabilities of
:py:class:`~.FrequencySeries` (e.g., overloading the mathematical operators) are
implemented in the parent classes.

Additionally, two functions are defined in this module
:py:func:`~.load_FrequencySeries` and :py:func:`~.load_noise_curve`. Both loads
:py:class:`~.FrequencySeries` from a text file, but :py:func:`~.load_noise_curve`
is a simpler interface for real-valued signals.

"""

import numpy as np
from scipy.signal import argrelextrema

from kuibit import timeseries
from kuibit.series import BaseSeries, sample_common


def load_FrequencySeries(path, *args, complex_on_two_columns=False, **kwargs):
    """Load a text file as a :py:class:`~.FrequencySeries`.

    The backend is ``np.loadtxt``, and the unknown arguments passed to this
    function are given to ``np.loadtxt``. This can be used, for example, to
    specify the columns of the file that have to be read.

    :param path: Path of the file to be loaded.
    :type path: str
    :param complex_on_two_columns: When true, it is assumed that the real and
                                   the imaginary parts of the frequency series
                                   are on two columns. Otherwise, on one.
                                   This has to be False to load real data
                                   (e.g., noise curves).
    :type complex_on_two_columns: bool
    :returns: Loaded data as :py:class:`~.FrequencySeries`.
    :rtype: :py:class:`~.FrequencySeries`

    """
    if complex_on_two_columns:
        f, fft_real, fft_imag = np.loadtxt(
            path, unpack=True, ndmin=2, *args, **kwargs
        )
        fft = fft_real + 1j * fft_imag
    else:
        f, fft = np.loadtxt(path, unpack=True, ndmin=2, *args, **kwargs)
    return FrequencySeries(f, fft)


def load_noise_curve(path, *args, **kwargs):
    """Load a noise curve as a :py:class:`~.FrequencySeries`.

    Unknown arguments are passed to ``np.loadtxt``.

    This is syntactic sugar for the function :py:func:`~.load_FrequencySeries.`

    :param path: Path of the file to be loaded.
    :type path: str
    :returns: Noise curve.
    :rtype: :py:class:`~.FrequencySeries`
    """
    return load_FrequencySeries(
        path, complex_on_two_columns=False, *args, **kwargs
    )


class FrequencySeries(BaseSeries):
    """Class representing a Fourier spectrum.

    :ivar f:   Frequency
    :vartype f: 1D NumPy array or float

    :ivar fft:   Fourier transform
    :vartype fft: 1D NumPy array or float

    """

    # skiqc PYL-W0235
    def __init__(self, f, fft, guarantee_f_is_monotonic=False):
        """Create a :py:class:`~.FrequencySeries` providing frequencies and the value at
        those frequencies.

        It is your duty to make sure everything makes sense!

        When ``guarantee_f_is_monotonic`` is True, no checks will be perform to
        make sure that f is monotonically increasing (increasing performance).
        This should is used internally whenever a new series is returned from
        self (since we have already checked that ``f`` is good.) or in performance
        critical routines.

        :param f:  Frequencies.
        :type f: 1D NumPy array or float

        :param fft:   Fourier transform.
        :type fft: 1D NumPy array or float

        :param guarantee_f_is_monotonic: If true, it will be assumes that ``f`` is
                                         monotonically increasing.
        :type guarantee_f_is_monotonic: bool

        """
        # Use BaseClass init
        super().__init__(f, fft, guarantee_f_is_monotonic)

    # The following are the setters and getters, so that we can
    # resolve "self.f" and "self.fft"
    # Read documentation on BaseSeries
    @property
    def f(self):
        """Frequencies.

        :returns: Frequencies.
        :rtype: 1d NumPy array
        """
        # This is defined BaseClass
        return self.x

    @f.setter
    def f(self, f):
        # This is defined BaseClass
        self.x = f

    @property
    def fft(self):
        """Fourier components.

        :returns: Fourier components.
        :rtype: 1d NumPy array
        """
        # This is defined BaseClass
        return self.y

    @fft.setter
    def fft(self, fft):
        # This is defined BaseClass
        self.y = fft

    @property
    def fmin(self):
        """Return the minimum frequency.

        :returns:  Minimum frequency of the series.
        :rtype:    float
        """
        return self.xmin

    @property
    def fmax(self):
        """Return the maximum frequency.

        :returns:  Maximum frequency of the series.
        :rtype:    float
        """
        return self.xmax

    @property
    def frange(self):
        """Return the range of frequencies.
        The range is defined as the maximum frequency minus the minimum.

        :returns:  Range of the series (``f_max`` - ``f_min``).
        :rtype:    float
        """
        return self.fmax - self.fmin

    @property
    def amplitude(self):
        """Return the amplitude of frequencies.

        :returns:  Amplitude of the series.
        :rtype:    1d NumPy array
        """
        return abs(self.fft)

    # Writing amplitude all the times can be boring
    amp = amplitude

    @property
    def df(self):
        """Return the spacing (``delta_f``) if the series is regularly sampled,
        otherwise raise error.

        :returns: Frequency spacing (``delta_f``).
        :rtype: float

        """
        if not self.is_regularly_sampled():
            raise ValueError("Frequencyseries is not regularly sampled")

        return self.f[1] - self.f[0]

    def normalized(self):
        """Return a new :py:class:`~.FrequencySeries` with maximum amplitude of 1.

        :returns: Normalized :py:class:`~.FrequencySeries` series.
        :rtype: :py:class:`~.FrequencySeries`

        """
        m = self.amplitude.max()

        if m <= 0:
            raise ValueError("Non positive PSD maximum!")

        return self / m

    def normalize(self):
        """Scale values so that the maximum of the amplitude is 1."""
        self._apply_to_self(self.normalized)

    def low_passed(self, f):
        """Remove frequencies higher or equal than the given.

        :param f: Frequency above which the series will be zeroed.
        :type f: float
        :returns: Low-passed :py:class:`~.FrequencySeries`.
        :rtype: :py:class:`~.FrequencySeries`


        """
        msk = np.abs(self.f) <= f
        return FrequencySeries(self.f[msk], self.fft[msk])

    def low_pass(self, f):
        """Remove frequencies higher or equal than ``f`` (absolute value).

        :param f: Frequency above which series will be zeroed.
        :type f: float

        """
        self._apply_to_self(self.low_passed, f)

    def high_passed(self, f):
        """Remove frequencies lower or equal than the given.

        :param f: Frequency below which series will be zeroed.
        :type f: float
        :returns: High-passed :py:class:`~.FrequencySeries`.
        :rtype: :py:class:`~.FrequencySeries`

        """
        msk = np.abs(self.f) >= f
        return FrequencySeries(self.f[msk], self.fft[msk])

    def high_pass(self, f):
        """Remove all the frequencies smaller than f

        :param f: Frequency below which series will be zeroed.
        :type f: float

        """
        self._apply_to_self(self.high_passed, f)

    def negative_frequencies_removed(self):
        """Remove frequencies lower than 0.

        :returns:  Frequencyeseries with only positive frequencies.
        :rtype: :py:class:`~.FrequencySeries`

        """
        msk = self.f >= 0
        return FrequencySeries(self.f[msk], self.fft[msk])

    def negative_frequencies_remove(self):
        """Remove all the frequencies smaller than 0."""
        self._apply_to_self(self.negative_frequencies_removed)

    def band_passed(self, fmin, fmax):
        """Remove frequencies outside the given range ``fmin, fmax``.

        :param fmin: Minimum frequency.
        :type fmin: float
        :param fmax: Maximum frequency.
        :type fmax: float

        """
        ret = self.low_passed(fmax)
        return ret.high_passed(fmin)

    def band_pass(self, fmin, fmax):
        """Remove all the frequencies below ``fmin`` and above ``fmax``.

        :param fmin: Minimum frequency.
        :type fmin: float
        :param fmax: Maximum frequency.
        :type fmax: float

        """
        self._apply_to_self(self.band_passed, fmin, fmax)

    def peaks(self, amp_threshold=0.0):
        """Return the location and amplitude of the peaks of the amplitude.

        Peaks at the boundaries are not considered.

        :param amp_threshold: Ignore peaks smaller than this value.
        :type amp_threshold:  float

        :returns: Peaks of the amplitude as list of tuples of three elements:
                frequency of the bin in which the peak is, fitted frequency of
                the maximum with parabolic approximation, amplitude of the peak.
        :rtype: List of tuples of three elements.

        """
        peaks_f_indexes = argrelextrema(self.amp, np.greater)

        # We do not consider the peaks near the boundaries and those smaller
        # than amp_threshold
        peaks_f_indexes = [
            i
            for i in peaks_f_indexes[0]
            if 1 < i < len(self) - 2 and self.amp[i] > amp_threshold
        ]

        peaks_f = self.f[peaks_f_indexes]
        peaks_amp = self.amp[peaks_f_indexes]
        peaks_ff = [
            (
                self.f[i]
                + 0.5
                * self.df
                * (self.amp[i + 1] - self.amp[i - 1])
                / (2.0 * self.amp[i] - self.amp[i - 1] - self.amp[i + 1])
            )
            for i in peaks_f_indexes
        ]

        return tuple(zip(peaks_f, peaks_ff, peaks_amp))

    def peaks_frequencies(self, amp_threshold=0.0):
        """Return the frequencies of the peaks of the amplitude larger
        than ``amp_threshold``.

        The frequency is computed with a quadratic fit using the left and right
        neighbors.

        Peaks at the boundaries are not considered.

        :param amp_threshold: Ignore peaks smaller than this value.
        :type amp_threshold:  float

        :returns: Fitted frequencies of the peaks of the amplitude.
        :rtype: 1d NumPy array

        """
        return np.array([p[1] for p in self.peaks(amp_threshold)])

    def to_TimeSeries(self):
        """Perform a inverse Fourier transform.

        If only positive frequencies are found, we will assume that the
        original signal was real.

        :returns: Inverse Fourier transform.
        :rtype: :py:class:`.TimeSeries`

        """
        if self.is_masked():
            raise RuntimeError(
                "Fourier transform with masked data is not supported."
            )

        # If fmin >= 0, then, the signal was probably real to begin with.
        # We will restore the negative frequencies so that the operation
        # is the actual inverse of taking the dft.
        #
        if self.fmin < 0:
            # TimeSeries.to_FrequencySeries() rearranges the frequency so that
            # negative are on the left and positive on the right. Here, we undo
            # that.

            fft = np.fft.ifftshift(self.fft)

            t = np.fft.fftfreq(len(self.f), d=self.df)
            t = np.fft.fftshift(t)
            y = np.fft.ifft(fft)
        else:
            y = np.fft.irfft(self.fft)

            # To find the times we have to restore the negative frequencies
            # So, we simply recompute them assuming the current df
            frequencies = np.linspace(-self.fmax, self.fmax, len(y))
            t = np.fft.fftfreq(len(frequencies), d=self.df)
            # This not the order NumPy likes
            t = np.fft.fftshift(t)

        # We need the normalization df to compute physical quantities.
        # Intuitively, NumPy computes a_k = 1 / N \sum A_k exp(2 pi f t), to
        # transform this into an integral (true Fourier transform), we have to
        # multiply this by the measure of integration.

        # NOTE: Why exactly do we need len(t) here?
        #       It works, and PyCBC does it too. But what is the reason?
        return timeseries.TimeSeries(t, y * len(t) * self.df)

    def inner_product(
        self,
        other,
        fmin=0,
        fmax=np.inf,
        noises=None,
        same_domain=False,
    ):
        r"""Compute the (network) inner product with another :py:class:`~.FrequencySeries`.

        This is defined as:

        :math:`(h_1, h_2) = 4 \Re \int_{f_min}^{f_max} \frac{h_1 h_2^*}{S_n}`

        where ``S_n`` is the noise curve, and ``h_1``, ``h_2`` the series.

        In case multiple noise curves are supplied, compute

        :math:`(h_1, h_2) = \sum_{detectors}
        4 \Re \int_{f_min}^{f_max} \frac{h_1 h_2^*}{S_n}`

        This is the network inner product. To compute this quantity, you have
        to provide a list of noises.

        We assume that the :py:class:`~.FrequencySeries` are zero outside of the
        interval of definition, so if ``fmax`` (``fmin``) is larger (smaller)
        than the one available, it is effectively set to the one available.

        Since Fourier typically transforms explode at fmin = 0, the result of
        the integration is highly sensitive to regions near that frequency.

        If ``same_domain`` is True, it is assumed that all the
        :py:class:`~.FrequencySeries` involved are defined over the same
        frequencies. Turning this on speeds up computations, but it will result
        in incorrect results if the assumption is violated. If it is False, the
        domain of definition of the series is checked, if it is not the same for
        all the series, then they will be resampled.

        :param other: Second frequency series in the inner product.
        :type other: :py:class:`.FrequencySeries`
        :param fmin: Remove frequencies below this value.
        :type fmin: float
        :param fmax: Remove frequencies above this value.
        :type fmax: float
        :param noise: If None, no weight is applied.
        :type noise: :py:class:`.FrequencySeries`, list
                     of :py:class:`.FrequencySeries` or None
        :param same_domain: Whether to assume that the :py:class:`~.FrequencySeries`
                            are defined over the same frequencies. If you can
                            guarantee this, the computation will be faster.
        :type same_domain: bool

        :returns: Inner product between ``self`` and ``other``.
        :rtype: float

        """
        if not isinstance(other, type(self)):
            raise TypeError("The other object is not a FrequencySeries")

        if (
            (not isinstance(noises, type(self)))
            and (not isinstance(noises, list))
            and (noises is not None)
        ):
            raise TypeError("Noise is not (a list of) FrequencySeries or None")

        if fmin >= fmax:
            raise ValueError("fmin has to be smaller than fmax")

        if fmin < 0:
            raise ValueError("fmin has to be non-negative")

        if noises is None:
            # If noises is None, it means that the weight is one everywhere so,
            # we prepare a FrequencySeries that has the same frequencies as
            # self.
            # Everything will be resampled to a common set
            noises = FrequencySeries(self.f, np.ones_like(self.fft))

        # "res" = "resampled"
        to_be_res_list = [self, other]
        # Check if noises is a list, in that case add all the elements to
        # to to_be_res_list
        if isinstance(noises, list):
            to_be_res_list.extend(noises)
        else:
            # noises is not a list, just append it
            to_be_res_list.append(noises)

        if not same_domain:
            # Noises typically have Lorentian features, better to use a 0d
            # spline, so we enable piecewise_constant
            [res_self, res_other, *res_noises] = sample_common(
                to_be_res_list, resample=True, piecewise_constant=True
            )
        else:
            [res_self, res_other, *res_noises] = to_be_res_list

        for series in [res_self, res_other, *res_noises]:
            series.negative_frequencies_remove()
            series.band_pass(fmin=fmin, fmax=fmax)

        # Sum all the integrands
        integrand = FrequencySeries(res_self.f, np.zeros_like(res_self.fft))

        # We manipulate directly the fft fields because we have already
        # established that the series are defined on the same frequencies.
        # This is faster because it skips several sanity checks.
        for res_noise in res_noises:
            integrand += res_self * res_other.conjugate() / res_noise

        # 4 Re * \int
        # To align with PyCBC we do a rectangular integration here instead of
        # a trapeziodial one
        return 4 * np.sum(integrand.fft.real) * integrand.df

    def overlap(
        self,
        other,
        fmin=0,
        fmax=np.inf,
        noises=None,
        same_domain=False,
    ):
        r"""Compute the (network) overlap.

        This is defined as:

        :math:`\textrm{overlap} = (h_1, h_2) / \sqrt{(h_1, h_1)(h_2, h_2)}`

        where ``h_1``, ``h_2`` are the series.

        To compute is the network overlap, you have to provide a list of noises.

        We assume that the :py:class:`~.FrequencySeries` are zero outside of the
        interval of definition, so if ``fmax`` (``fmin``) is larger (smaller)
        than the one available, it is effectively set to the one available.

        Since Fourier typically transforms explode at fmin = 0, the result of
        the integration is highly sensitive to regions near that frequency.

        If ``same_domain`` is True, it is assumed that all the
        :py:class:`~.FrequencySeries` involved are defined over the same
        frequencies. Turning this on speeds up computations, but it will result
        in incorrect results if the assumption is violated. If it is False, the
        domain of definition of the series is checked, if it is not the same for
        all the series, then they will be resampled.


        :param other: Second frequency series in the overlap.
        :type other: :py:class:`.FrequencySeries`
        :param fmin: Remove frequencies below this value.
        :type fmin: float
        :param fmax: Remove frequencies above this value.
        :type fmax: float
        :param noise: If None, no weight is applied. If it is a list,
                      the netowrk overlap is computed.
        :type noise: (list of) :py:class:`.FrequencySeries` or None
        :param same_domain: Whether to assume that the :py:class:`~.FrequencySeries`
                            are defined over the same frequencies. If you can
                            guarantee this, the computation will be faster.
        :type same_domain: bool

        :returns: Overlap between ``self`` and ``other``.
        :rtype: float

        """
        # Error handling is done by inner_product
        inner_11 = self.inner_product(
            self,
            fmin=fmin,
            fmax=fmax,
            noises=noises,
            same_domain=same_domain,
        )
        inner_22 = other.inner_product(
            other,
            fmin=fmin,
            fmax=fmax,
            noises=noises,
            same_domain=same_domain,
        )
        inner_12 = self.inner_product(
            other, fmin, fmax, noises=noises, same_domain=same_domain
        )

        return inner_12 / np.sqrt(inner_11 * inner_22)
