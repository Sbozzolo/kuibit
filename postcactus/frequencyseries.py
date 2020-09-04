#!/usr/bin/env python3

"""The :py:mod:`~.frequencyseries` module provides a representation of
frequency series.

"""

import numpy as np
from scipy.signal import argrelextrema

from postcactus import timeseries
from postcactus.series import BaseSeries, sample_common


def load_FrequencySeries(path, complex_on_two_columns=False, *args, **kwargs):
    """Load a text file as a FrequencySeries.

    The backend is np.loadtxt, so you can pass args or kwargs (for example to
    specify the columns).

    :param path: Path of the file to be loaded
    :type path: str
    :param complex_on_two_columns: When true, it is assumed that the real and
                                   the imaginary parts of the frequency series
                                   are on two columns. Otherwise, on one.
                                   This has to be False to load real data
                                   (e.g., noise curves).
    :type complex_on_two_columns: bool
    :returns: Loaded Frequencyseries
    :rtype: :py:mod:`~.FrequencySeries`

    """
    if (complex_on_two_columns):
        f, fft_real, fft_imag = np.loadtxt(path, unpack=True, ndmin=2,
                                           *args, **kwargs)
        fft = fft_real + 1j * fft_imag
    else:
        f, fft = np.loadtxt(path, unpack=True, ndmin=2,
                            *args, **kwargs)
    return FrequencySeries(f, fft)


def load_noise_curve(path, *args, **kwargs):
    """Load a noise curve as a FrequencySeries.

    This is syntatic sugar for the function load_FrequencySeries.

    :param path: Path of the file to be loaded
    :type path: str
    :returns: Loaded Frequencyseries
    :rtype: :py:mod:`~.FrequencySeries`
    """
    return load_FrequencySeries(path, complex_on_two_columns=False,
                                *args, **kwargs)


class FrequencySeries(BaseSeries):
    """Class representing a Fourier spectrum.

    :ivar f:   Frequency
    :vartype f: 1D numpy array or float

    :ivar fft:   Fourier transform
    :vartype fft: 1D numpy array or float

    """

    def __init__(self, f, fft):
        """Create a FrequencySeries providing frequencies and the
        value at those frequencies.

        It is your duty to make sure everything makes sense!

        :param f:  Frequencies
        :type f: 1D numpy array or float

        :param fft:   Fourier transform
        :type fft: 1D numpy array or float

        """
        # Use BaseClass init
        super().__init__(f, fft)

    # The following are the setters and getters, so that we can
    # resolve "self.f" and "self.fft"
    # Read documentation on BaseSeries
    @property
    def f(self):
        # This is defined BaseClass
        return self.data_x

    @f.setter
    def f(self, f):
        # This is defined BaseClass
        self.data_x = f

    @property
    def fft(self):
        # This is defined BaseClass
        return self.data_y

    @fft.setter
    def fft(self, fft):
        # This is defined BaseClass
        self.data_y = fft

    @property
    def fmin(self):
        """Return the minimum frequency.

        :returns:  Minimum frequency of the frequencyseries
        :rtype:    float
        """
        return np.amin(self.f)

    @property
    def fmax(self):
        """Return the maximum frequency.

        :returns:  Maximum frequency of the frequencyseries
        :rtype:    float
        """
        return np.amax(self.f)

    @property
    def frange(self):
        """Return the range of frequencies.

        :returns:  Range of the frequencyseries
        :rtype:    float
        """
        return self.fmax - self.fmin

    @property
    def amplitude(self):
        """Return the amplitude of frequencies.

        :returns:  Range of the frequencyseries
        :rtype:    1d numpy array of float
        """
        return abs(self.fft)

    # Writing amplitude all the times can be boring
    amp = amplitude

    @property
    def df(self):
        """Return the delta f if the series is regularly sampled,
        otherwise raise error.

        :returns: Delta t
        :rtype: float

        """
        df = self.f[1:] - self.f[:-1]
        df0 = df[0]

        if (not np.allclose(df, df0)):
            raise ValueError("FrequencySeries is not regularly sampled")

        return df0

    def normalized(self):
        """Return a new frequencyseries with maximum amplitude of 1.

        :returns: Normalized frequency series.
        :rtype: :py:mod:`~.FrequencySeries`

        """
        m = self.amplitude.max()

        if (m <= 0):
            raise ValueError("Non positive PSD maximum!")

        return self / m

    def normalize(self):
        """Scale values so that the maximum of the amplitude is 1.

        """
        self._apply_to_self(self.normalized)

    def low_passed(self, f):
        """FIXME

        :param f: Frequency above which series will be zeroed.
        :returns:
        :rtype:

        """
        msk = (np.abs(self.f) <= f)
        return FrequencySeries(self.f[msk], self.fft[msk])

    def low_pass(self, f):
        """Remove frequencies higher or equal than f (absolute value).

        """
        self._apply_to_self(self.low_passed, f)

    def high_passed(self, f):
        """Remove frequencies lower or equal than f

        :param f:

        """
        msk = (np.abs(self.f) >= f)
        return FrequencySeries(self.f[msk], self.fft[msk])

    def high_pass(self, f):
        """FIXME! briefly describe function

        :param f:
        :returns:
        :rtype:

        """
        self._apply_to_self(self.high_passed, f)

    def band_passed(self, fmin, fmax):
        """Remove frequencies outside the range fmin, fmax

        :param fmin:
        :param fmax:

        """
        ret = self.low_passed(fmax)
        return ret.high_passed(fmin)

    def band_pass(self, fmin, fmax):
        """FIXME! briefly describe function

        :param fmin:
        :param fmax:
        :returns:
        :rtype:

        """
        self._apply_to_self(self.band_passed, fmin, fmax)

    def peaks(self, amp_threshold=0.0):
        """Return the location and amplitue of the peaks of the amplitue.

        Peaks at the boundaries are not considered.

        :param amp_threshold: Ignore peaks smaller than this value
        :type amp_threshold:  float

        :returns: Peaks of the amplitude
        :rtype: List of tuples of three elements: frequency of the bin in which
                the peak is, fitted frequency of the maximum with parabolic
                approximation, amplitude of the peak
        """
        peaks_f_indexes = argrelextrema(self.amp, np.greater)

        # We do not consider the peaks near the boundaries and those smaller
        # than amp_threshold
        peaks_f_indexes = [
            i for i in peaks_f_indexes[0]
            if 1 < i < len(self) - 2 and self.amp[i] > amp_threshold
        ]

        peaks_f = self.f[peaks_f_indexes]
        peaks_amp = self.amp[peaks_f_indexes]
        peaks_ff = [
            (self.f[i] + 0.5 * self.df * (self.amp[i + 1] - self.amp[i - 1]) /
             (2.0 * self.amp[i] - self.amp[i - 1] - self.amp[i + 1]))
            for i in peaks_f_indexes
        ]

        return tuple(zip(peaks_f, peaks_ff, peaks_amp))

    def peaks_frequencies(self, amp_threshold=0.0):
        """Return the frequencies of the peaks of the amplitude larger
        than amp_threshold.

        The frequency is computed with a quadratic fit using the left and right
        neighbours.

        Peaks at the boundaries are not considered.

        :param amp_threshold: Ignore peaks smaller than this value
        :type amp_threshold:  float

        :returns: Fitted frequencies of the peaks of the amplitude
        :rtype: 1d numpy array

        """
        return np.array([p[1] for p in self.peaks(amp_threshold)])

    def to_TimeSeries(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        # TimeSeries.to_FrequencySeries() rearranges the frequency so that
        # negative are on the left and positive on the right. Here, we undo
        # that.

        fft = np.fft.ifftshift(self.fft)

        t = np.fft.fftfreq(len(self), d=self.df)
        t = np.fft.fftshift(t)
        y = np.fft.ifft(fft)

        return timeseries.TimeSeries(t, y)

    def inner_product(self, other, fmin=0, fmax=np.inf, noise=None):
        r"""Compute the inner product.

        :math:`(h_1, h_2) = 4 \Re \int_{f_min}^{f_max} \frac{h_1 h_2^*}{S_n}`

        We assume that the frequencyseries are zero outside of the interval of
        definition, so if fmax (fmin) is larger (smaller) than the one
        available, it is effectively set to the one available.

        :param other: Second frequency series in the inner product
        :type other: :py:class:`.FrequencySeries`
        :param fmin: Remove frequencies below fmin
        :type fmin: float
        :param fmax: Remove frequencies above fmin
        :type fmax: float
        :param noise: If None, no weight is applied
        :type noise: :py:class:`.FrequencySeries` or None

        :returns: Inner product between self and other
        :rtype: float

        """
        if (not isinstance(other, type(self))):
            raise TypeError("The other object is not a FrequencySeries")

        if ((not isinstance(noise, type(self))) and (noise is not None)):
            raise TypeError("Noise is not FrequencySeries or None")

        if (noise is None):
            # If noise is None, it means that the weight is one everywhere so,
            # we prepare a FrequencySeries that has the same frequencies as self.
            # Everything will be resampled to a common set
            noise = FrequencySeries(self.f, np.ones_like(self.fft))

        # "res" = "resampled"
        [res_self, res_other, res_noise] = sample_common([self, other, noise])
        # Noise has better be real.
        integrand = 4 * res_self * res_other.conjugate() / res_noise
        # 4 Re * \int
        integral = integrand.integrated().real()

        # We assume that the frequencyseries are zero outside of the interval of
        # definition
        if fmax > integral.fmax:
            fmax = integral.fmax
        if fmin < integral.fmin:
            fmin = integral.fmin

        return integral(fmax) - integral(fmin)

    def overlap(self, other, fmin=0, fmax=np.inf, noise=None):
        r"""Compute the overlap.

        :math:`\textrm{overlap} = (h_1, h_2) / \sqrt{(h_1, h_1)(h_2, h_2)}`

        We assume that the frequencyseries are zero outside of the interval of
        definition, so if fmax (fmin) is larger (smaller) than the one
        available, it is effectively set to the one available.

        :param other: Second frequency series in the overlap
        :type other: :py:class:`.FrequencySeries`
        :param fmin: Remove frequencies below fmin
        :type fmin: float
        :param fmax: Remove frequencies above fmin
        :type fmax: float
        :param noise: If None, no weight is applied
        :type noise: :py:class:`.FrequencySeries` or None

        :returns: Overlap between self and other
        :rtype: float

        """
        # Error handling is done by inner_product
        inner_11 = self.inner_product(self, fmin, fmax, noise)
        inner_22 = other.inner_product(other, fmin, fmax, noise)
        inner_12 = self.inner_product(other, fmin, fmax, noise)
        return inner_12 / np.sqrt(inner_11 * inner_22)
