#!/usr/bin/env python3
"""The :py:mod:`~.timeseries` module provides a representation of time series
and convenience functions to create timeseries

"""

import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import signal
import warnings


def remove_duplicate_iters(t, y):
    """Remove overlapping segments from a time series in (t,y).

    Only the latest of overlapping segments is kept, the rest
    removed.

    This function is used for cleaning up simulations with multiple
    checkpoints.

    Note, if t = [1, 2, 3, 4, 2, 3] the output will be [1, 2, 3].
    The '4' is discarded because it is not the last segment. The
    idea is that if this corresponds to a simulation restart, you
    may have changed the paramters, so that 4 is not anymore correct.
    We consider the second restart the "truth".

    :param t:  Times
    :type t:   1D numpy array
    :param y:  Values
    :type t:   1D numpy array

    :returns:  Strictly monotonic time series
    :rtype:    :py:class:`~.TimeSeries`

    """
    # Let's unpack this code.
    # First, we define a new variable t2.
    #
    # t2 is essentially the "cumulative minimum" of the times of the time
    # series: t2[i] is the minimum up to index i
    #
    # To be more specific, we walk the time array backwards (t[::-1]) and
    # we compute the cumulative minima. Then, we reverse the array ([::-1])
    #
    # For example, if t = [1, 2, 3, 4, 2, 3]
    # Then t[::-1] = [3, 2, 4, 3, 2, 1], and
    # np.minimum.accumulate(t[::-1]) = [3, 2, 2, 2, 2, 1]
    # Reversing it: t2 = [1, 2, 2, 2, 2, 3]
    #
    # If t had no nuplicates t2 and t would be the same.
    # When t has duplicates, t2 is like t but in place of the duplicates
    # it has values that are equal or smaller.
    #
    # What we want is to have as output [1, 2, 3]
    # To get that, we compare t and t2. Values that are not duplicated
    # are those subtracted with the following are positive.
    # (t[:-1] < t2[1:])

    # First, we make sure that we are dealing with arrays and not lists
    t = np.array(t)
    y = np.array(y)

    t2 = np.minimum.accumulate(t[::-1])[::-1]
    # Here we append [True] because the last point is always included
    msk = np.hstack((t[:-1] < t2[1:], [True]))

    return TimeSeries(t[msk], y[msk])


def unfold_phase(phase):
    """Remove phase jumps to get a continous (unfolded) phase.

    :param phase:     Phase
    :type phase:      1D numpy array

    :returns:         Phase plus multiples of pi chosen to minimize jumps
    :rtype:           1D numpy array
    """
    # TODO: This function should be generalized to allow arbitary jumps.
    #       This is trivially done by adding an argument jump with default
    #       value of 2 pi.

    # nph is how many time we reach 2 pi
    nph = phase / (2 * np.pi)
    # wind is the winding number, how many time we have went over 2 * np.pi
    wind = np.zeros_like(phase)
    # wind[0] = 0. Then, we find the jumps, when the phase goes from 2 pi to 0
    # (or anything with the same offset). Since we divided by 2 pi, a jump is
    # when nph goes from 1 to 0. np.rint allows us to identify the offest of 2
    # pi: when the difference between phase[:-1] - phase[1:] is greater than 2
    # pi, this will be rounded up to 1, when it is smaller, it is rounded down
    # to 0. For example, if phase[i] = np.pi + eps and phase[i+1] = -np.pi,
    # then, this is a jump and np.rint rounds to 1.
    wind[1:] = np.rint(nph[:-1] - nph[1:])
    # Finally, we collect how many jumps have occurred. This is the winding
    # number and tell us how many 2 pi we have to add.
    wind = np.cumsum(wind)
    return phase + (2 * np.pi) * wind


def combine_ts(series, prefer_late=True):
    """Combine several overlapping time series into one.

    In intervals covered by two or more time series, which data is used depends
    on the parameter prefer_late. If two segments start at the same time, the
    longer one gets used.

    :param series: The timeseries to combine
    :type series:  list of :py:class:`~.TimeSeries`
    :param prefer_late: Prefer data that starts later for overlapping segments
    :type prfer_late:   bool

    :returns:      The combined time series
    :rtype:        :py:class:`~.TimeSeries`

    """

    # Late and early can be implemented in one shot by implementing one and
    # sending t -> -t for the other. For the "straight" way we implement
    # combine_ts_early.
    #
    # Let's consider a simple example for the reversed case
    # t1 = [1, 2, 3], t2 = [2, 3, 4], we want to have t = [1, 2, 3, 4]
    # sign = -1
    # timeseries = [t2, t1]
    # times = t2[::-1] = [4, 3, 2]
    # Next we walk through the remaining elements of the list
    # We want only to keep those with t < times[-1] = 2 (hence the switch)
    # In this case msk = [3, 2, 1] < 2 = [False, False, True], so
    # s_t[msk] = [1] and times = [4, 3, 2, 1].
    # At the end, we need to reverse the order again

    # sign is responsible of inverting the sorting key
    sign = -1 if prefer_late else 1

    # Tuples are compared lexicographically; the first items are compared; if
    # they are the same then the second items are compared, and so on.
    # So here we sort by tmin and tmax
    timeseries = sorted(series,
                        key=lambda x: (sign * x.tmin(), sign * x.tmax()))
    # Now we are going to build up the t and y array, starting with the first
    times = timeseries[0].t[::sign]
    values = timeseries[0].y[::sign]
    for s in timeseries[1:]:
        # We need to walk backwards for "prefer_late"
        s_t = s.t[::sign]
        s_y = s.y[::sign]
        # We only keep those times that we don't have yet in the array times
        msk = s_t < times[-1] if prefer_late else s_t > times[-1]
        times = np.append(times, s_t[msk])
        values = np.append(values, s_y[msk])

    return TimeSeries(times[::sign], values[::sign])


def sample_common(series):
    """Resample a list of timeseries to the largest time interval covered
    by all timeseries, using regularly spaced time.

    The number of sample points is the minimum over all time series.

    :param ts: The timeseries to resample
    :type ts:  List of :py:class:`~.TimeSeries`

    :returns:  Resampled time series so that they are all defined in
               the same interval
    :rtype:    List of :py:class:`~.TimeSeries`

    """
    # Find the series with max tmin
    s_tmin = max(series, key=lambda x: x.tmin())
    # Find the series with min tmax
    s_tmax = min(series, key=lambda x: x.tmax())
    # Find the series with min number of points
    s_ns = min(series, key=len)
    t = np.linspace(s_tmin.tmin(), s_tmax.tmax(), len(s_ns))
    return [s.resampled(t) for s in series]


class TimeSeries:
    """This class represents real or complex valued time series.

    TimeSeries are defined providing a time list or array and the corresponding
    values. For example,

    .. code-block:: python

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        ts = TimeSeries(times, values)


    Times cannot be empty or not monotonically increasing.
    Times and values must have the same length.

    TimeSeries are well-behaved classed, many operations and methods are
    implemented. For instance, you can sum/multiply two Timeseries.

    numpy acts on TimeSeries cleanly, eg. ``np.log10(TimeSeries)`` is a
    TimeSeries with ``log10(data)``.

    TimeSeries have methods for smoothing, windowing, extracting phase and
    more.

    """

    # NOTE: Are you adding a function? Document it in timeseries.rst!

    def __init__(self, t, y):
        """Constructor.

        :param t: Sampling times, need to be strictly increasing
        :type t:  1D numpy array or list

        :param y: Data samples, can be real or complex valued
        :type y:  1D numpy array or list

        """
        # First, let's check if we have a scalar as input
        # In that case, we turn it into an array
        if (not hasattr(t, '__len__')):
            t = np.array([t])
            y = np.array([y])
        else:
            # Make sure these are arrays
            t = np.array(t)
            y = np.array(y)

        if (len(t) != len(y)):
            raise ValueError('Times and Values length mismatch')
        #
        if (len(t) == 0):
            raise ValueError('Trying to construct empty TimeSeries.')

        if (len(t) > 1):
            # Example:
            # self.t = [1,2,3]
            # self.t[1:] = [2, 3]
            # self.t[:-1] = [1, 2]
            # dt = [1,1]
            dt = t[1:] - t[:-1]
            if (dt.min() <= 0):
                raise ValueError('Time not monotonically increasing')

        # The copy is because we don't want to change the input values
        self.t = np.array(t).copy()
        self.y = np.array(y).copy()

        if (len(t) <= 3):
            warnings.warn('Spline will not be computed: too few points')
        else:
            self._make_spline()

    def _make_spline(self, *args, k=3, s=0, **kwargs):
        """Private function to make spline representation of the data.

        This function is not meant to be called directly.

        Values outside the interval are extrapolated if ext=0, set to 0 if
        ext=1, raise a ValueError if ext=2, or if ext=3, return the boundary
        value.

        k is the degree of the spline fit. It is recommended to use cubic
        splines. Even values of k should be avoided especially with small s
        values. 1 <= k <= 5

        make_spline is called at initialization and every time a modifying
        function is called (through _apply_to_self).

        :param k: Order of the spline representation
        :type k:  int
        :param s: Smoothing of the spline
        :type s:  float

        """
        if (len(self) < k):
            raise ValueError(
                f"Too few points to compute a spline of order {k}")

        self.spline_real = interpolate.splrep(self.t, self.y.real,
                                              k=k, s=s, *args, **kwargs)

        if (self.is_complex()):
            self.spline_imag = interpolate.splrep(self.t, self.y.imag,
                                                  k=k, s=s,
                                                  *args, **kwargs)

    def evaluate_with_spline(self, times, ext=2):
        """Evaluate the spline on the points times.

        Values outside the interval are extrapolated if ext=0, set to 0 if
        ext=1, raise a ValueError if ext=2, or if ext=3, return the boundary
        value.

        This method is meant to be used only if you want to use a different ext
        for a specific call, otherwise, just use __call__.

        :param times: Array of times where to evaluate the timeseries or single
                      time
        :type times: 1D numpy array of float

        :param ext: How to deal values outside the bounaries. Values outside
                    the interval are extrapolated if ext=0, set to 0 if ext=1,
                    raise a ValueError if ext=2, or if ext=3, return the
                    boundary value.
        :type ext:  bool

        :returns: Values of the timeseries evaluated on the input times
        :rtype:   1D numpy array or float

        """
        y_real = interpolate.splev(times, self.spline_real, ext=ext)
        if (self.is_complex()):
            y_imag = interpolate.splev(times, self.spline_imag, ext=ext)
            ret = y_real + 1j * y_imag
        else:
            ret = y_real

        # When this method is called with a scalar input, at this point, ret
        # would be a 0d numpy scalar array. What's that? - you may ask. I have
        # no idea, but the user is expecting a scalar as output. Hence, we cast
        # the 0d array into at "at_least_1d" array, then we can see its length
        # and act consequently
        ret = np.atleast_1d(ret)
        return ret if len(ret) > 1 else ret[0]

    def __call__(self, times):
        """Evaluate the spline on the points times. If the value is outside the
        range, a ValueError will be raised.
        """
        return self.evaluate_with_spline(times, ext=2)

    def __len__(self):
        """The number of time points."""
        return len(self.t)

    def tmin(self):
        """Return the starting time.

        :returns:  Initial time of the timeseries
        :rtype:    float
        """
        return self.t[0]

    def tmax(self):
        """Return the final time.

        :returns:  Final time of the timeseries
        :rtype:    float
        """
        return self.t[-1]

    def length(self):
        """Return the length of the covered time interval.

        :returns:  Length of time covered by the timeseries (tmax - tmin)
        :rtype:    float
        """
        return self.tmax() - self.tmin()

    def copy(self):
        """Return a deep copy.

        :returns:  Deep copy of the timeseries
        :rtype:    :py:class:`~.TimeSeries`
        """
        ret = TimeSeries(self.t, self.y)
        # spline-real and spline_imag are tuples. We add an empty tuple to
        # "make a deep copy" of those variables
        ret.spline_real = self.spline_real + ()
        if (self.is_complex()):
            ret.spline_imag = self.spline_imag + ()
        return ret

    def resampled(self, new_times, ext=2):
        """Return a new timeseries resampled from this to times new_times.

        You can specify the details of the spline with the method make_spline.

        :param new_times: New sample times.
        :type new_times:  1D numpy array or list of float.
        :param ext: How to handle points outside the time interval.
        :type ext: 0 for extrapolation, 1 for returning zero, 2 for ValueError,
                   3 for extending the boundary
        :returns: Resampled time series.
        :rtype:   :py:class:`~.TimeSeries`

        """
        return TimeSeries(new_times, self.evaluate_with_spline(new_times,
                                                               ext=ext))

    def resample(self, new_times, ext=2):
        """Resample the timeseries to new times tn using splines.

        :param new_times: New sample times.
        :type new_times:  1D numpy array or list of float.
        :param ext: How to handle points outside the time interval.
        :type ext: 0 for extrapolation, 1 for returning zero, 2 for ValueError,
                   3 for extending the boundary

        """
        self._apply_to_self(self.resampled, new_times, ext=ext)

    def regular_resampled(self):
        """Return a new timeseries resampled to regularly spaced times,
        with the
        same number of points.

        :returns: Regularly resampled time series
        :rtype:   :py:class:`~.TimeSeries`
        """
        t = np.linspace(self.tmin(), self.tmax(), len(self))
        return self.resampled(t)

    def regular_resample(self):
        """Resample the timeseries to regularly spaced times,
        with the same number of points.

        """
        self._apply_to_self(self.regular_resampled)

    def fixed_frequency_resampled(self, frequency):
        """Return a TimeSeries with same tmin and tmax but resampled at a fixed
        frequency.

        Tmax may vary if the frequency does not lead a integer number of
        timesteps.

        :param frequency: Sampling rate
        :type frequency: float
        :returns:  Time series resampled with given frequency
        :rtype:   :py:class:`~.TimeSeries`
        """
        dt = 1.0 / float(frequency)
        if (dt > self.length()):
            raise ValueError("Frequency too short for resampling")
        n = int(np.floor(self.length() / dt))
        # We have to add one to n, so that we can include the tmax point
        new_times = self.tmin() + np.arange(0, n + 1) * dt

        return self.resampled(new_times)

    def fixed_frequency_resample(self, frequency):
        """Resample the timeseries to regularly spaced times
        with given frequency.

        Tmax may vary if the frequency does not lead a integer number of
        timesteps.

        :param frequency: Sampling rate
        :type frequency: float
        """
        self._apply_to_self(self.fixed_frequency_resampled, frequency)

    def fixed_timestep_resample(self, timestep):
        """Resample the timeseries to regularly spaced times
        with given timestep.

        Tmax may vary if the timestep does not lead a integer number of
        timesteps.

        :param timestep: New timestep
        :type timestep: float
        :returns:  Time series resampled with given timestep
        :rtype:   :py:class:`~.TimeSeries`

        """
        self._apply_to_self(self.fixed_timestep_resampled, timestep)

    def fixed_timestep_resampled(self, timestep):
        if (timestep > self.length()):
            raise ValueError("Timestep larger then duration of the TimeSeries")
        frequency = 1.0 / float(timestep)
        return self.fixed_frequency_resampled(frequency)

    def __neg__(self):
        return TimeSeries(self.t, -self.y)

    def __abs__(self):
        return TimeSeries(self.t, np.abs(self.y))

    def _apply_binary(self, other, function):
        """This is an abstract function that is used to implement mathematical
        operations with other timeseries (if they have the same times) or
        scalars.

        _apply_binary takes another object that can be a
        TimeSeries or a scalar, and applies function(self.y, other.y),
        performing type checking.

        :param other: Other object
        :type other: :py:class:`~.TimeSeries` or float
        :param function: Dyadic function
        :type function: callable

        :returns:  Return value of function when called with self and ohter
        :rtype:   :py:class:`~.TimeSeries` (typically)


        """
        # If the other object is a TimeSeries
        if (isinstance(other, TimeSeries)):
            if ((not np.allclose(other.t, self.t))
                    or (len(self.t) != len(other.t))):
                raise ValueError(
                    "The twoTimeSeries do not have the same times!")
            return TimeSeries(self.t, function(self.y, other.y))
        # If it is a number
        elif isinstance(other, (int, float, complex)):
            return TimeSeries(self.t, function(self.y, other))

        # If we are here, it is because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")

    def __add__(self, other):
        """Add two timeseries (if they have the same t), or add a scalar to a
        timeseries.
        """
        return self._apply_binary(other, np.add)

    __radd__ = __add__

    def __sub__(self, other):
        """Subtract two timeseries (if they have the same t), or subtract a
        scalar from a timeseries.

        """
        return self._apply_binary(other, np.subtract)

    def __rsub__(self, other):
        """Subtract two timeseries (if they have the same t), or subtract a
        scalar from a timeseries.

        """
        return -self._apply_binary(other, np.subtract)

    def __mul__(self, other):
        """Multiply two timeseries (if they have the same t), or multiply
        by scalar.
        """
        return self._apply_binary(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide two timeseries (if they have the same t), or divide by a
        scalar.
        """
        if (other == 0):
            raise ValueError("Cannot divide by zero")
        return self._apply_binary(other, np.divide)

    def __pow__(self, other):
        """Raise the first timeseries to second timeseries (if they have the
        same t), or raise the timeseries to a scalar.

        """
        return self._apply_binary(other, np.power)

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other

    def __ipow__(self, other):
        return self ** other

    def __eq__(self, other):
        """Check for equality up to numerical precision.
        """
        if (isinstance(other, TimeSeries)):
            return (np.allclose(self.t, other.t)
                    and np.allclose(self.y, other.y))
        return False

    def _apply_to_self(self, f, *args, **kwargs):
        """Apply the method f to self, modifying self.
        This is used to transform the commands from returning an object
        to modifying self.
        """
        ret = f(*args, **kwargs)
        self.t, self.y = ret.t, ret.y
        self._make_spline()

    def is_complex(self):
        """Return whether the data is complex.

        :returns:  True if the data is complex, false if it is not
        :rtype:   bool

        """
        return issubclass(self.y.dtype.type, complex)

    def save(self, fname, *args, **kwargs):
        """Saves into simple ASCII format with 2 columns (t,y) for real valued
        data and 3 columns (t, Re(y), Im(y)) for complex valued data.

        :param fname: Path (with extensiton) of the output file
        :type fname: str

        """
        if self.is_complex():
            np.savetxt(fname, np.transpose((self.t, self.y.real, self.y.imag),
                                           *args, **kwargs))
        else:
            np.savetxt(fname, np.transpose((self.t, self.y),
                                           *args, **kwargs))

    def zero_padded(self, N):
        """Return a timeseries that is zero-padded and that has in total
        N points.

        This operation will work only if the timeseries is equispaced.

        :param N: Total number of points of the output timeseries
        :type N: int

        :returns: A new timeseries with in total N points where all
                  the trailing ones are zero
        :rtype: :py:class:`~.TimeSeries`
        """
        N_new_zeros = N - len(self)

        if (N_new_zeros < 0):
            raise ValueError(
                'Zero-padding cannot decrease the number of points')

        # Check that sequence is equispaced in time
        # by checking that all the elements are equal to the first
        dts = np.diff(self.t)
        dt0 = dts[0]

        if (not np.allclose(dts, dt0)):
            raise ValueError('Sequences not equispaced in time')

        new_zeros_t = np.linspace(self.tmax() + dt0,
                                  self.tmax() + N_new_zeros * dt0, N_new_zeros)
        return TimeSeries(np.append(self.t, new_zeros_t),
                          np.append(self.y, np.zeros(N_new_zeros)))

    def zero_pad(self, N):
        """Pad the timeseries with zeros so that it has a total of N points.

        This operation will work only if the timeseries is equispaced and if N
        is larger than the number of points already present.

        :param N: Total number new points with zeros at the end
        :type N: int

        """
        self._apply_to_self(self.zero_padded, N)

    def mean_removed(self):
        """Return a timeseries with mean removed.

        :returns: A new timeseries zero mean
        :rtype: :py:class:`~.TimeSeries`
        """
        return TimeSeries(self.t, self.y - self.y.mean())

    def mean_remove(self):
        """Remove the mean value from the data."""
        self._apply_to_self(self.mean_removed)

    def nans_removed(self):
        """Filter out nans/infinite values.
        Return a Time series with finite values only.

        :returns: A new timeseries with only finite values
        :rtype: :py:class:`~.TimeSeries`
        """
        msk = np.isfinite(self.y)
        return TimeSeries(self.t[msk], self.y[msk])

    def nans_remove(self):
        """Filter out nans/infinite values."""
        self._apply_to_self(self.nans_removed)

    def time_shifted(self, tshift):
        """Return a new timeseries with time shifted by tshift (what was t = 0
        will be tshift).

        :param tshift: Amount of time to shift
        :type tshift: float

        :returns: A new timeseries with time shifted
        :rtype: :py:class:`~.TimeSeries`
        """
        return TimeSeries(self.t + tshift, self.y)

    def time_shift(self, tshift):
        """Shift the timeseries by tshift (what was t = 0 will be tshift).

        :param N: Amount of time to shift
        :type N: float

        """
        self._apply_to_self(self.time_shifted, tshift)

    def phase_shifted(self, pshift):
        """Return a new timeseries with complex phase shifted by pshift. If the
        signal is real, it is turned complex with phase of pshift.

        :param pshift: Amount of phase to shift
        :type pshift: float

        :returns: A new timeseries with phase shifted
        :rtype: :py:class:`~.TimeSeries`
        """
        return TimeSeries(self.t, self.y * np.exp(1j * pshift))

    def phase_shift(self, pshift):
        """Shift the complex phase timeseries by pshift. If the signal is real,
        it is turned complex with phase of pshift.

        :param pshift: Amount of phase to shift
        :type pshift: float

        """
        self._apply_to_self(self.phase_shifted, pshift)

    def time_unit_changed(self, unit, inverse=False):
        """Return a new timeseries with time scaled by unit.

        When inverse is False, t -> t/unit. For example, if initially the
        units where seconds, with unit=1e-3 the new units will be milliseconds.

        When inverse is True, t -> t * unit. This is useful to convert
        geometrized units to physical units with unitconv.
        For example,

        .. code-block:: python

            # Gravitational waves in geometrized units
            gw_cu = TimeSeries(...)
            # Gravitational waves in seconds, assuming a mass of 1 M_sun
            CU = uc.geom_umass_msun(1)
            gw_s = gw_cu.time_unit_changed(CU.time, inverse=True)

        :param unit: New time unit
        :type unit: float
        :param inverse: If True, time = 1 -> time = unit, otherwise
                        time = unit -> 1
        :type inverse: bool

        :returns: A timeseries with new time unit
        :rtype: :py:class:`~.TimeSeries`

        """
        factor = unit if inverse else 1/unit
        return TimeSeries(self.t * factor, self.y)

    def time_unit_change(self, unit, inverse=False):
        """Rescale time units by unit.

        When inverse is False, t -> t/unit. For example, if initially the
        units where seconds, with unit=1e-3 the new units will be milliseconds.

        When inverse is True, t -> t * unit. This is useful to convert
        geometrized units to physical units with unitconv.
        For example,

        .. code-block:: python

            # Gravitational waves in geometrized units
            gw_cu = TimeSeries(...)
            # Gravitational waves in seconds, assuming a mass of 1 M_sun
            CU = uc.geom_umass_msun(1)
            gw_cu.time_unit_change(CU.time, inverse=True)

        :param unit: New time unit
        :type unit: float
        :param inverse: If True, time = 1 -> time = unit, otherwise
                        time = unit -> 1
        :type inverse: bool

        """
        self._apply_to_self(self.time_unit_changed, unit, inverse)

    def redshifted(self, z):
        """Return a new timeseries with time rescaled so that frequencies are
        redshited by 1 + z.

        :param z: Redshift factor
        :type z: float

        :returns: A new redshifted timeseries
        :rtype: :py:class:`~.TimeSeries`

        """
        return self.time_unit_changed(1 + z, inverse=True)

    def redshift(self, z):
        """Apply redshift to the data by rescaling the time.

        :param z: Redshift factor

        """
        self._apply_to_self(self.redshifted, z)

    def unfolded_phase(self):
        """Compute the complex phase of a complex-valued signal such that
        no phase wrap-arounds occur, i.e. if the input is continous, so is
        the output.

        :returns:   Continuous complex phase
        :rtype:     :py:class:`~.TimeSeries`
        """

        return TimeSeries(self.t, unfold_phase(np.angle(self.y)))

    def phase_angular_velocity(self, use_splines=True, tsmooth=None, order=3):
        """Compute the phase angular velocity, i.e. the time derivative of the
        complex phase.

        Optionally smooth the with a savgol filter with smoothing length
        tsmooth and order order. If you do so, the timeseries is resampled to
        regular timesteps.

        :param use_splines: Wheter to use splines of finite differencing for
                            the derivative
        :type use_splines: bool
        :param tsmooth: Time over which smoothing is applied
        :type tsmooth: float
        :param order: Order of the for the savgol smoothing
        :type order: int

        :returns:  Time derivative of the complex phase
        :rtype:    :py:class:`~.TimeSeries`
        """
        if use_splines:
            ret_value = self.unfolded_phase().spline_derived()
        else:
            ret_value = self.unfolded_phase().derived()

        if tsmooth is not None:
            ret_value.savgol_smooth_time(tsmooth, order)

        return ret_value

    def phase_frequency(self, use_splines=True, tsmooth=None, order=3):
        """Compute the phase frequency, i.e. the time derivative
        of the complex phase divided by 2 pi.

        Optionally smooth the with a savgol filter with smoothing length
        tsmooth and order order. If you do so, the timeseries is resampled
        to regular timesteps.

        :param use_splines: Wheter to use splines of finite differencing for
                            the derivative
        :type use_splines: bool
        :param tsmooth: Time over which smoothing is applied
        :type tsmooth: float
        :param order: Order of the for the savgol smoothing
        :type order: int

        :returns:  Time derivative of the complex phase divided by 2 pi
        :rtype:    :py:class:`~.TimeSeries`
        """
        return self.phase_angular_velocity(use_splines,
                                           tsmooth, order) / (2 * np.pi)

    def cropped(self, tmin=None, tmax=None):
        """Return a timeseriews with data removed outside the time intarval
        [tmin, tmax]. if tmin or tmax are not specified or None, it does not
        remove anything from this side.

        :param tmin: New minimum time
        :type tmin: float
        :param tmax: New maximum time
        :type tmax: float

        :returns:  New timeseries with new time limits
        :rtype:    :py:class:`~.TimeSeries`
        """
        t = self.t
        y = self.y
        if (tmin is not None):
            m = (t >= tmin)
            t = t[m]
            y = y[m]
        if (tmax is not None):
            m = (t <= tmax)
            t = t[m]
            y = y[m]
        return TimeSeries(t, y)

    def crop(self, tmin=None, tmax=None):
        """Throw away data outside the time intarval [tmin, tmax].
        if tmin or tmax are not specified or None, it does not remove
        anything from this side.

        :param tmin: New minimum time
        :type tmin: float
        :param tmax: New maximum time
        :type tmax: float

        """
        self._apply_to_self(self.cropped, tmin, tmax)

    # Define aliases
    clip = crop
    clipped = cropped

    def integrated(self):
        """Return a timeseries that is the integral computed with method of
        the trapeziod.

        :returns:  New timeseries cumulative integral
        :rtype:    :py:class:`~.TimeSeries`
        """
        return TimeSeries(self.t,
                          integrate.cumtrapz(self.y, x=self.t, initial=0))

    def integrate(self):
        """Integrate timeseries with method of the trapeziod.
        """
        self._apply_to_self(self.integrated)

    def spline_derived(self, order=1):
        """Return a timeseries that is the derivative of the current one using
        the spline interpolation. You shouldn't trust the values at the
        boundaries too much, you may want to crop it out.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int

        :returns:  New timeseries with derivative
        :rtype:    :py:class:`~.TimeSeries`

        """
        if ((order > 3) or (order < 0)):
            raise ValueError(f'Cannot compute differential of order {order}')

        if (self.is_complex()):
            ret_value = (interpolate.splev(self.t,
                                           self.spline_real,
                                           der=order)
                         + 1j * interpolate.splev(self.t,
                                                  self.spline_imag,
                                                  der=order))
        else:
            ret_value = interpolate.splev(self.t, self.spline_real, der=order)

        return TimeSeries(self.t, ret_value)

    def spline_derive(self, order=1):
        """Derive the timeseries current one using the spline interpolation.
        To keep the timeseries of the same size as the original one, the value
        of the derivative at the boundaries is set to zero. Don't trust it!

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int

        """
        self._apply_to_self(self.spline_derived, order)

    def derived(self, order=1):
        """Return a timeseries that is the numerical order-differentiation of
        the present timeseries. (order = number of derivatives, ie order=2 is
        second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int

        :returns:  New timeseries with derivative
        :rtype:    :py:class:`~.TimeSeries`

        """
        ret_value = self.y
        for _num_deriv in range(order):
            ret_value = np.gradient(ret_value, self.t)
        return TimeSeries(self.t, ret_value)

    def derive(self, order=1):
        """Derive with the numerical order-differentiation. (order = number of
        derivatives, ie order=2 is second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int

        """
        self._apply_to_self(self.derived, order)

    def windowed(self, window_function, *args, **kwargs):
        """Return a timeseries windowed with window_function.

        ``window_function`` has to be a function that takes as first argument
        the number of points of the signal. window_function can take additional
        arguments as passed by windowed.

        :param window: Window function to apply to the timeseries
        :type window: callable

        :returns:  New windowed timeseries
        :rtype:    :py:class:`~.TimeSeries`

        """
        window_array = window_function(len(self), *args, **kwargs)
        return TimeSeries(self.t, self.y * window_array)

    def window(self, window_function, *args, **kwargs):
        """Apply window_function to the data.

        ``window_function`` has to be a function that takes as first argument
        the number of points of the signal. window_function can take additional
        arguments as passed by windowed.

        :param window: Window function to apply to the timeseries
        :type window: callable

        """
        self._apply_to_self(self.windowed, window_function,
                            *args, **kwargs)

    def tukey_windowed(self, alpha):
        """Return a timeseries with Tukey window with paramter alpha applied.

        :param alpha: Tukey parameter
        :type alpha: float

        :returns:  New windowed timeseries
        :rtype:    :py:class:`~.TimeSeries`

        """
        return self.windowed(signal.tukey, alpha)

    def tukey_window(self, alpha):
        """Apply Tukey window.

        :param alpha: Tukey parameter
        :type alpha: float

        """
        self.window(signal.tukey, alpha)

    def hamming_windowed(self):
        """Return a timeseries with Hamming window applied.

        :returns:  New windowed timeseries
        :rtype:    :py:class:`~.TimeSeries`

        """
        return self.windowed(signal.hamming)

    def hamming_window(self):
        """Apply Hamming window.

        """
        self.window(signal.hamming)

    def blackman_windowed(self):
        """Return a timeseries with Blackman window applied.

        """
        return self.windowed(signal.blackman)

    def blackman_window(self):
        """Apply Blackman window.

        """
        self.window(signal.blackman)

    def savgol_smoothed(self, window_size, order=3):
        """Return a smoothed timeseries with a Savitzky-Golay filter with
        window of size WINDOW-SIZE and order ORDER.

        This is just like a regular "Moving average" filter, but instead of
        just calculating the average, a polynomial (usually 2nd or 4th order)
        fit is made for every point, and only the "middle" point is chosen.
        Since 2nd (or 4th) order information is concerned at every point, the
        bias introduced in "moving average" approach at local maxima or minima,
        is circumvented.

        :param window_size: Number of points of the smoothing window (need to
                            be odd)
        :type window_size: int
        :param order: Order of the filter
        :type order: int

        :returns:  New smoothed timeseries
        :rtype:    :py:class:`~.TimeSeries`

        """
        if self.is_complex():
            return TimeSeries(self.t,
                              signal.savgol_filter(self.y.imag,
                                                   window_size,
                                                   order)
                              + 1j * signal.savgol_filter(self.y.real,
                                                          window_size,
                                                          order))

        return TimeSeries(self.t, signal.savgol_filter(self.y, window_size,
                                                       order))

    def savgol_smooth(self, window_size, order=3):
        """Smooth the timeseries with a Savitzky-Golay filter with window of
        size WINDOW-SIZE and order ORDER.

        This is just like a regular "Moving average" filter, but instead of
        just calculating the average, a polynomial (usually 2nd or 4th order)
        fit is made for every point, and only the "middle" point is chosen.
        Since 2nd (or 4th) order information is concerned at every point, the
        bias introduced in "moving average" approach at local maxima or minima,
        is circumvented.

        :param window_size: Number of points of the smoothing window (need to
                            be odd)
        :type window_size: int
        :param order: Order of the filter
        :type order: int

        """
        self._apply_to_self(self.savgol_smoothed, window_size, order)

    def savgol_smoothed_time(self, tsmooth, order=3):
        """Return a resampled timeseries with uniform timesteps, smoothed it
        with savgol_smooth with a window that is tsmooth in time (as opposed
        to a number of points).

        :param tsmooth: Time interval over which to smooth
        :type tsmooth: float
        :param order: Order of the filter
        :type order: int

        :returns:  New smoothed and resampled timeseries
        :rtype:    :py:class:`~.TimeSeries`

        """
        ts = self.regular_resampled()
        dt = ts.t[1] - ts.t[0]
        # The savgol method requires a odd window
        # If it is not, we add one point
        window = int(np.rint(tsmooth / dt))
        window = window + 1 if (window % 2 == 0) else window
        return self.savgol_smoothed(window, order)

    def savgol_smooth_time(self, tsmooth, order=3):
        """Resampl the timeseries with uniform timesteps, smooth it with
        savgol_smooth with a window that is tsmooth in time (as opposed to a
        number of points).

        :param tsmooth: Time interval over which to smooth
        :type tsmooth: float
        :param order: Order of the filter
        :type order: int

        """
        self._apply_to_self(self.savgol_smoothed_time, tsmooth, order)

    def _apply_unary(self, function):
        """Apply a unary function to the data.

        :param function: Function to apply to the timeseries
        :type function: callable

        :return: New timeseries with function applied to the data
        :rtype: :py:class:`~.TimeSeries`

        """
        return TimeSeries(self.t, function(self.y))

    def abs(self):
        return abs(self)

    def real(self):
        return self._apply_unary(np.real)

    def imag(self):
        return self._apply_unary(np.imag)

    def sin(self):
        return self._apply_unary(np.sin)

    def cos(self):
        return self._apply_unary(np.cos)

    def tan(self):
        return self._apply_unary(np.tan)

    def arcsin(self):
        return self._apply_unary(np.arcsin)

    def arccos(self):
        return self._apply_unary(np.arccos)

    def arctan(self):
        return self._apply_unary(np.arctan)

    def sinh(self):
        return self._apply_unary(np.sinh)

    def cosh(self):
        return self._apply_unary(np.cosh)

    def tanh(self):
        return self._apply_unary(np.tanh)

    def arcsinh(self):
        return self._apply_unary(np.arcsinh)

    def arccosh(self):
        return self._apply_unary(np.arccosh)

    def arctanh(self):
        return self._apply_unary(np.arctanh)

    def sqrt(self):
        return self._apply_unary(np.sqrt)

    def exp(self):
        return self._apply_unary(np.exp)

    def log(self):
        return self._apply_unary(np.log)

    def log10(self):
        return self._apply_unary(np.log10)

    def conjugate(self):
        return self._apply_unary(np.conjugate)
