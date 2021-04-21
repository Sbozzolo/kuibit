#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
#
# Based on by code originally developed by Wolfgang Kastaun. See, GitHub,
# wokast/PyCactus/PostCactus/timeseries.py
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

"""The :py:mod:`~.timeseries` module provides a representation of time series
and convenience functions to create :py:class:`~.TimeSeries`.

:py:class:`~.TimeSeries` can be evenly or unevenly sampled are rich in features.
They support all the mathematical operations and operators you may expect, and
have additional methods, which include ones for taking derivatives, integrals,
apply windows, smooth the signal, take Fourier transform, and more. Most of
these methods are available in two flavors: those that return a new
:py:class:`~.TimeSeries`, and those which modify the object in place. The latter
have names with imperative verbs.

:py:class:`~.TimeSeries` are derived from the :py:class:`~.BaseSeries`, which in
turn is derived from the abstract class :py:class:`~.BaseNumerical`. Some of the
capabilities of :py:class:`~.TimeSeries` (e.g., overloading the mathematical
operators) are implemented in the parent classes.

The additional functions provided in :py:mod:`~.timeseries` are:

- :py:func:`~.remove_duplicated_iters` cleans the input arrays by removing duplicated times.
- :py:func:`~.unfold_phase` takes as argument a NumPy array representing a phase
  and unfolds it removing all the jumps of 2 pi. This is useful in gravitational
  wave analysis.
- :py:func:`~.combine_ts` takes a list of timeseries and removes all the overlapping segments.


"""

import warnings

import numpy as np
from scipy import signal

from kuibit import frequencyseries
from kuibit.series import BaseSeries


def remove_duplicated_iters(t, y):
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

    :param t:  Times.
    :type t:   1D NumPy array
    :param y:  Values.
    :type t:   1D NumPy array

    :returns:  Strictly monotonic time series.
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
    """Remove phase jumps to get a continuous (unfolded) phase.

    :param phase:     Phase wrapped around the provided jump.
    :type phase:      1D NumPy array

    :returns:         Phase plus multiples of pi chosen to minimize jumps.
    :rtype:           1D NumPy array
    """
    return np.unwrap(phase)


def combine_ts(series, prefer_late=True):
    """Combine several overlapping time series into one.

    In intervals covered by two or more time series, which data is used depends
    on the parameter prefer_late. If two segments start at the same time, the
    longer one gets used.

    :param series: The timeseries to combine.
    :type series:  list of :py:class:`~.TimeSeries`
    :param prefer_late: If true, prefer data that starts later for overlapping
                        segments, otherwise, use data from the ones that come
                        earlier.
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
    timeseries = sorted(series, key=lambda x: (sign * x.tmin, sign * x.tmax))
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


class TimeSeries(BaseSeries):
    """This class represents real or complex valued time series.

    :py:class:`~.TimeSeries` are defined providing a time list or array and the
    corresponding values. For example,

    .. code-block:: python

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        ts = TimeSeries(times, values)


    Times cannot be empty or not monotonically increasing.
    Times and values must have the same length.

    TimeSeries are well-behaved classed, many operations and methods are
    implemented. For instance, you can sum/multiply two :py:class:`~.TimeSeries`.

    NumPy acts on TimeSeries cleanly, eg. ``np.log10(TimeSeries)`` is a
    :py:class:`~.TimeSeries` with ``log10(data)``.

    :py:class:`~.TimeSeries` have methods for smoothing, windowing, extracting phase and
    more.

    :ivar t: Times.
    :vartype t: 1D NumPy array or float
    :ivar y: Values.
    :vartype y: 1D NumPy array or float
    :ivar spline_real: Coefficients for a spline represent of the real part
                       of y.
    :vartype spline_real: tuple
    :ivar spline_imag: Coefficients for a spline represent of the real part
                       of y.
    :vartype spline_imag: tuple

    """

    # NOTE: Are you adding a function? Document it in timeseries.rst!

    def __init__(self, t, y, guarantee_t_is_monotonic=False):
        """Constructor.

        When guarantee_t_is_monotonic is True no checks will be perform to make
        sure that t is monotonically increasing (increasing performance). This
        should is used internally whenever a new series is returned from self
        (since we have already checked that t is good.) or in performance
        critical routines.

        :param t: Sampling times, need to be strictly increasing.
        :type t:  1D NumPy array or list

        :param y: Data samples, can be real or complex valued.
        :type y:  1D NumPy array or list

        :param guarantee_t_is_monotonic: The code will assume that t is
                                         monotonically increasing.
        :type guarantee_t_is_monotonic: bool

        """
        # Use BaseClass init
        super().__init__(t, y, guarantee_t_is_monotonic)

    # The following are the setters and getters, so that we can "resolve" .t
    # and .y

    # The @property decorator allows us to call .t instead of .t()
    @property
    def t(self):
        """Return the time.

        :returns: Times.
        :rtype: 1d NumPy array.
        """
        # This is defined BaseClass and it is where the actual data is stored.
        return self.x

    @t.setter
    def t(self, t):
        # This is defined BaseClass
        self.x = t

    @property
    def tmin(self):
        """Return the starting time.

        :returns:  Initial time of the timeseries.
        :rtype:    float
        """
        return self.xmin

    @property
    def tmax(self):
        """Return the final time.

        :returns:  Final time of the timeseries.
        :rtype:    float
        """
        return self.xmax

    @property
    def dt(self):
        """Return the timestep if the series is regularly sampled,
        otherwise raise error.

        :returns: Timestep of the series (if evenly sampled).
        :rtype: float

        """
        if not self.is_regularly_sampled():
            raise ValueError("Timeseries is not regularly sampled")

        return self.t[1] - self.t[0]

    @property
    def time_length(self):
        """Return the length of the covered time interval.

        :returns:  Length of time covered by the timeseries (tmax - tmin).
        :rtype:    float
        """
        return self.tmax - self.tmin

    duration = time_length

    def time_at_maximum(self):
        """Return the time at which the timeseries is maximum in absolute
        value.

        :returns:  Time at absolute maximum.
        :rtype:    float
        """
        return self.x_at_abs_maximum_y()

    def time_at_minimum(self):
        """Return the time at which the timeseries is minimum in absolute
        value.

        :returns:  Time at absolute minimum.
        :rtype:    float
        """
        return self.x_at_abs_minimum_y()

    def aligned_at_minimum(self):
        """Return a new timeseries with absolute minimum at t=0.

        :returns:  Timeseries shifted so that the minimum is a t=0.
        :rtype:    :py:class:`~.TimeSeries`
        """
        return self.time_shifted(-self.time_at_minimum())

    def align_at_minimum(self):
        """Time shift the series so that the absolute minimum is at t=0."""
        self._apply_to_self(self.aligned_at_minimum)

    def aligned_at_maximum(self):
        """Return a new timeseries with absolute maximum at t=0.

        :returns:  Timeseries shifted so that the maximum is a t=0.
        :rtype:    :py:class:`~.TimeSeries`
        """
        return self.time_shifted(-self.time_at_maximum())

    def align_at_maximum(self):
        """Time shift the series so that the absolute maximum is at t=0."""
        self._apply_to_self(self.aligned_at_maximum)

    def regular_resampled(self):
        """Return a new timeseries resampled to regularly spaced times,
        with the same number of points.

        :returns: Regularly resampled time series.
        :rtype:   :py:class:`~.TimeSeries`
        """
        t = np.linspace(self.tmin, self.tmax, len(self))
        return self.resampled(t)

    def regular_resample(self):
        """Resample the timeseries to regularly spaced times,
        with the same number of points.

        """
        self._apply_to_self(self.regular_resampled)

    def fixed_frequency_resampled(self, frequency):
        """Return a  :py:class:`~.TimeSeries` with same tmin and tmax
        but resampled at a fixed frequency. The final time will change
        if the frequency does not lead a integer number of timesteps.

        :param frequency: Sampling rate.
        :type frequency: float
        :returns:  Time series resampled with given frequency.
        :rtype:   :py:class:`~.TimeSeries`
        """
        dt = 1.0 / float(frequency)
        if dt > self.time_length:
            raise ValueError("Frequency too short for resampling")
        n = int(np.floor(self.time_length / dt))
        # We have to add one to n, so that we can include the tmax point
        new_times = self.tmin + np.arange(0, n + 1) * dt

        return self.resampled(new_times)

    def fixed_frequency_resample(self, frequency):
        """Resample the timeseries to regularly spaced times with the given frequency.
        The final time will change if the frequency does not lead a integer
        number of timesteps.

        :param frequency: Sampling rate.
        :type frequency: float

        """
        self._apply_to_self(self.fixed_frequency_resampled, frequency)

    def fixed_timestep_resample(self, timestep):
        """Resample the timeseries to regularly spaced times with given timestep.
        The final time will change if the timestep does not lead a integer
        number of timesteps.

        :param timestep: New timestep.
        :type timestep: float

        """
        self._apply_to_self(self.fixed_timestep_resampled, timestep)

    def fixed_timestep_resampled(self, timestep):
        """Return a new :py:class:`~.TimeSeries` with evenly spaced with the given
        timestep. The final time will change if the timestep does not lead a
        integer number of timesteps.

        :param timestep: New timestep.
        :type timestep: float
        :returns:  Time series resampled with given timestep.
        :rtype:   :py:class:`~.TimeSeries`

        """
        if timestep > self.time_length:
            raise ValueError("Timestep larger then duration of the TimeSeries")
        frequency = 1.0 / float(timestep)
        return self.fixed_frequency_resampled(frequency)

    def zero_padded(self, N):
        """Return a :py:class:`~.TimeSeries` that is zero-padded and that has
        in total ``N`` points.

        .. note::

            ``N`` is the final number of points, not the number of points added.

        This operation will work only if the series is equispaced.

        :param N: Total number of points of the output.
        :type N: int

        :returns: A new timeseries with in total N points where all
                  the trailing ones are zero.
        :rtype: :py:class:`~.TimeSeries`
        """
        N_new_zeros = N - len(self)

        if N_new_zeros < 0:
            raise ValueError(
                "Zero-padding cannot decrease the number of points"
            )

        new_zeros_t = np.linspace(
            self.tmax + self.dt,
            self.tmax + N_new_zeros * self.dt,
            N_new_zeros,
        )
        return TimeSeries(
            np.append(self.t, new_zeros_t),
            np.append(self.y, np.zeros(N_new_zeros)),
        )

    def zero_pad(self, N):
        """Pad the timeseries with zeros so that it has a total of N points.

        This operation will work only if the timeseries is equispaced and if N
        is larger than the number of points already present.

        .. note::

            ``N`` is the final number of points, not the number of points added.

        :param N: Total number new points with zeros at the end.
        :type N: int

        """
        self._apply_to_self(self.zero_padded, N)

    def mean_removed(self):
        """Return a :py:class:`~.TimeSeries` with mean removed, so that its new
        total average is zero.

        :returns: A new :py:class:`~.TimeSeries` with zero mean.
        :rtype: :py:class:`~.TimeSeries`
        """
        return TimeSeries(self.t, self.y - self.y.mean())

    def mean_remove(self):
        """Remove the mean value from the data."""
        self._apply_to_self(self.mean_removed)

    def initial_time_removed(self, time_init):
        """Return a :py:class:`~.TimeSeries` without the initial ``time_init`` amount of
        time.

        When ``tmin = 0``, this is the same as cropping, otherwise the
        difference is that in one case the time interval is specified, whereas
        in the other (cropping) the new ``tmin`` is specified.

        If a series goes from t=-1 to t=10 and you set time_init=2,
        the series will go from t=1 to t=10.

        :param time_init: Amount of time to be removed from the beginning.
        :type time_init: float

        :returns: A new :py:class:`~.TimeSeries` without the initial ``time_init``.
        :rtype: :py:class:`~.TimeSeries`

        """
        return self.cropped(init=self.tmin + time_init)

    def initial_time_remove(self, time_init):
        """Remove the first ``time_init`` amount of time in the timeseries.

        When ``tmin = 0``, this is the same as cropping, otherwise the
        difference is that in one case the time interval is specified, whereas
        in the other (cropping) the new ``tmin`` is specified.

        :param time_init: Amount of time to be removed from the beginning.
        :type time_init: float

        """
        self._apply_to_self(self.initial_time_removed, time_init)

    def final_time_removed(self, time_end):
        """Return a :py:class:`~.TimeSeries` without the final ``time_end`` amount of
        time.

        If a series goes from t=-1 to t=10 and you set time_end=2,
        the series will go from t=-1 to t=8.

        :param time_end: Amount of time to be removed from the end.
        :type time_end: float

        :returns: A new :py:class:`~.TimeSeries` without the final ``time_end``.
        :rtype: :py:class:`~.TimeSeries`

        """
        return self.cropped(end=self.tmax - time_end)

    def final_time_remove(self, time_end):
        """Remove the final ``time_end`` amount of time in the timeseries.

        :param time_end: Amount of time to be removed from the end.
        :type time_end: float

        """
        self._apply_to_self(self.final_time_removed, time_end)

    def time_shifted(self, tshift):
        """Return a new timeseries with time shifted by ``tshift`` so that
        what was t = 0 will be ``tshift``.

        :param tshift: Amount of time to shift.
        :type tshift: float
        :returns: A new :py:class:`~.TimeSeries` with time shifted.
        :rtype: :py:class:`~.TimeSeries`
        """
        return TimeSeries(self.t + tshift, self.y)

    def time_shift(self, tshift):
        """Shift the timeseries by ``tshift`` so that what was t = 0 will be ``tshift``.

        :param N: Amount of time to shift.
        :type N: float

        """
        self._apply_to_self(self.time_shifted, tshift)

    def phase_shifted(self, pshift):
        """Return a new :py:class:`~.TimeSeries` with complex phase shifted by ``pshift``.
        If the signal is real, it is turned complex with phase of ``pshift``.

        :param pshift: Amount of phase to shift.
        :type pshift: float
        :returns: A new :py:class:`~.TimeSeries` with phase shifted.
        :rtype: :py:class:`~.TimeSeries`

        """
        return TimeSeries(self.t, self.y * np.exp(1j * pshift))

    def phase_shift(self, pshift):
        """Shift the complex phase timeseries by ``pshift``. If the signal is real,
        it is turned complex with phase of ``pshift``.

        :param pshift: Amount of phase to shift.
        :type pshift: float

        """
        self._apply_to_self(self.phase_shifted, pshift)

    def time_unit_changed(self, unit, inverse=False):
        """Return a new :py:class:`~.TimeSeries` with time scaled by ``unit``.

        This amounts to sending t to ``t / unit``. For example, if initially the
        units where seconds, with unit=1e-3 the new units will be milliseconds.

        When inverse is True, the opposite is done and t is sent to ``t *
        unit``. This is useful to convert geometrized units to physical units
        with :py:mod:`~.unitconv`. For example,

        .. code-block:: python

            # Gravitational waves in geometrized units
            gw_cu = TimeSeries(...)
            # Gravitational waves in seconds, assuming a mass of 1 M_sun
            CU = uc.geom_umass_msun(1)
            gw_s = gw_cu.time_unit_changed(CU.time, inverse=True)

        :param unit: New time unit.
        :type unit: float
        :param inverse: If True, time = 1 -> time = unit, otherwise
                        time = unit -> 1.
        :type inverse: bool

        :returns: A :py:class:`~.TimeSeries` with new time unit.
        :rtype: :py:class:`~.TimeSeries`

        """
        factor = unit if inverse else 1 / unit
        return TimeSeries(self.t * factor, self.y)

    def time_unit_change(self, unit, inverse=False):
        """Rescale time units by unit.

        This amounts to sending t to ``t / unit``. For example, if initially the
        units where seconds, with unit=1e-3 the new units will be milliseconds.

        When inverse is True, the opposite is done and t is sent to ``t *
        unit``. This is useful to convert geometrized units to physical units
        with :py:mod:`~.unitconv`. For example,

        .. code-block:: python

            # Gravitational waves in geometrized units
            gw_cu = TimeSeries(...)
            # Gravitational waves in seconds, assuming a mass of 1 M_sun
            CU = uc.geom_umass_msun(1)
            gw_s = gw_cu.time_unit_changed(CU.time, inverse=True)

        :param unit: New time unit.
        :type unit: float
        :param inverse: If True, time = 1 -> time = unit, otherwise
                        time = unit -> 1.
        :type inverse: bool
        """
        self._apply_to_self(self.time_unit_changed, unit, inverse)

    def redshifted(self, z):
        """Return a new :py:class:`~.TimeSeries` with time rescaled so that frequencies
        are redshifted by ``1 + z``.

        :param z: Redshift factor.
        :type z: float

        :returns: A new redshifted :py:class:`~.TimeSeries`.
        :rtype: :py:class:`~.TimeSeries`

        """
        return self.time_unit_changed(1 + z, inverse=True)

    def redshift(self, z):
        """Apply redshift to the data by rescaling the time so that the frequencies
        are redshifted by ``1 + z``.

        :param z: Redshift factor.
        :type z: float

        """
        self._apply_to_self(self.redshifted, z)

    def unfolded_phase(self, t_of_zero_phase=None):
        """Compute the complex phase of a complex-valued signal such that no phase
        wrap-around occur, i.e. if the input is continuous, so is the output.
        Optionally, add a phase shift such that phase is zero at the given time.

        :param t_of_zero_phase: Time at which the phase is set to zero.
        :type t_of_zero_phase:   float or None

        :returns:   Continuous complex phase.
        :rtype:     :py:class:`~.TimeSeries`

        """
        ret = TimeSeries(self.t, unfold_phase(np.angle(self.y)))
        if t_of_zero_phase is not None:
            ret -= ret(t_of_zero_phase)
        return ret

    def phase_angular_velocity(self, use_splines=True, tsmooth=None, order=3):
        """Compute the phase angular velocity, i.e. the time derivative of the
        complex phase.

        Optionally smooth the with a savgol filter with smoothing length
        tsmooth and order order. If you do so, the timeseries is resampled to
        regular timesteps.

        :param use_splines: Wheter to use splines of finite differencing for
                            the derivative.
        :type use_splines: bool
        :param tsmooth: Time over which smoothing is applied.
        :type tsmooth: float
        :param order: Order of the for the savgol smoothing.
        :type order: int

        :returns:  Time derivative of the complex phase.
        :rtype:    :py:class:`~.TimeSeries`
        """
        if use_splines:
            ret_value = self.unfolded_phase().spline_differentiated()
        else:
            ret_value = self.unfolded_phase().differentiated()

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
                            the derivative.
        :type use_splines: bool
        :param tsmooth: Time over which smoothing is applied.
        :type tsmooth: float
        :param order: Order of the for the savgol smoothing.
        :type order: int

        :returns:  Time derivative of the complex phase divided by 2 pi
        :rtype:    :py:class:`~.TimeSeries`
        """
        return self.phase_angular_velocity(use_splines, tsmooth, order) / (
            2 * np.pi
        )

    def windowed(self, window_function, *args, **kwargs):
        """Return a :py:class:`~.TimeSeries` windowed with ``window_function``.

        ``window_function`` has to be a function that takes as first argument
        the number of points of the signal. ``window_function`` can take
        additional arguments as passed by ``windowed``. Alternatively,
        ``window_function`` can be a string with the name of the window
        function, if this is already implemented in :py:class:`~.TimeSeries`
        (e.g., ``tukey``).

        :param window_function: Window function to apply to the timeseries.
        :type window_function: callable or str

        :returns:  New windowed :py:class:`~.TimeSeries`.
        :rtype:    :py:class:`~.TimeSeries`

        """

        if callable(window_function):
            window_array = window_function(len(self), *args, **kwargs)
            return TimeSeries(self.t, self.y * window_array)

        if isinstance(window_function, str):
            window_function_method = f"{window_function}_windowed"
            if not hasattr(self, window_function_method):
                raise ValueError(f"Window {window_function} not implemented")
            window_function_callable = getattr(self, window_function_method)
            return window_function_callable(*args, **kwargs)

        raise TypeError("Window function is neither a callable or a string")

    def window(self, window_function, *args, **kwargs):
        """Apply window_function to the data.

        ``window_function`` has to be a function that takes as first argument
        the number of points of the signal. ``window_function`` can take
        additional arguments as passed by ``windowed``. Alternatively,
        ``window_function`` can be a string with the name of the window
        function, if this is already implemented in :py:class:`~.TimeSeries`
        (e.g., ``tukey``).

        :param window_function: Window function to apply to the timeseries.
        :type window_function: callable or str

        """
        self._apply_to_self(self.windowed, window_function, *args, **kwargs)

    def tukey_windowed(self, alpha):
        """Return a :py:class:`~.TimeSeries` with Tukey window with parameter ``alpha``
        applied.

        :param alpha: Tukey parameter.
        :type alpha: float

        :returns:  New windowed :py:class:`~.TimeSeries`.
        :rtype:    :py:class:`~.TimeSeries`

        """
        return self.windowed(signal.tukey, alpha)

    def tukey_window(self, alpha):
        """Apply Tukey window with parameter ``alpha``.

        :param alpha: Tukey parameter.
        :type alpha: float

        """
        self.window(signal.tukey, alpha)

    def hamming_windowed(self):
        """Return a timeseries with Hamming window applied.

        :returns:  New windowed :py:class:`~.TimeSeries`.
        :rtype:    :py:class:`~.TimeSeries`

        """
        return self.windowed(signal.hamming)

    def hamming_window(self):
        """Apply Hamming window."""
        self.window(signal.hamming)

    def blackman_windowed(self):
        """Return a timeseries with Blackman window applied."""
        return self.windowed(signal.blackman)

    def blackman_window(self):
        """Apply Blackman window."""
        self.window(signal.blackman)

    def savgol_smoothed_time(self, tsmooth, order=3):
        """Return a resampled timeseries with uniform timesteps, smoothed with
        ``savgol_smooth`` with a window that is ``tsmooth`` in time (as opposed
        to a number of points).

        :param tsmooth: Time interval over which to smooth.
        :type tsmooth: float
        :param order: Order of the filter.
        :type order: int

        :returns:  New smoothed and resampled :py:class:`~.TimeSeries`.
        :rtype:    :py:class:`~.TimeSeries`

        """
        if not self.is_regularly_sampled():
            warnings.warn(
                "TimeSeries is not regularly samples. Resampling.",
                RuntimeWarning,
            )
            ts = self.regular_resampled()
        else:
            ts = self
        dt = ts.t[1] - ts.t[0]
        # The savgol method requires a odd window
        # If it is not, we add one point
        window = int(np.rint(tsmooth / dt))
        window = window + 1 if (window % 2 == 0) else window
        return self.savgol_smoothed(window, order)

    def savgol_smooth_time(self, tsmooth, order=3):
        """Resample the timeseries with uniform timesteps, smooth it with
        ``savgol_smooth`` with a window that is ``tsmooth`` in time (as opposed to a
        number of points).

        :param tsmooth: Time interval over which to smooth.
        :type tsmooth: float
        :param order: Order of the filter.
        :type order: int

        """
        self._apply_to_self(self.savgol_smoothed_time, tsmooth, order)

    def to_FrequencySeries(self):
        """Return a :py:class:`~.FrequencySeries` that is the Fourier transform of the
        timeseries.

        If the signal is not complex, only positive frequencies are kept.

        If the timeseries is not regularly sampled, it will be resampled before
        transforming.

        :: warning:

            To have meaningful results, you should consider removing the
            mean and windowing the signal before calling this method!

        :returns: Fourier Transform.
        :rtype: :py:class:`~.FrequencySeries`

        """
        if not self.is_regularly_sampled():
            warnings.warn(
                "TimeSeries is not regularly samples. Resampling.",
                RuntimeWarning,
            )
            regular_ts = self.regular_resampled()
        else:
            regular_ts = self

        dt = regular_ts.dt

        if self.is_complex():
            frequencies = np.fft.fftfreq(len(regular_ts), d=dt)
            fft = np.fft.fft(regular_ts.y)

            f = np.fft.fftshift(frequencies)
            fft = np.fft.fftshift(fft)
        else:
            # Note the "r"
            f = np.fft.rfftfreq(len(regular_ts), d=dt)
            fft = np.fft.rfft(regular_ts.y)

        # We need the normalization dt to compute physical quantities.
        # Intuitively, NumPy computes A_k = \sum a_k exp(-2 pi f t), to
        # transform this into an integral (true Fourier transform), we have to
        # multiply this by the measure of integration.
        return frequencyseries.FrequencySeries(f, fft * dt)
