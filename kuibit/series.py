#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
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

"""The :py:mod:`~.series` module provides a base class :py:class:`~.BaseSeries`
for representing and handling series (from which :py:class:`~.TimeSeries` and
:py:class:`~.FrequencySeries` are derived).

:py:class:`~.BaseSeries` handles series that have a independent variable ``x``
and a dependent variable ``y``. The derived classes have to implement setters
and getters if they need to rename these variables (e.g. ``x -> t``). The
independent variable has to be monotonically increasing.

:py:class:`~.BaseSeries` implements several methods for operations on series.
Most of these methods are available in two flavors: those that return a new
:py:class:`~.BaseSeries`, and those which modify the object in place. The latter
have names with imperative verbs.

This module also provides the useful function :py:func:`~.sample_common`, which
takes a list of series and resamples them to their common points.

"""

import numpy as np
from scipy import integrate, interpolate, signal

from kuibit.attr_dict import AttributeDictionary
from kuibit.numerical import BaseNumerical


# Note, we test this class testing its derived class TimeSeries
class BaseSeries(BaseNumerical):
    """Base class (not intended for direct use) for generic series data in
    which the independendent variable x is sorted.

    This class is already rich of features.

    .. note:

        Derived class should define setters and getters to handle ``x``
        and ``y``. This is where the data is stored.

        The idea is the following. The actual data is stored in the
        ``BaseSeries` properties ``data_x`` and ``data_y``. These are
        accessible from the derived classes. However, we don't want the
        derived classes to use directly ``data_x`` and ``data_y``: they
        should use something that clearly inform the user of their meaning,
        like ``t`` or ``f`` (time or frequency). To do this, we have to
        define getters and setters that access and modify ``data_x``
        and ``y`` but use more meaningful names. To define a getters,
        simply use the ``@property`` decorator:

        .. code-block:: python

            @property
            def t(self):
                 return self.data_x

        With these, ``.t`` will return ``self.data_x``. For a setter,

        .. code-block:: python

            @t.setter
            def t(self, t):
                # This is defined BaseClass
                self.data_x = t

        This is called when with ``.t = something``. Once these are defined,
        the derived classes should use their getters and setters.

    :ivar data_x: Independent variable.
    :vartype data_x: 1D NumPy array or float
    :ivar y: Dependent variable.
    :vartype y: 1D NumPy array or float

    :ivar spline_real: Coefficients for a spline represent of the real part
                       of y.
    :vartype spline_real: Tuple
    :ivar spline_imag: Coefficients for a spline represent of the real part
                       of y.
    :vartype spline_imag: Tuple

    """

    @staticmethod
    def _make_array(x):
        """Return a NumPy array version of x (if x is not already an array)."""
        return np.atleast_1d(x) if not isinstance(x, np.ndarray) else x

    def _return_array_if_monotonic(self, x_array):
        """Return the import array if it has length 1 or if it is
        monotonically increasing. Otherwise return error.

        We assume x_array is an array. We will not check for this, it is up to
        the developer to guarantee this. If this is not true, some errors will
        be thrown.

        :param x_array: Array to check if it is monotonically increasing.
        :type x_array: 1d NumPy array

        :returns: Input array, if increasing monotonically.
        :rtype: 1d NumPy array

        """
        if len(x_array) > 1:
            # Here we compute directly the diff because it seems faster
            # than using np.diff

            # Example:
            # self.x = [1,2,3]
            # self.x[1:] = [2, 3]
            # self.x[:-1] = [1, 2]
            # dx = [1,1]
            dx = x_array[1:] - x_array[:-1]
            if dx.min() <= 0:
                # HACK: To provide more useful information we assume
                #       that the derived classes are named like TimeSeries.
                #       Then, we remove the "series"
                name = type(self).__name__
                x_name = name[:-6]
                raise ValueError(f"{x_name} not monotonically increasing")

        return x_array

    def __init__(self, x, y, guarantee_x_is_monotonic=False):
        """When guarantee_x_is_monotonic is True no checks will be perform to
        make sure that x is monotonically increasing (increasing performance).
        This should is used internally whenever a new series is returned from
        self, since we have already checked that data_x is good.

        :param x: Independent variable.
        :type x: 1d NumPy array or list
        :param y: Dependent variable.
        :type y: 1d NumPy array or list
        :param guarantee_x_is_monotonic: Whether we can skip the check on monotonicity.
        :param guarantee_x_is_monotonic: bool

        """

        x_array = self._make_array(x)
        y_array = self._make_array(y)

        if len(x_array) != len(y_array):
            raise ValueError("Data length mismatch")

        if len(x_array) == 0:
            raise ValueError("Trying to construct empty Series.")

        if not guarantee_x_is_monotonic:
            x_array = self._return_array_if_monotonic(x_array)

        # The copy is because we don't want to change the input values
        self.__data_x = x_array.copy()
        self.__data_y = y_array.copy()

        # The data is stored in the members self.data_x and self.data_y. We
        # will never access these directly. We have setters and getters to that
        # we can do stuff when variables change. For example, we want to
        # compute or update splines. The setter and getters are for x and y

        # We keep this flag around to know when we have to recompute the
        # splines. Operations that invalidate the splines MUST reset this flag
        # to True.
        self.invalid_spline = True
        # Here we also define the splines as empty objects so that we know
        # that they are attributes of the class and they are not uninitialized
        self.spline_real = None
        self.spline_imag = None

    @property
    def x(self):
        return self.__data_x

    @x.setter
    def x(self, x):
        x_array = self._make_array(x)
        if len(x_array) != len(self.x):
            raise ValueError("You cannot change the length of the series")
        x_array = self._return_array_if_monotonic(x_array)

        # This series should own the data, so we copy (to avoid accidentally
        # changing some other variable).
        # If you do self.x = z
        # and then self.x = *2
        # z will change (if we don't copy)
        self.__data_x = x_array.copy()

        # Invalidate the spline
        self.invalid_spline = True

    @property
    def y(self):
        return self.__data_y

    @y.setter
    def y(self, y):
        y_array = self._make_array(y)
        if len(y_array) != len(self.__data_y):
            raise ValueError("You cannot change the length of the series")

        # This series should own the data, so we copy (to avoid accidentally
        # changing some other variable)
        self.__data_y = y_array.copy()

        # Invalidate the spline
        self.invalid_spline = True

    # Here is where we pretend to be pandas. We want to be able to plot our
    # series with matplotlib. Unfortunately, there is no easy way to provide a
    # custom object to the plot functions. However, matplotlib has a special
    # hook for pandas in the function matplotlib.cbook.index_of. In this
    # function is checked if the index property is available, in which case,
    # index.values and values are returned. We use this to make our objects
    # plottable.
    # The function in matplotlib is:
    # try:
    #    return y.index.values, y.values
    # except AttributeError:
    #    y = _check_1d(y)
    #    return np.arange(y.shape[0], dtype=float), y
    #
    # If we provide index.values and values, we can return x and y

    @property
    def values(self):
        """Fake pandas properties, to make Series objects plottable by
        matplotlib.
        """
        return self.y

    @property
    def index(self):
        """Fake pandas properties, to make Series objects plottable by
        matplotlib.
        """
        return AttributeDictionary({"values": self.x})

    @property
    def xmin(self):
        """Return the minimum of the independent variable x.

        :rvalue: Minimum of x.
        :rtype: float
        """
        return self.x[0]

    @property
    def xmax(self):
        """Return the maximum of the independent variable x.

        :rvalue: Maximum of x
        :rtype: float
        """
        return self.x[-1]

    def is_regularly_sampled(self):
        """Return whether the series is regularly sampled.

        If the series is only one point, an error is raised.

        :returns:  Is the series regularly sampled?
        :rtype:    bool
        """
        if len(self) == 1:
            raise RuntimeError(
                "Series is only one point, "
                "it does not make sense to compute dx"
            )

        dx = self.x[1:] - self.x[:-1]

        return np.allclose(dx, dx[0], atol=1e-14)

    def __len__(self):
        """The number of data points."""
        return len(self.x)

    def __iter__(self):
        for x, y in zip(self.x, self.y):
            yield x, y

    def is_complex(self):
        """Return whether the data is complex.

        :returns:  True if the data is complex, false if it is not.
        :rtype:   bool

        """
        return issubclass(self.y.dtype.type, complex)

    def x_at_abs_maximum_y(self):
        """Return the value of x when abs(y) is maximum.

        :returns: Value of x when abs(y) is maximum.
        :rtype: float
        """
        return self.x[np.argmax(np.abs(self.y))]

    def x_at_abs_minimum_y(self):
        """Return the value of x when abs(y) is minimum.

        :returns: Value of x when abs(y) is minimum.
        :rtype: float
        """
        return self.x[np.argmin(np.abs(self.y))]

    def _make_spline(self, *args, k=3, s=0, **kwargs):
        """Private function to make spline representation of the data.

        This function is not meant to be called directly.

        ``k`` is the degree of the spline fit. It is recommended to use cubic
        splines. Even values of ``k`` should be avoided especially with small ``s``
        values. 1 <= k <= 5

        Unknown arguments are pass to ``scipy.interpolate.splrep``.

        :param k: Order of the spline representation.
        :type k:  int
        :param s: Smoothing of the spline.
        :type s:  float

        """
        if len(self) < k:
            raise ValueError(
                f"Too few points to compute a spline of order {k}"
            )

        self.spline_real = interpolate.splrep(
            self.x, self.y.real, k=k, s=s, *args, **kwargs
        )

        if self.is_complex():
            self.spline_imag = interpolate.splrep(
                self.x, self.y.imag, k=k, s=s, *args, **kwargs
            )

        self.invalid_spline = False

    def evaluate_with_spline(self, x, ext=2):
        """Evaluate the spline on the points ``x``.

        Values outside the interval are extrapolated if ``ext=0``, set to 0 if
        ``ext=1``, raise a ``ValueError`` if ``ext=2``, or if ``ext=3``, return
        the boundary value.

        This method is meant to be used only if you want to use a different ext
        for a specific call, otherwise, just use __call__.

        :param x: Array of x where to evaluate the series or single x.
        :type x: 1D NumPy array of float

        :param ext: How to deal values outside the bounaries. Values outside the
                    interval are extrapolated if ``ext=0``, set to 0 if
                    ``ext=1``, raise a ValueError if ``ext=2``, or if ``ext=3``,
                    return the boundary value.
        :type ext: int

        :returns: Values of the series evaluated on the input x.
        :rtype:   1D NumPy array or float

        """
        if self.invalid_spline:
            self._make_spline()

        y_real = interpolate.splev(x, self.spline_real, ext=ext)
        if self.is_complex():
            y_imag = interpolate.splev(x, self.spline_imag, ext=ext)
            ret = y_real + 1j * y_imag
        else:
            ret = y_real

        # When this method is called with a scalar input, at this point, ret
        # would be a 0d NumPy scalar array. What's that? - you may ask. I have
        # no idea, but the user is expecting a scalar as output. Hence, we cast
        # the 0d array into at "at_least_1d" array, then we can see its length
        # and act consequently.
        ret = np.atleast_1d(ret)
        return ret if len(ret) > 1 else ret[0]

    def __call__(self, x):
        """Evaluate the spline on the points x. If the value is outside the
        range, a ValueError will be raised.
        """
        # We call the spline only if we need to.

        # TODO (REFACTORING): This is not a Pythonic way to write this function.
        #
        # The main problem is that it is is not vectorized. It is also not
        # efficient, we are going over the array a lot of times, re-checking the
        # same elements over and over. Also, we are not considering the floating
        # point arithmetic, we should allow for some tolerance.

        # First we consider the scalar case
        if not hasattr(x, "__len__"):
            if x in self.x:
                return self.y[np.searchsorted(self.x, x)]
            return self.evaluate_with_spline(x, ext=2)

        ret = np.zeros(len(x), dtype=type(self.y[0]))
        # Hash-maps are more efficient than searching every time through the
        # array, but there is some overhead cost in defining the dictionary.
        # Experiments show that it is sill more performant.
        dic_data = dict(zip(self.x, self.y))
        for index, elem in enumerate(x):
            if elem in self.x:
                ret[index] = dic_data[elem]
            else:
                ret[index] = self.evaluate_with_spline(elem, ext=2)

        return ret

    def copy(self):
        """Return a deep copy.

        :returns:  Deep copy of the series.
        :rtype:    :py:class:`~.BaseSeries` or derived class
        """
        # The following is more complicated copy constructor that is designed
        # to copy also the spline information without re-computing it.
        # This can speed up some computations.
        copied = type(self).__new__(self.__class__)
        # We don't use the setters
        copied.__data_x = self.__data_x.copy()
        copied.__data_y = self.__data_y.copy()
        if not self.invalid_spline:
            # splines are tuples, with a direct call to the function
            # tuple() we make a deep copy
            copied.spline_real = tuple(self.spline_real)
            if self.is_complex():
                copied.spline_imag = tuple(self.spline_imag)
            copied.invalid_spline = False
        copied.invalid_spline = True
        return copied

    def resampled(self, new_x, ext=2, piecewise_constant=False):
        """Return a new series resampled from this to new_x.

        You can specify the details of the spline with the method make_spline.

        If you want to resample without using the spline, and you want a nearest
        neighbor resampling, pass the keyword ``piecewise_constant=True``.
        This may be a good choice for data with large discontinuities, where the
        splines are ineffective.

        :param new_x: New independent variable.
        :type new_x:  1D NumPy array or list of float
        :param ext: How to handle points outside the data interval.
        :type ext: 0 for extrapolation, 1 for returning zero, 2 for ``ValueError``,
                   3 for extending the boundary
        :param piecewise_constant: Do not use splines, use the nearest neighbors.
        :type piecewise_constant: bool
        :returns: Resampled series.
        :rtype:   :py:class:`~.BaseSeries` or derived class

        """
        # If x is the same, there's no need to resample
        if len(self.x) == len(new_x) and np.allclose(
            self.x, new_x, atol=1e-14
        ):
            return self.copy()

        # Unfortunately there is no nearest neighor resampling in SciPy's splines.
        # Hence, we use directly the method interp1d.
        if piecewise_constant:
            interp_function = interpolate.interp1d(
                self.x, self.y, kind="nearest", assume_sorted=True
            )
            new_y = interp_function(new_x)
        else:
            new_y = self.evaluate_with_spline(new_x, ext=ext)

        return type(self)(new_x, new_y)

    def resample(self, new_x, ext=2, piecewise_constant=False):
        """Resample the series to new independent variable new_x.

        If you want to resample without using the spline, and you want a nearest
        neighbor resampling, pass the keyword ``piecewise_constant=True``.
        This may be a good choice for data with large discontinuities, where the
        splines are ineffective.

        :param new_x: New independent variable.
        :type new_x:  1D NumPy array or list of float
        :param ext: How to handle points outside the interval.
        :type ext: 0 for extrapolation, 1 for returning zero, 2 for ValueError,
                   3 for extending the boundary
        :param piecewise_constant: Do not use splines, use the nearest neighbors.
        :type piecewise_constant: bool

        """
        self._apply_to_self(
            self.resampled,
            new_x,
            ext=ext,
            piecewise_constant=piecewise_constant,
        )

    def _apply_binary(self, other, function):
        """This is an abstract function that is used to implement mathematical
        operations with other series (if they have the same x) or
        scalars.

        :py:meth:`~._apply_binary` takes another object that can be of the same
        type or a scalar, and applies ``function(self.y, other.y)``, performing type
        checking.

        :param other: Other object.
        :type other: :py:class:`~.BaseSeries` or derived class or float
        :param function: Dyadic function (function that takes two arguments).
        :type function: callable

        :returns:  Return value of ``function`` when called with self and other.
        :rtype:   :py:class:`~.BaseSeries` or derived class (typically)

        """
        # If the other object is of the same type
        if isinstance(other, type(self)):
            if (len(self.x) != len(other.x)) or (
                not np.allclose(other.x, self.x, atol=1e-14)
            ):
                raise ValueError("The objects do not have the same x!")
            return type(self)(
                self.x,
                function(self.y, other.y),
                True,
            )
        # If it is a number
        if isinstance(other, (int, float, complex)):
            return type(self)(self.x, function(self.y, other), True)

        # If we are here, it is because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")

    def __eq__(self, other):
        """Check for equality up to numerical precision."""
        if isinstance(other, type(self)):
            return np.allclose(self.x, other.x, atol=1e-14) and np.allclose(
                self.y, other.y, atol=1e-14
            )
        return False

    def _apply_to_self(self, f, *args, **kwargs):
        """Apply the method ``f`` to ``self``, modifying ``self``.

        This is used to transform the commands from returning an object to
        modifying ``self``. The function ``f`` has to return a new copy of the
        object (not a reference).

        """
        ret = f(*args, **kwargs)
        # We avoid the setters to avoid checking for consistency because this
        # was already done
        self.__data_x, self.__data_y = ret.x, ret.y
        # We have to recompute the splines
        self.invalid_spline = True

    def save(self, file_name, *args, **kwargs):
        """Saves into simple ASCII format with 2 columns ``(x, y)``
        for real valued data and 3 columns ``(x, Re(y), Im(y))``
        for complex valued data.

        Unknown arguments are passed to ``NumPy.savetxt``.

        :param file_name: Path (with extension) of the output file.
        :type file_name: str

        """
        if self.is_complex():
            np.savetxt(
                file_name,
                np.transpose(
                    (self.x, self.y.real, self.y.imag),
                    *args,
                    **kwargs,
                ),
            )
        else:
            np.savetxt(
                file_name,
                np.transpose((self.x, self.y), *args, **kwargs),
            )

    def nans_removed(self):
        """Filter out nans/infinite values.
        Return a new series with finite values only.

        :returns: A new series with only finite values.
        :rtype: :py:class:`~.BaseSeries` or derived class
        """
        msk = np.isfinite(self.y)
        return type(self)(self.x[msk], self.y[msk], True)

    def nans_remove(self):
        """Filter out nans/infinite values."""
        self._apply_to_self(self.nans_removed)

    def integrated(self, dx=None):
        """Return a series that is the integral computed with method of
        the rectangles.

        The spacing ``dx`` can be optionally provided. If provided, it will be
        used (increasing performance), otherwise it will be computed internally.

        :param dx: Delta x in the independent variable. If None it will be
                   computed internally.
        :type dx: float or None
        :returns:  New series with the cumulative integral.
        :rtype:    :py:class:`~.BaseSeries` or derived class

        """
        # We pass self.x only if dx was not provided
        passing_x = self.x if dx is None else None
        return type(self)(
            self.x,
            integrate.cumtrapz(self.y, x=passing_x, dx=dx, initial=0),
            True,
        )

    def integrate(self, dx=None):
        """Integrate series with method of the rectangles.

        The spacing ``dx`` can be optionally provided. If provided, it will be
        used (increasing performance), otherwise it will be computed internally.

        """
        self._apply_to_self(self.integrated, dx=dx)

    def spline_differentiated(self, order=1):
        """Return a series that is the derivative of the current one using
        the spline representation.

        The optional parameter ``order`` specifies the order of the derivative.

        .. warning::

            The values at the boundary are typically not accurate.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int

        :returns:  New series with derivative
        :rtype:    :py:class:`~.BaseSeries` or derived class

        """
        if (order > 3) or (order < 0):
            raise ValueError(f"Cannot compute differential of order {order}")

        if self.invalid_spline:
            self._make_spline()

        if self.is_complex():
            ret_value = interpolate.splev(
                self.x, self.spline_real, der=order
            ) + 1j * interpolate.splev(self.x, self.spline_imag, der=order)
        else:
            ret_value = interpolate.splev(self.x, self.spline_real, der=order)

        return type(self)(self.x, ret_value, True)

    def spline_differentiate(self, order=1):
        """Differentiate the series using the spline representation.

        The optional parameter ``order`` specifies the order of the derivative.

        .. warning::

            The values at the boundary are typically not accurate.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int

        """
        self._apply_to_self(self.spline_differentiated, order)

    def differentiated(self, order=1):
        """Return a series that is the numerical order-differentiation of
        the present series.

        The optional parameter ``order`` specifies the order of the derivative.

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int

        :returns:  New series with derivative.
        :rtype:    :py:class:`~.BaseSeries` or derived class

        """
        ret_value = self.y
        for _num_deriv in range(order):
            ret_value = np.gradient(ret_value, self.x, edge_order=2)
        return type(self)(self.x, ret_value, True)

    def differentiate(self, order=1):
        """Differentiate with the numerical order-differentiation.

        The optional parameter ``order`` specifies the order of the derivative.

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        :param order: Order of derivative (e.g. 2 = second derivative).
        :type order: int

        """
        self._apply_to_self(self.differentiated, order)

    def savgol_smoothed(self, window_size, order=3):
        """Return a smoothed series with a Savitzky-Golay filter with
        window of size ``window_size`` and order ``order``.

        This is just like a regular "Moving average" filter, but instead of
        just calculating the average, a polynomial (usually 2nd or 4th order)
        fit is made for every point, and only the "middle" point is chosen.
        Since 2nd (or 4th) order information is concerned at every point, the
        bias introduced in "moving average" approach at local maxima or minima,
        is circumvented.

        :param window_size: Number of points of the smoothing window (needs to
                            be odd).
        :type window_size: int
        :param order: Order of the filter.
        :type order: int

        :returns:  New smoothed series.
        :rtype:    :py:class:`~.BaseSeries` or derived class

        """
        if self.is_complex():
            return type(self)(
                self.x,
                signal.savgol_filter(self.y.imag, window_size, order)
                + 1j * signal.savgol_filter(self.y.real, window_size, order),
                True,
            )

        return type(self)(
            self.x,
            signal.savgol_filter(self.y, window_size, order),
            True,
        )

    def savgol_smooth(self, window_size, order=3):
        """Smooth the series with a Savitzky-Golay filter with window of
        size ``window_size`` and order ``order``.

        This is just like a regular "Moving average" filter, but instead of
        just calculating the average, a polynomial (usually 2nd or 4th order)
        fit is made for every point, and only the "middle" point is chosen.
        Since 2nd (or 4th) order information is concerned at every point, the
        bias introduced in "moving average" approach at local maxima or minima,
        is circumvented.

        :param window_size: Number of points of the smoothing window (needs to
                            be odd).
        :type window_size: int
        :param order: Order of the filter.
        :type order: int

        """
        self._apply_to_self(self.savgol_smoothed, window_size, order)

    def cropped(self, init=None, end=None):
        """Return a series with data removed outside the interval ``[init, end]``. If
        ``init`` or ``end`` are not specified or None, it does not remove
        anything from this side.

        :param init: Data with ``x <= init`` will be removed.
        :type init: float or None
        :param end: Data with ``x >= init`` will be removed.
        :type end: float or None

        :returns:  Series with enforced minimum and maximum
        :rtype:    :py:class:`~.BaseSeries` or derived class

        """
        x = self.x
        y = self.y
        if init is not None:
            m = x >= init
            x = x[m]
            y = y[m]
        if end is not None:
            m = x <= end
            x = x[m]
            y = y[m]
        return type(self)(x, y, True)

    def crop(self, init=None, end=None):
        """Remove data outside the the interval ``[init, end]``. If
        ``init`` or ``end`` are not specified or None, it does not remove
        anything from this side.

        :param init: Data with ``x <= init`` will be removed.
        :type init: float or None
        :param end: Data with ``x >= init`` will be removed.
        :type end: float or None

        """
        self._apply_to_self(self.cropped, init, end)

    # Define aliases
    clip = crop
    clipped = cropped

    def _apply_unary(self, function):
        """Apply a unary function to the data.

        :param function: Function to apply to the series.
        :type function: callable

        :return: New series with function applied to the data.
        :rtype: :py:class:`~.BaseSeries` or derived class

        """
        return type(self)(self.x, function(self.y), True)

    def _apply_reduction(self, reduction):
        """Apply a reduction to the data.

        :param function: Function to apply to the series.
        :type function: callable

        :return: Reduction applied to the data
        :rtype: float

        """
        return reduction(self.y)


def sample_common(series, resample=False, piecewise_constant=False):
    """Take a list of series and return new ones so that they are all defined on the
    same points.

    If ``resample`` is False (default), take as input a list of series and
    return a new list with the same series but only defined on those points that
    are common to all the lists. If ``resample`` is True, instead of removing
    points, find the common interval of definition, and resample all the series
    on that internal. The number of sample points is the minimum over all
    series. Additionally, if ``piecewise_constant=True``, the approximant used
    for resampling is a piecewise constant function, splines are not used,
    instead, the nearest neighbors are used. Use this when you have series with
    discontinuities.

    :param series: The series to resample or redefine on the common points
    :type series:  list of :py:class:`~.Series`
    :param resample: Whether to resample the series, or just find the common
                     points.
    :type resample: bool
    :param piecewise_constant: Whether to use the nearest neighbor resampling
                               method instead of splines.
                               If ``piecewise_constant=True``, the approximant used
                               for resampling is a piecewise constant function.
    :type piecewise_constant: bool
    :returns:  Resampled series so that they are all defined in
               the same interval.
    :rtype:    list of :py:class:`~.Series`

    """
    # In many cases there is no real need for resampling because the array
    # already have the desired shape. It is worth checking if we need to
    # resample. If the series are regularly sampled, it is easy to check
    # if the are the same. We also need to check that they are regularly
    # sampled, to do this, we check that the first is regularly sampled,
    # and that all the other ones have the same x.
    s1, *s_others = series
    if s1.is_regularly_sampled():
        for s in s_others:
            if not (len(s) == len(s1)):
                break
            if not np.allclose(s1.x, s.x, atol=1e-14):
                break
            # This is an else to the for loop
        else:
            # We have to copy, otherwise one can accidentally modify input data
            return [ss.copy() for ss in series]

    if resample:
        # Find the series with max xmin
        s_xmin = max(series, key=lambda x: x.xmin)
        # Find the series with min xmax
        s_xmax = min(series, key=lambda x: x.xmax)
        # Find the series with min number of points
        s_ns = min(series, key=len)
        x = np.linspace(s_xmin.xmin, s_xmax.xmax, len(s_ns))
        return [
            s.resampled(x, piecewise_constant=piecewise_constant)
            for s in series
        ]

    def float_intersection(array_1, array_2):
        """Here we find the intersection between the two arrays also
        considering the floating points.
        """
        # https://stackoverflow.com/a/32516182
        return array_2[
            np.isclose(array_1[:, None], array_2, atol=1e-14).any(0)
        ]

    # Here we find the common intersection between all the x, starting with
    # the first one
    x = s1.x
    for s in s_others:
        x = float_intersection(x, s.x)
        if len(x) == 0:
            raise ValueError("Series do not have any point in common")

    return [s.resampled(x) for s in series]
