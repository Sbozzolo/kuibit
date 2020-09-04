#!/usr/bin/env python3

"""The :py:mod:`~.series` module provides a base class :py:class:`~.BaseSeries`
for representing and handling series (from which time and frequency series are
derived).

"""

import warnings

import numpy as np
from scipy import integrate, interpolate, signal


# Note, we test this class testing its derived class TimeSeries
class BaseSeries:
    """Base class (not intended for direct use) for generic series data.

    This class is already rich of features.

    .. note:

        Derived class should define setters and getters to handle ``data_x``
        and ``data_y``. This is where the data is stored.

        The idea is the following. The actual data is stored in the
        ``BaseSeries` properties ``data_x`` and ``data_y``. These are
        accessible from the derived classes. However, we don't want the
        derived classes to use directly ``data_x`` and ``data_y``: they
        should use something that clearly inform the user of their meaning,
        like ``t`` or ``f`` (time or frequency). To do this, we have to
        define getters and setters that access and modify ``data_x``
        and ``data_y`` but use more meaningful names. To define a getters,
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

    :ivar data_x:   x
    :vartype data_x: 1D numpy array or float
    :ivar data_y:   y
    :vartype data_y: 1D numpy array or float

    :ivar spline_real: Coefficients for a spline represent of the real part
                       of y
    :vartype spline_real: Tuple

    :ivar spline_imag: Coefficients for a spline represent of the real part
                       of y
    :vartype spline_imag: Tuple

    """

    def __init__(self, x, y):

        if (not hasattr(x, '__len__')):
            x = np.array([x])
            y = np.array([y])
        else:
            # Make sure xhese are arrays
            x = np.array(x)
            y = np.array(y)

        if (len(x) != len(y)):
            raise ValueError('Data length mismatch')
        #
        if (len(x) == 0):
            raise ValueError('Trying to construct empty Series.')

        # The copy is because we don't want to change the input values
        self.data_x = np.array(x).copy()
        self.data_y = np.array(y).copy()

        if (len(x) <= 3):
            warnings.warn('Spline will not be computed: too few points')
        else:
            self._make_spline()

    @property
    def xmin(self):
        """Return the min of the independent variable x

        :rvalue: Min of x
        :rtype: float
        """
        # Derived classes can implement more efficients methods if x has known
        # ordering
        return np.amin(self.data_x)

    @property
    def xmax(self):
        """Return the max of the independent variable x

        :rvalue: Max of x
        :rtype: float
        """
        # Derived classes can implement more efficients methods if x has known
        # ordering
        return np.amax(self.data_x)

    def __len__(self):
        """The number of data points."""
        return len(self.data_x)

    def __iter__(self):
        for x, y in zip(self.data_x, self.data_y):
            yield x, y

    def is_complex(self):
        """Return whether the data is complex.

        :returns:  True if the data is complex, false if it is not
        :rtype:   bool

        """
        return issubclass(self.data_y.dtype.type, complex)

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

        self.spline_real = interpolate.splrep(self.data_x, self.data_y.real,
                                              k=k, s=s, *args, **kwargs)

        if (self.is_complex()):
            self.spline_imag = interpolate.splrep(self.data_x,
                                                  self.data_y.imag,
                                                  k=k, s=s,
                                                  *args, **kwargs)

    def evaluate_with_spline(self, x, ext=2):
        """Evaluate the spline on the points x.

        Values outside the interval are extrapolated if ext=0, set to 0 if
        ext=1, raise a ValueError if ext=2, or if ext=3, return the boundary
        value.

        This method is meant to be used only if you want to use a different ext
        for a specific call, otherwise, just use __call__.

        :param x: Array of x where to evaluate the series or single x
        :type x: 1D numpy array of float

        :param ext: How to deal values outside the bounaries. Values outside
                    the interval are extrapolated if ext=0, set to 0 if ext=1,
                    raise a ValueError if ext=2, or if ext=3, return the
                    boundary value.
        :type ext:  bool

        :returns: Values of the series evaluated on the input x
        :rtype:   1D numpy array or float

        """
        y_real = interpolate.splev(x, self.spline_real, ext=ext)
        if (self.is_complex()):
            y_imag = interpolate.splev(x, self.spline_imag, ext=ext)
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

    def __call__(self, x):
        """Evaluate the spline on the points x. If the value is outside the
        range, a ValueError will be raised.
        """
        return self.evaluate_with_spline(x, ext=2)

    def copy(self):
        """Return a deep copy.

         :returns:  Deep copy of the series
         :rtype:    :py:class:`~.BaseSeries` or derived class
         """
        return type(self)(self.data_x, self.data_y)

    def resampled(self, new_x, ext=2):
        """Return a new series resampled from this to new_x.

        You can specify the details of the spline with the method make_spline.

        :param new_x: New independent variable
        :type new_x:  1D numpy array or list of float
        :param ext: How to handle points outside the data interval
        :type ext: 0 for extrapolation, 1 for returning zero, 2 for ValueError,
                   3 for extending the boundary
        :returns: Resampled series.
        :rtype:   :py:class:`~.BaseSeries` or derived class

        """
        return type(self)(new_x, self.evaluate_with_spline(new_x,
                                                           ext=ext))

    def resample(self, new_x, ext=2):
        """Resample the series to new independent variable new_x using splines.

        :param new_x: New independent variable
        :type new_x:  1D numpy array or list of float
        :param ext: How to handle points outside the interval
        :type ext: 0 for extrapolation, 1 for returning zero, 2 for ValueError,
                   3 for extending the boundary

        """
        self._apply_to_self(self.resampled, new_x, ext=ext)

    def __neg__(self):
        return type(self)(self.data_x, -self.data_y)

    def __abs__(self):
        return type(self)(self.data_x, np.abs(self.data_y))

    def _apply_binary(self, other, function):
        """This is an abstract function that is used to implement mathematical
        operations with other series (if they have the same data_x) or
        scalars.

        _apply_binary takes another object that can be of the same type or a
        scalar, and applies function(self.y, other.y), performing type
        checking.

        :param other: Other object
        :type other: :py:class:`~.BaseSeries` or derived class or float
        :param function: Dyadic function
        :type function: callable

        :returns:  Return value of function when called with self and ohter
        :rtype:   :py:class:`~.BaseSeries` or derived class (typically)

        """
        # If the other object is of the same type
        if (isinstance(other, type(self))):
            if ((not np.allclose(other.data_x, self.data_x))
                    or (len(self.data_x) != len(other.data_x))):
                raise ValueError(
                    "The objects do not have the same x!")
            return type(self)(self.data_x, function(self.data_y, other.data_y))
        # If it is a number
        if isinstance(other, (int, float, complex)):
            return type(self)(self.data_x, function(self.data_y, other))

        # If we are here, it is because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")

    def __add__(self, other):
        """Add two series (if they have the same data_x), or add a scalar to a
        series.
        """
        return self._apply_binary(other, np.add)

    __radd__ = __add__

    def __sub__(self, other):
        """Subtract two series (if they have the same data_x), or subtract a
        scalar from a series.

        """
        return self._apply_binary(other, np.subtract)

    def __rsub__(self, other):
        """Subtract two series (if they have the same data_x), or subtract a
        scalar from a series.

        """
        return -self._apply_binary(other, np.subtract)

    def __mul__(self, other):
        """Multiply two series (if they have the same data_x), or multiply
        by scalar.
        """
        return self._apply_binary(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide two series (if they have the same data_x), or divide by a
        scalar.
        """
        if (other == 0):
            raise ValueError("Cannot divide by zero")
        return self._apply_binary(other, np.divide)

    def __pow__(self, other):
        """Raise the first series to second series (if they have the
        same data_x), or raise the series to a scalar.

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
        if (isinstance(other, type(self))):
            return (np.allclose(self.data_x, other.data_x)
                    and np.allclose(self.data_y, other.data_y))
        return False

    def _apply_to_self(self, f, *args, **kwargs):
        """Apply the method f to self, modifying self.
        This is used to transform the commands from returning an object
        to modifying self.
        """
        ret = f(*args, **kwargs)
        self.data_x, self.data_y = ret.data_x, ret.data_y
        self._make_spline()

    def save(self, fname, *args, **kwargs):
        """Saves into simple ASCII format with 2 columns (data_x, data_y)
        for real valued data and 3 columns (data_x, Re(data_y), Im(data_y))
        for complex valued data.

        :param fname: Path (with extensiton) of the output file
        :type fname: str

        """
        if self.is_complex():
            np.savetxt(fname, np.transpose((self.data_x, self.data_y.real,
                                            self.data_y.imag),
                                           *args, **kwargs))
        else:
            np.savetxt(fname, np.transpose((self.data_x, self.data_y),
                                           *args, **kwargs))

    def nans_removed(self):
        """Filter out nans/infinite values.
        Return a new series with finite values only.

        :returns: A new series with only finite values
        :rtype: :py:class:`~.BaseSeries` or derived class
        """
        msk = np.isfinite(self.data_y)
        return type(self)(self.data_x[msk], self.data_y[msk])

    def nans_remove(self):
        """Filter out nans/infinite values."""
        self._apply_to_self(self.nans_removed)

    def integrated(self):
        """Return a series that is the integral computed with method of
        the trapeziod.

        :returns:  New series cumulative integral
        :rtype:    :py:class:`~.BaseSeries` or derived class
        """
        return type(self)(self.data_x,
                          integrate.cumtrapz(self.data_y, x=self.data_x,
                                             initial=0))

    def integrate(self):
        """Integrate series with method of the trapeziod.
        """
        self._apply_to_self(self.integrated)

    def spline_derived(self, order=1):
        """Return a series that is the derivative of the current one using
        the spline interpolation. You shouldn't trust the values at the
        boundaries too much, you may want to crop it out.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int

        :returns:  New series with derivative
        :rtype:    :py:class:`~.BaseSeries` or derived class

        """
        if ((order > 3) or (order < 0)):
            raise ValueError(f'Cannot compute differential of order {order}')

        if (self.is_complex()):
            ret_value = (interpolate.splev(self.data_x,
                                           self.spline_real,
                                           der=order)
                         + 1j * interpolate.splev(self.data_x,
                                                  self.spline_imag,
                                                  der=order))
        else:
            ret_value = interpolate.splev(self.data_x, self.spline_real,
                                          der=order)

        return type(self)(self.data_x, ret_value)

    def spline_derive(self, order=1):
        """Derive the series current one using the spline interpolation.
        To keep the series of the same size as the original one, the value
        of the derivative at the boundaries is set to zero. Don't trust it!

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int

        """
        self._apply_to_self(self.spline_derived, order)

    def derived(self, order=1):
        """Return a series that is the numerical order-differentiation of
        the present series. (order = number of derivatives, ie order=2 is
        second derivative)

        The derivative is calulated as centered differencing in the interior
        and one-sided derivatives at the boundaries. Higher orders are computed
        applying the same rule recursively.

        :param order: Order of derivative (e.g. 2 = second derivative)
        :type order: int

        :returns:  New series with derivative
        :rtype:    :py:class:`~.BaseSeries` or derived class

        """
        ret_value = self.data_y
        for _num_deriv in range(order):
            ret_value = np.gradient(ret_value, self.data_x)
        return type(self)(self.data_x, ret_value)

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

    def savgol_smoothed(self, window_size, order=3):
        """Return a smoothed series with a Savitzky-Golay filter with
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

        :returns:  New smoothed series
        :rtype:    :py:class:`~.BaseSeries` or derived class

        """
        if self.is_complex():
            return type(self)(self.data_x,
                              signal.savgol_filter(self.data_y.imag,
                                                   window_size,
                                                   order)
                              + 1j * signal.savgol_filter(self.data_y.real,
                                                          window_size,
                                                          order))

        return type(self)(self.data_x, signal.savgol_filter(self.data_y,
                                                            window_size,
                                                            order))

    def savgol_smooth(self, window_size, order=3):
        """Smooth the series with a Savitzky-Golay filter with window of
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

    def cropped(self, init=None, end=None):
        """Return a series with data removed outside the intarval
        [init, end]. If init or end are not specified or None, it does not
        remove anything from this side.

        :param init: New minimum data_x
        :type init: float
        :param end: New maximum data_x
        :type end: float

        :returns:  Series with enforced minimum and maximum
        :rtype:    :py:class:`~.BaseSeries` or derived class
        """
        x = self.data_x
        y = self.data_y
        if (init is not None):
            m = (x >= init)
            x = x[m]
            y = y[m]
        if (end is not None):
            m = (x <= end)
            x = x[m]
            y = y[m]
        return type(self)(x, y)

    def crop(self, init=None, end=None):
        """Remove data outside the intarval [init, end]. If init or end
        are not specified or None, it does not remove anything from this side.

        :param init: New minimum data_x
        :type init: float
        :param end: New maximum data_x
        :type end: float

        """
        self._apply_to_self(self.cropped, init, end)

    # Define aliases
    clip = crop
    clipped = cropped

    def _apply_unary(self, function):
        """Apply a unary function to the data.

        :param function: Function to apply to the series
        :type function: callable

        :return: New series with function applied to the data
        :rtype: :py:class:`~.BaseSeries` or derived class

        """
        return type(self)(self.data_x, function(self.data_y))

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


def sample_common(series):
    """Resample a list of series to the largest interval covered by all series,
    using regularly spaced x.

    The number of sample points is the minimum over all series.

    :param ts: The series to resample
    :type ts:  list of :py:class:`~.Series`

    :returns:  Resampled series so that they are all defined in
               the same interval
    :rtype:    list of :py:class:`~.Series`

    """
    # Find the series with max xmin
    s_xmin = max(series, key=lambda x: x.xmin)
    # Find the series with min xmax
    s_xmax = min(series, key=lambda x: x.xmax)
    # Find the series with min number of points
    s_ns = min(series, key=len)
    x = np.linspace(s_xmin.xmin, s_xmax.xmax, len(s_ns))
    return [s.resampled(x) for s in series]
