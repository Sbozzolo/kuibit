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

"""The :py:mod:`~.numerical` module provides an abstract class
:py:class:`~.BaseNumerical` that implements basic mathematical operations. This
is used by series and grid functions.

To work, class derived from :py:class:`~.BaseNumerical` have to implement
methods to describe (1) how to apply a function that takes the object as argument,
(2) how to apply a function that takes two objects as arguments, and (3) how to
apply a reduction.

"""

from abc import ABC, abstractmethod

import numpy as np


# Note, we test this class testing its derived class TimeSeries
class BaseNumerical(ABC):
    """Base abstract class for generic numerical data.

    This class provides the infrastructure needed to implement mathematical
    operations in all the derived classes.

    The derived classes have to implement:
    - _apply_unary(self, function, *args, **kwargs)
       that returns function(self, *args, **kwargs)
    - _apply_binary(self, other, function, *args, **kwargs)
       that returns function(self, other, *args, **kwargs)
    - _apply_reduction(self, function, *args, **kwargs)
       that returns function(self, *args, **kwargs)
    - _apply_to_self(self, function, *args, **kwargs)
       that returns applies function(self, *args, **kwargs)
       and modifies self.

    """

    @abstractmethod
    def _apply_unary(self, function, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _apply_binary(self, other, function, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _apply_reduction(self, reduction, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _apply_to_self(self, function, *args, **kwargs):
        raise NotImplementedError

    def __add__(self, other):
        return self._apply_binary(other, np.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._apply_binary(other, np.subtract)

    def __rsub__(self, other):
        return -self._apply_binary(other, np.subtract)

    def __mul__(self, other):
        return self._apply_binary(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if other == 0:
            raise ValueError("Cannot divide by zero")
        return self._apply_binary(other, np.divide)

    def __rtruediv__(self, other):
        # This self._apply_binary(other, np.divide)
        # divives self by other, so, we reverse that
        # with ** -1
        return (self._apply_binary(other, np.divide)) ** -1

    def __pow__(self, other):
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
        return self**other

    def __neg__(self):
        return self._apply_unary(np.negative)

    def __abs__(self):
        return self._apply_unary(np.abs)

    def min(self):
        return self._apply_reduction(np.min)

    def max(self):
        return self._apply_reduction(np.max)

    def mean(self):
        return self._apply_reduction(np.mean)

    average = mean

    def median(self):
        return self._apply_reduction(np.median)

    def std(self):
        return self._apply_reduction(np.std)

    def nanmin(self):
        return self._apply_reduction(np.nanmin)

    def nanmax(self):
        return self._apply_reduction(np.nanmax)

    def abs_min(self):
        """Return the minimum of the absolute value"""
        # skipcq PYL-W0212
        return abs(self)._apply_reduction(np.min)

    def abs_max(self):
        """Return the maximum of the absolute value"""
        # skipcq PYL-W0212
        return abs(self)._apply_reduction(np.max)

    def abs_nanmin(self):
        """Return the minimum of the absolute value ignoring NaNs"""
        # skipcq PYL-W0212
        return abs(self)._apply_reduction(np.nanmin)

    def abs_nanmax(self):
        """Return the maximum of the absolute value ignoring NaNs"""
        # skipcq PYL-W0212
        return abs(self)._apply_reduction(np.nanmax)

    def abs(self):
        return self._apply_unary(np.abs)

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

    def log2(self):
        return self._apply_unary(np.log2)

    def log10(self):
        return self._apply_unary(np.log10)

    def conjugate(self):
        return self._apply_unary(np.conjugate)

    # Masked functions

    def ma_sqrt(self):
        return self._apply_unary(np.ma.sqrt)

    def ma_exp(self):
        return self._apply_unary(np.ma.exp)

    def ma_log(self):
        return self._apply_unary(np.ma.log)

    def ma_log2(self):
        return self._apply_unary(np.ma.log2)

    def ma_log10(self):
        return self._apply_unary(np.ma.log10)

    def ma_sin(self):
        return self._apply_unary(np.ma.sin)

    def ma_cos(self):
        return self._apply_unary(np.ma.cos)

    def ma_tan(self):
        return self._apply_unary(np.ma.tan)

    def ma_arcsin(self):
        return self._apply_unary(np.ma.arcsin)

    def ma_arccos(self):
        return self._apply_unary(np.ma.arccos)

    def ma_arctan(self):
        return self._apply_unary(np.ma.arctan)

    def ma_arccosh(self):
        return self._apply_unary(np.ma.arccosh)

    def ma_arctanh(self):
        return self._apply_unary(np.ma.arctanh)

    # Create masks

    def masked_equal(self, value):
        """Return a new objected masked where data is equal to given value."""
        return self._apply_unary(np.ma.masked_equal, value)

    def mask_equal(self, value):
        """Mask where data is equal to given value."""
        self._apply_to_self(self.masked_equal, value)

    def masked_greater(self, value):
        """Return a new objected masked where data is greater to given value."""
        return self._apply_unary(np.ma.masked_greater, value)

    def mask_greater(self, value):
        """Mask where data is greater to given value."""
        self._apply_to_self(self.masked_greater, value)

    def masked_greater_equal(self, value):
        """Return a new objected masked where data is greater or equal to given value."""
        return self._apply_unary(np.ma.masked_greater_equal, value)

    def mask_greater_equal(self, value):
        """Mask where data is greater or equal to given value."""
        self._apply_to_self(self.masked_greater_equal, value)

    def masked_inside(self, value1, value2):
        """Return a new objected masked where data is inside the given values."""
        return self._apply_unary(np.ma.masked_inside, value1, value2)

    def mask_inside(self, value1, value2):
        """Mask where data is inside the given values."""
        self._apply_to_self(self.masked_inside, value1, value2)

    def masked_invalid(self):
        """Return a new objected masked where data is invalid (NaNs or infs)."""
        return self._apply_unary(np.ma.masked_invalid)

    def mask_invalid(self):
        """Mask where data is invalid (NaNs of infs)."""
        self._apply_to_self(self.masked_invalid)

    def masked_less(self, value):
        """Return a new objected masked where data is less to given value."""
        return self._apply_unary(np.ma.masked_less, value)

    def mask_less(self, value):
        """Mask where data is less to given value."""
        self._apply_to_self(self.masked_less, value)

    def masked_less_equal(self, value):
        """Return a new objected masked where data is less or equal to given value."""
        return self._apply_unary(np.ma.masked_less_equal, value)

    def mask_less_equal(self, value):
        """Mask where data is less or equal to given value."""
        self._apply_to_self(self.masked_less_equal, value)

    def masked_not_equal(self, value):
        """Return a new objected masked where data is not equal to given value."""
        return self._apply_unary(np.ma.masked_not_equal, value)

    def mask_not_equal(self, value):
        """Mask where data is not equal to given value."""
        self._apply_to_self(self.masked_not_equal, value)

    def masked_outside(self, value1, value2):
        """Return a new objected masked where data is outside the given values."""
        return self._apply_unary(np.ma.masked_outside, value1, value2)

    def mask_outside(self, value1, value2):
        """Mask where data is outside the given values."""
        self._apply_to_self(self.masked_outside, value1, value2)
