#!/usr/bin/env python3

# Copyright (C) 2020 Gabriele Bozzola, Wolfgang Kastaun
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
:py:class:`~.BaseNumerical` that implements basic mathematical operations.
This is used by series and grid functions.

"""

from abc import ABC, abstractmethod

import numpy as np


# Note, we test this class testing its derived class TimeSeries
class BaseNumerical(ABC):
    """Base class (not intended for direct use) for generic numerical data.

    The derived classes have to implement:
    - _apply_unary(self, function) that returns function(self)
    - _apply_binary(self, other, function) that returns function(self, other)
    - _apply_reduction(self, function) that returns function(self)
    """

    @abstractmethod
    def _apply_unary(self, function):
        pass

    @abstractmethod
    def _apply_binary(self, other, function):
        pass

    @abstractmethod
    def _apply_reduction(self, reduction):
        pass

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
        return self ** other

    def __neg__(self):
        return self._apply_unary(np.negative)

    def __abs__(self):
        return self._apply_unary(np.abs)

    def min(self):
        return self._apply_reduction(np.min)

    def max(self):
        return self._apply_reduction(np.max)

    def abs_min(self):
        """Return the minimum of the absolute value"""
        return abs(self)._apply_reduction(np.min)

    def abs_max(self):
        """Return the maximum of the absolute value"""
        return abs(self)._apply_reduction(np.max)

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

    def log10(self):
        return self._apply_unary(np.log10)

    def conjugate(self):
        return self._apply_unary(np.conjugate)
