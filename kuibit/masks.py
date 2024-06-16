#!/usr/bin/env python3

# Copyright (C) 2021-2024 Gabriele Bozzola
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

"""The :py:mod:`~.masks` module provides functions to work with masked data.

The module contains syntactic sugar to call masked methods in objects.
"""


class _MaskedFunction:
    def __init__(self, name):
        self.__doc__ = f"""Invoke the method ``ma_{name}`` in the given object.
        This is just syntactic sugar.

        :param struct: Object with a method named ``ma_{name}``.
        :type struct: anything

        :returns: Return value of ``ma_{name}``.
        :rtype: Same type as input
        """

        self.__name__ = name

    def __str__(self):
        return f"Masked version of {self.__name__}"

    def __call__(self, struct):
        _name = self.__name__
        if not hasattr(struct, f"ma_{_name}"):
            raise AttributeError(
                f"{type(struct)} does not support masked {_name}"
            )

        return getattr(struct, f"ma_{_name}")()


sqrt = _MaskedFunction("sqrt")
exp = _MaskedFunction("exp")
log = _MaskedFunction("log")
log2 = _MaskedFunction("log2")
log10 = _MaskedFunction("log10")
sin = _MaskedFunction("sin")
cos = _MaskedFunction("cos")
tan = _MaskedFunction("tan")
arcsin = _MaskedFunction("arcsin")
arccos = _MaskedFunction("arccos")
arctan = _MaskedFunction("arctan")
arccosh = _MaskedFunction("arccosh")
arctanh = _MaskedFunction("arctanh")
