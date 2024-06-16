#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
#
# Based on code originally developed by Wolfgang Kastaun. This file may contain
# algorithms and/or structures first implemented in
# GitHub:wokast/PyCactus/PostCactus/attr_dict.py
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

"""The :py:mod:`~.attr_dict` module provides supporting infrastructure to
access dictionary as attributes. That is, to be able to do something like
object.attribute instead of object['attribute'] with attribute dynamically
determined.

This module is widely used in `kuibit`, for example in the ``fields``
attributes. The class :py:class:`~.AttributeDictionary` is used in conjuction
with :py:class:`~.TransformDictionary`, which we use to invoke ``_getitem__``
of the parent class.

"""

import re


class AttributeDictionary:
    """AttributeDictionary provide syntactic sugar to transform a dictionary
    in a class in which the attributes are the key/values of the dictionary.

    With this, you can obtain the value of ``ad['attr']`` with the notation
    ``ad.attr``.

    This works by using storing the dictionary as a member and using
    ``__getattr__`` to access the values.

    """

    def __init__(self, elements):
        """Store elements in self._elem.

        :param elements: Dictionary that has to be converted in collections of
                         attributes.
        :type elements: dict

        """
        # Here we define a new attribute _elem, which will be a dictionary with
        # all the data.
        #
        # If ad is an AttributeDictionary, we will have ad._elem = elements
        #
        # Here we have to use super() because we are going change the
        # __setattr__ method of this class to make the class immutable.
        # If we were to write instead: self._elem = elements we would get
        # the error "Attributes are immutable".
        #
        # Notice also that because of the non-standard way this class is written,
        # AttributeDictionary would not be pickable without writing suitable
        # __getstate__ and __setstate__ methods.
        super().__setattr__("_elem", elements)

    def __setattr__(self, name, value):
        # We prevent writing directly the attributes
        raise RuntimeError("Attributes are immutable")

    def __getattr__(self, name):
        """Read _elem and return the value associated to the key name.

        :param name: Key in the dictionary _elem.
        :type name: str
        :returns: Value of _elem[name].

        """
        if name not in self._elem:
            raise AttributeError(f"Object has no attribute {name}")
        return self._elem[name]

    def __setitem__(self, name, value):
        # We prevent writing directly the values of the dictionary
        raise RuntimeError("Attributes are immutable")

    def __getitem__(self, name):
        """Read _elem and return the value associated to the key name.

        :param name: Key in the dictionary _elem.
        :returns: Value of _elem[name].

        """
        return self._elem[name]

    def __dir__(self):
        """Return the list of the attributes"""
        return list(self._elem.keys())

    def keys(self):
        """Return the list of the attributes"""
        return self._elem.keys()

    # We need __getstate__ and __setstate__ so that this class can be put in
    # pickle. This class is used everywhere in kuibit, so we must support
    # pickles as they are the foundations of multiprocessing.
    def __getstate__(self):
        return self._elem

    def __setstate__(self, elements):
        super().__setattr__("_elem", elements)

    def __str__(self):
        return f"Fields available:\n{list(self.keys())}"


class TransformDictionary:
    """TransformDictionary is a wrapper around a dictionary that apply
    transform to the values of the dictionary with the supplied function
    transform. This can be used to sanitize data.

    When initializated, TransformDictionary first scans the input dictionary
    elem to find if there are values that are dictionary themselves. In this
    case, it calls itself recursively. Objects that are not dictionaries
    are left untouched. Everything is then stored in the attribute _elem.

    Let's see an example:
    ``elem = {'first': 'value', 'second': {'nested': 'dictionary'} }``

    _elem will be a dictionary ``{'first': 'value', 'second': td}``, where td
    is :py:class:``~.TransformDictionary`` with
    ``td._elem = {'nested': 'dictionary'}``.

    In the most simple case, a dictionary is mapped into a new dictionary-like
    object with the same keys and values.

    The way we are going to use this is different. Consider a class Vars that
    contains a set of variables. To access this variables you can call
    Vars['varname']. If we initialize :py:class:``~.TransformDictionary`` with
    a dictionary of the form ``{varname: varname}`` and
    ``transform = Vars.__getitem__`` what happens is that calling the
    internal dictionary ``self._elem`` will just be  ``{varname: varname}``,
    but every time we call it, we first have to apply the transform function,
    so what we end up calling is ``Vars.__getitem__(varname)``.

    """

    def __init__(self, elem, transform=lambda x: x):
        if not hasattr(elem, "items"):
            raise TypeError("Input is not dictionary-like")

        def dict_filter(dict_or_elem):
            return (
                TransformDictionary(dict_or_elem, transform)
                if isinstance(dict_or_elem, dict)
                else dict_or_elem
            )

        self._elem = {k: dict_filter(v) for k, v in elem.items()}
        self._transform = transform

    def __getitem__(self, name):
        elem = self._elem[name]

        if isinstance(elem, type(self)):
            # Leave TransformDictionary untouched
            return elem

        return self._transform(elem)

    def keys(self):
        """Return the list of the available elements"""
        # Like normal dictionaries
        return self._elem.keys()

    def __contains__(self, name):
        """This allows to use the 'in' keyword."""
        return name in self._elem


def pythonize_name_dict(names_list, transform=lambda x: x):
    """Take a list of names, like ['rho[0]', 'rho[1], 'energy', 'bob'] and
    return a AttributeDictionary with attributes passed through the function.

    Names that those that are not like are 'rho[0]' are set as keys (the values
    are transform(name)). Names that are like 'rho[0]', the key is set as rho,
    and the value is set a dictionary with {0: 'rho[0]'}.

    Example:
    p = pythonize_name_dict(['energy', 'rho[0]'])
    p.energy = 'energy'
    p.rho[0] = 'rho[0]'

    p.rho is a dictionary-like object.

    We will use this function with transform = __getitem__ so that p.energy
    will return the value of energy (not the key).

    """
    res = {}
    pattern = re.compile(r"^([^\[\]]+)\[([\d]+)\]$")
    # Let's understand this regexp:
    # - ^....$ means that we match the entire string
    # - We have two capturing groups: ([^\[\]]+) and ([\d]+)
    # - ([^\[\]]+) match everything that is not [ and ] (var name)
    # - \[([\d]+)\] matches numbers in brakets (e.g. [0])
    for name in names_list:
        matched = pattern.search(name)
        if matched is None:
            # It's not something like rho[0], we can use it
            # as a key
            res[name] = name
        else:
            # The setdefault() method returns the value of the item with the
            # specified key. if the key does not exist, insert the key, with
            # the specified value e.g.:
            # res.setdefault("a", {})[1] = "a[1]" a -> {'a': {1: 'a[1]'}}
            res.setdefault(matched.group(1), {})[int(matched.group(2))] = name

    res = TransformDictionary(res, transform)
    return AttributeDictionary(res)
