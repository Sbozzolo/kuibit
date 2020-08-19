#!/usr/bin/env python3

"""The :py:mod:`~.attr_dict` module provides supporting infrastructure to access
dictionary as attributes. That is, to be able to do something like
object.attribute instead of object['attribute'] with attribute dynamically
determined.

Then, we have TransformDictionary

"""

import re


class AttributeDictionary:
    """AttributeDictionary provide syntatic sugar to transform a dictionary
    in a class in which the members are the key/values of the dictionary.

    This works by using storing the dictionary as a member and using
    __getattr__ to access the values.
    """

    def __init__(self, elements):
        """Store elements in _elem

        :param elements: Dictionary
        :type elements: dict

        """
        # Here we define a new attribute _elem

        # If ad is an AttributeDictionary, we will have
        # ad._elem = elements
        super().__setattr__('_elem', elements)

    def __setattr__(self, name, value):
        # We prevent writing directly the attributes
        raise RuntimeError("Attributes are immutable")

    def __getattr__(self, name):
        """Read _elem and return the value associated to the key name.

        :param name: Key in the dictionary _elem
        :returns: Value of _elemn[name]

        """
        return self._elem[name]

    def __dir__(self):
        """Return the list of the attributes
        """
        return list(self._elem.keys())


class TransformDictionary:
    """TransformDictionary is a wrapper around a dictionary that apply
    transform to the values of the dictionary with the supplied function
    transform. This can be used to sanitize data.

    When initializated, TransformDictionary first scans the input dictionary
    elem to find if there are values that are dictionary themselves. In this
    case, it calls itself recursively. Objects that are not dictionaries
    are left untouched. Everything is then stored in the attribute _elem.

    Let's see an example:
    elem = {'first': 'value',
            'second': {'nested': 'dictionary'} }

    _elem will be a dictionary {'first': 'value', 'second': td}, where td is
    TransformDictionary with td._elem = {'nested': 'dictionary'}.
    """

    def __init__(self, elem, transform=lambda x:x):

        if not hasattr(elem, 'items'):
            raise TypeError("Input is not dictionary-like")

        def dict_filter(x):
            return TransformDictionary(x, transform) if isinstance(x, dict) else x

        self._elem = {k:dict_filter(v) for k,v in elem.items()}
        self._transform  = transform

    def __getitem__(self, name):
        e = self._elem[name]

        if isinstance(e, type(self)):
            # Leave TransformDictionary untouched
            return e

        return self._transform(e)

    def keys(self):
        # Like normal dictionaries
        return list(self._elem.keys())

    def __contains__(self, name):
        """This allows to use the 'in' keyword.
        """
        return name in self._elem


def pythonize_name_dict(names_list, transform=lambda x:x):
    """Take a list of names, like ['rho[0]', 'rho[1], 'energy', 'bob'] and return a
    AttributeDictionary with attributes passed through the function.

    Names that those that are not like are 'rho[0]' are set as keys (the values
    are transform(name)). Names that are like 'rho[0]', the key is set as rho,
    and the value is set a dictionary with {0: 'rho[0]'}.

    Example:
    p = pythonize_name_dict(['energy', 'rho[0]'])
    p.energy = 'energy'
    p.rho[0] = 'rho[0]'

    p.rho is a dictionary-like object.

    We will use this function with transform = __getitem__ so that p.energy will
    return the value of energy (not the key).

    """
    res = {}
    pattern = re.compile(r'^([^\[\]]+)\[([\d]+)\]$')
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
          # The setdefault() method returns the value of the item with the specified
          # key. if the key does not exist, insert the key, with the specified value
          # e.g.:
          # res.setdefault("a", {})[1] = "a[1]"
          # a -> {'a': {1: 'a[1]'}}
          res.setdefault(matched.group(1), {})[int(matched.group(2))] = name

    res = TransformDictionary(res, transform)
    return AttributeDictionary(res)
