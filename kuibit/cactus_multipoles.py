#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
#
# Based on code originally developed by Wolfgang Kastaun. This file may contain
# algorithms and/or structures first implemented in
# GitHub:wokast/PyCactus/PostCactus/cactus_multipoles.py
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

"""This module provides access to data saved by the Multipole thorn.

There are multiple classes defined in this module:

- :py:class`~.MultipolesDir` interfaces with :py:class:`~.SimDir` and organizes the
  data according to the variable available. This is a dictionary-like object with keys
  the variable names.
- :py:class`~.MultipoleAllDets` takes all the files that correspond to a given variable
  and organize them according to the extraction radius.
- :py:class`~.MultipoleOneDet` represents one single extraction radius, for a single
  variable. It is a dictionary-like object with keys the multipolar numbers and values
  the multipolar decomposition represented as represented as :py:class:`~.TimeSeries`
  objects.

These are hierarchical classes, one containing the others, so one typically ends
up with a series of brackets or dots to access the actual data. For example, if
``sim`` is a :py:class:`~.SimDir`, ``sim.multipoles['rho_b'][100][2,2]`` is
(2,2) decomposition of ``rho_b`` at radius 100 represented as
:py:class:`~.TimeSeries`.

"""

import os
import re
from typing import Optional

import h5py
import numpy as np

from kuibit import timeseries
from kuibit.attr_dict import pythonize_name_dict


class MultipoleOneDet:
    """This class collects multipole components of a specific variable
    a given spherical surface.

    Multipoles are tightly connected with gravitational waves, so morally a
    sphere where multipoles are computed is a "detector". Hence, the name
    of the class.

    :py:class:`~.MultipoleOneDet` is a dictionary-like object with components as
    the tuples (l,m) and values the corresponding multipolar decomposition as a
    :py:class:`~.TimeSeries` object. Alternatively, this can also be called
    directly ``multipoleonedet(l,m)``. Iteration is supported and yields tuples
    ``(l, m, data)``, which can be used to loop through all the multipoles
    available.

    The reason we allow for ``l_min`` is to remove those files that are not
    necessary when considering gravitational waves (l<2).

    :ivar dist: Radius of the sphere.
    :vartype dist: float
    :ivar radius: Radius of the sphere.
    :type radius: float
    :ivar l_min: l smaller than ``l_min`` are dropped.
    :type l_min: int
    :ivar available_l: Available l values.
    :type available_l: set
    :ivar available_m: Available m values.
    :ivar available_m: set
    :ivar available_lm: Available ``(l, m)`` values.
    :ivar available_lm: set of tuples
    :ivar missing_lm: Missing (l, m) values to have all from ``l_min`` to ``l_max``.
    :ivar missing_lm: set of tuples

    """

    def __init__(self, dist, data, l_min=0):
        """Constructor.

        :param dist: Radius of the spherical surface.
        :type dist: float
        :param data: List of tuples with the two multipolar numbers and
                     the data as :py:class:`~.TimeSeries`.
        :type data: list of tuple ``(l, m, timeseries)``
        :ivar l_min: l smaller than ``l_min`` are dropped.
        :type l_min: int

        """

        self.dist = float(dist)
        self.radius = self.dist  # This is just an alias

        self.l_min = l_min

        # Associate multipoles to list of timeseries
        multipoles_list_ts = {}

        # Now we populate the multipoles_ts_list dictionary. This is a
        # dictionary with keys (l, m) and with values lists of timeseries. If
        # the key is not already present, we create it with value an empty
        # list, then we append the timeseries to this list
        for mult_l, mult_m, ts in data:  # mult = "multipole"
            if mult_l >= l_min:
                lm_list = multipoles_list_ts.setdefault((mult_l, mult_m), [])
                # This means: multipoles[(mult_t, mult_m)] = []
                lm_list.append(ts)
                # At the end we have:
                # multipoles[(mult_t, mult_m)] = [ts1, ts2, ...]

        # Now self._multipoles is a dictionary in which all the timeseries are
        # collapse in a single one. So it is a straightforward map (l, m) -> ts
        self._multipoles = {
            lm: timeseries.combine_ts(ts)
            for lm, ts in multipoles_list_ts.items()
        }
        self.available_l = sorted(
            {mult_l for mult_l, _ in self._multipoles.keys()}
        )
        self.l_max = max(self.available_l)
        self.available_m = sorted(
            {mult_m for _, mult_m in self._multipoles.keys()}
        )
        self.available_lm = set(self._multipoles.keys())

        # Check if all the (l, m) from l_min to l_max are available
        all_lm = set()
        for mult_l in range(self.l_min, max(self.available_l) + 1):
            for mult_m in range(-mult_l, mult_l + 1):
                all_lm.add((mult_l, mult_m))

        # set subtraction
        self.missing_lm = all_lm - self.available_lm

        # Data is in the format expected by __init__
        self.data = [(lm[0], lm[1], ts) for lm, ts in self._multipoles.items()]

    def copy(self):
        """Return a deep copy.

        :returns: Deep copy of ``self``.
        :rtype: :py:class:`~.MultipoleOneDet`
        """
        return type(self)(self.dist, self.data, self.l_min)

    def crop(self, init: Optional[float] = None, end: Optional[float] = None):
        """Remove all the data before ``init`` and after ``end``.

        If ``init`` or ``end`` are not specified, do not crop on that side.
        """
        for _, _, ts in self.data:
            ts.crop(init=init, end=end)

    def cropped(
        self, init: Optional[float] = None, end: Optional[float] = None
    ):
        """Return a copy where the data is cropped in the provided interval.

        If ``init`` or ``end`` are not specified, do not crop on that side.

        :returns: Copy of ``self`` with data cropped.
        :rtype: :py:class:`~.MultipoleOneDet`
        """
        ret = self.copy()
        ret.crop(init=init, end=end)
        return ret

    def __contains__(self, key):
        return key in self._multipoles

    def __getitem__(self, key):
        return self._multipoles[key]

    def __call__(self, mult_l, mult_m):
        return self[(mult_l, mult_m)]

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self.dist == other.dist and self._multipoles == other._multipoles
        )

    # From Python's docs: In order to conform to the object model, classes that
    # define their own equality method should also define their own hash method,
    # or be unhashable.

    # Since we consider series unhashable, this object also has to be unhashable.
    __hash__ = None

    def __iter__(self):
        for (mult_l, mult_m), ts in sorted(self._multipoles.items()):
            yield mult_l, mult_m, ts

    def __len__(self):
        return len(self._multipoles)

    def keys(self):
        """Return available multipolar numbers.

        :returns: Available multipolar numbers.
        :rtype: dict_keys
        """
        return self._multipoles.keys()

    def __str__(self):
        ret = f"(l, m) available: {list(self.keys())}"
        if self.missing_lm:
            ret += f" (missing: {list(self.missing_lm)})"
        return ret

    def total_function_on_available_lm(
        self, function, *args, l_max=None, **kwargs
    ):
        """Evaluate ``function`` on each multipole and accumulate the result.

        ``total_function_on_available_lm`` will call ``function`` with the
        following arguments:

        ``function(timeseries, mult_l, mult_m, dist, *args, **kwargs)``

        If ``function`` does not need some paramters, it should use take
        the ``*args`` argument to ignore the additional paramters that
        are always passed ``(l, m, r)``.

        Values of l larger than ``l_max`` are ignored.

        This method is used to compute quantities like the total power in
        gravitational waves.

        ``function`` can take additional paramters passed directly from
        ``total_function_on_available_lm`` (e.g. ``pcut`` for FFI).

        :params function: Function that has to be applied on each multipole.
        :type function: callable

        :returns: Sum of function applied to each monopole
        :rtype: return type of function

        """
        # This function is used to compute many quantities with waves (e.g.,
        # total strain, total power emitted, ...)
        # It is a little bit ugly, but it works
        if l_max is None:
            l_max = self.l_max

        if l_max > self.l_max:
            raise ValueError("l max larger than l available")

        # The iterator is increasing in (l, m), so we can start from the first
        # element

        iter_self = iter(self)
        try:
            first_l, first_m, first_det = next(iter_self)
        except StopIteration:
            raise RuntimeError("No multipole moments available")
        if first_l > l_max:
            raise ValueError("l max smaller than all l available")

        result = function(
            first_det, first_l, first_m, self.dist, *args, **kwargs
        )

        for mult_l, mult_m, det in iter_self:
            if mult_l <= l_max:
                result += function(
                    det, mult_l, mult_m, self.dist, *args, **kwargs
                )

        return result


class MultipoleAllDets:
    """This class collects available surfaces with multipole data.

    It is a dictionary-like object with keys the spherical surface radius, and
    values :py:class:`MultipoleOneDet` object. Iteration is supported, sorted by
    ascending radius. You can iterate over all the radii and all the available l
    and m with a nested loop.

    :ivar radii:        Available surface radii.
    :type radii: float
    :ivar r_outer:      Radius of the outermost detector.
    :type r_outer: float
    :ivar l_min: l smaller than l_min are dropped.
    :type l_min: int
    :ivar outermost:    Outermost detector.
    :type outermost: :py:class:`~MultipoleOneDet`
    :ivar available_lm: Available components as tuple (l,m).
    :type available_lm: list of tuples
    :ivar available_l:  List of available l.
    :type available_l: list
    :ivar available_m:  List of available m.
    :type available_m: list

    """

    def __init__(self, data, l_min=0):
        """Constructor.

        :param data: List of tuples with ``(multipole_l, multipole_m,
                     extraction_radius, [timeseries])``, where
                     ``[timeseries]`` is a list of the :py:class:`~.TimeSeries`
                     associated.
        :type data: list of tuples

        """
        self.l_min = l_min

        detectors = {}
        self.available_lm = set()

        # Populate detectors
        # detectors is a dictionary with keys the radii and
        # items that look like ([l, m, timeseries, ...]).
        # We accumulate in the list all the ones for the same
        # radius
        for mult_l, mult_m, radius, ts in data:
            if mult_l >= self.l_min:
                # If we don't have the radius yet, let's create an empty list
                d = detectors.setdefault(radius, [])
                # Add the new values
                d.append((mult_l, mult_m, ts))
                # Tally the available l and m
                self.available_lm.add((mult_l, mult_m))

        self._detectors = {
            radius: MultipoleOneDet(radius, multipoles, self.l_min)
            for radius, multipoles in detectors.items()
        }

        # In Python3 .keys() is not a list
        self.radii = sorted(list(self._detectors.keys()))

        if len(self.radii) > 0:
            self.r_outer = self.radii[-1]
            self.outermost = self._detectors[self.r_outer]
        #
        self.available_l = sorted({mult_l for mult_l, _ in self.available_lm})
        if self.available_l:
            self.l_max = max(self.available_l)
        else:
            self.l_max = None
        self.available_m = sorted({mult_m for _, mult_m in self.available_lm})

        # Alias
        self._dets = self._detectors

        # Data is in the format expected by __init__
        # (multipole_l, multipole_m, extraction_radius, [timeseries])
        self.data = []
        for radius, det in self._dets.items():
            for mult_l, mult_m, ts in det:
                self.data.append((mult_l, mult_m, radius, ts))

    def copy(self):
        """Return a deep copy.

        :returns: Deep copy of ``self``.
        :rtype: :py:class:`~.MultipoleAllDets`
        """
        return type(self)(self.data, self.l_min)

    def has_detector(self, mult_l, mult_m, dist):
        """Check if a given multipole component extracted at a given
        distance is available.

        :param mult_l:     Multipole component l.
        :type mult_l:      int
        :param mult_m:     Multipole component m.
        :type mult_m:      int
        :param dist:  Distance of the detector.
        :type dist:   float

        :returns:     If available or not.
        :rtype:       bool
        """
        if dist in self:
            return (mult_l, mult_m) in self[dist]
        return False

    def __contains__(self, key):
        return key in self._dets

    def __getitem__(self, key):
        return self._dets[key]

    def __iter__(self):
        for r in self.radii:
            yield self[r]

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.radii == other.radii and self._dets == other._dets

    # From Python's docs: In order to conform to the object model, classes that
    # define their own equality method should also define their own hash method,
    # or be unhashable.

    # Since we consider series unhashable, this object also has to be unhashable.
    __hash__ = None

    def __len__(self):
        return len(self._dets)

    def keys(self):
        """Return available extraction radii.

        :returns: Available extraction radii.
        :rtype: dict_keys
        """
        return self._dets.keys()

    def __str__(self):
        ret = f"Avilable radii: {list(self.keys())}\n\n"
        for d in sorted(self.keys()):
            ret += f"At radius {d}, {self._dets[d]}\n"
        return ret


class MultipolesDir:
    """This class provides acces to various types of multipole data in a given
    simulation directory.

    This class is like a dictionary, you can access its values using the
    brackets operator, with values that are :py:mod:`~.MultipoleAllDets`. These
    contain the full multipolar description for all the available radii. Files
    are lazily loaded. If both HDF5 and ASCII are present, HDF5 are preferred.
    There's no attempt to combine the two. Alternatively, you can access
    variables with ``get`` or with ``fields.var_name``.

    """

    def __init__(self, sd):
        """Constructor.

        :param sd: Simulation directory.
        :type sd: :py:class:`~.SimDir`
        """
        # self._vars_*_files are dictionary. For _vars_ascii_files, the keys are
        # the variables and the items are sets of tuples of the form
        # (multipole_l, multipole_m, radius, filename) for text files.
        #
        # For example:
        # self._vars_ascii_files =
        # {'psi4': {(2, 2, 110.69, 'output-0000/mp_Psi4_l_m2_r110.69.asc'),
        #           (2, 2, 110.69, 'output-0001/mp_Psi4_l_m2_r110.69.asc')}}

        self._vars_ascii_files = {}

        # For _vars_h5_files, the keys are still the variables, but values are
        # sets with only the files (and not tuples), since we have to read the
        # content to find the l and m
        #
        # For example
        # self._vars_h5_files =
        # {'psi4': {'output-0000/mp_Psi4.h5', 'output-0001/mp_Psi4.h5'}}
        self._vars_h5_files = {}

        # self._vars is the dictionary where we cache the results. The keys are
        # the variables, the values are the corresponding MultipoleAllDets
        # objects. We fill this with __getitem__
        self._vars = {}

        # First, we need to find the multipole files.
        # There are text files and h5 files
        #
        # We use a regular expressions on the name
        # The structure is like mp_Psi4_l_m2_r110.69.asc
        #
        # Let's understand the regexp:
        # 0. ^ and $ means that we match the entire name
        # 1. We match mp_ followed my the variable name, which is
        #    any combination of characters
        # 2. We match _l with a number
        # 3. We match _m with possibly a minus sign and a number
        # 4. We match _r with any combination of numbers with possibly
        #    dots
        # 5. We possibly match a compression
        rx_ascii = re.compile(
            r"""^
        mp_([a-zA-Z0-9\[\]_]+)
        _l(\d+)
        _m([-]?\d+)
        _r([0-9.]+)
        .asc
        (?:.bz2|.gz)?
        $""",
            re.VERBOSE,
        )

        # For h5 files is easy: it is just the var name
        rx_h5 = re.compile(r"^mp_([a-zA-Z0-9\[\]_]+).h5$")

        for f in sd.allfiles:
            filename = os.path.split(f)[1]
            matched_h5 = rx_h5.match(filename)
            matched_ascii = rx_ascii.match(filename)
            if matched_h5 is not None:
                variable_name = matched_h5.group(1).lower()
                var_list = self._vars_h5_files.setdefault(variable_name, set())
                # We are flagging that this h5
                var_list.add(f)
            elif matched_ascii is not None:
                variable_name = matched_ascii.group(1).lower()
                mult_l = int(matched_ascii.group(2))
                mult_m = int(matched_ascii.group(3))
                radius = float(matched_ascii.group(4))
                var_list = self._vars_ascii_files.setdefault(
                    variable_name, set()
                )
                var_list.add((mult_l, mult_m, radius, f))

        # What pythonize_name_dict does is to make the various variables
        # accessible as attributes, e.g. self.fields.rho
        self.fields = pythonize_name_dict(list(self.keys()), self.__getitem__)

    def __contains__(self, key):
        return str(key).lower() in self.keys()

    # The following are staticmethods because they do not depend on the bound
    # object. Using this decorator we save memory because Python will
    # initialize them only once.

    @staticmethod
    def _multipole_from_textfile(path):
        """Read multipole data from a text file.

        :param path: File to read.
        :type path: str

        :returns: Multipole data.
        :rtype: :py:class:`~.TimeSeries`
        """
        a = np.loadtxt(path, unpack=True, ndmin=2)
        if len(a) != 3:
            raise RuntimeError(f"Wrong format in {path}")
        complex_mp = a[1] + 1j * a[2]
        return timeseries.remove_duplicated_iters(a[0], complex_mp)

    @staticmethod
    def _multipoles_from_h5file(path):
        """Read multipole data from a HDF5 file.

        :param path: File to read.
        :type path: str
        :returns: Multipole data.
        :rtype: :py:class:`~.TimeSeries`
        """
        alldets = []
        # This regex matches : l(number)_m(-number)_r(number)
        fieldname_pattern = re.compile(r"l(\d+)_m([-]?\d+)_r([0-9.]+)")

        try:
            with h5py.File(path, "r") as data:
                # Loop over the groups in the hdf5
                for entry in data.keys():
                    matched = fieldname_pattern.match(entry)
                    if matched:
                        mult_l = int(matched.group(1))
                        mult_m = int(matched.group(2))
                        radius = float(matched.group(3))
                        # Read the actual data
                        a = data[entry][()].T
                        complex_mp = a[1] + 1j * a[2]
                        ts = timeseries.remove_duplicated_iters(
                            a[0], complex_mp
                        )
                        alldets.append((mult_l, mult_m, radius, ts))
        except RuntimeError as exce:
            raise RuntimeError(f"File {data} cannot be processed") from exce

        return alldets

    def _multipoles_from_textfiles(self, mpfiles):
        """Read all the multipole data in several text files.

        :param mpfiles: Files to read.
        :type mpfiles: list of str

        :returns: :py:class:`~.MultipoleAllDets` with all the data read.
        :rtype: :py:class:`~.MultipoleAllDets`
        """
        # We prepare the data for MultipoleAllDets checking for errors
        alldets = [
            (
                mult_l,
                mult_m,
                radius,
                self._multipole_from_textfile(filename),
            )
            for mult_l, mult_m, radius, filename in mpfiles
        ]
        return MultipoleAllDets(alldets)

    def _multipoles_from_h5files(self, mpfiles):
        """Read all the multipole data in several HDF5 files.

        :param mpfiles: Files to read.
        :type mpfiles: list of str

        :returns: :py:class:`~.MultipoleAllDets` with all the data read.
        :rtype: :py:class:`~.MultipoleAllDets`
        """
        mult_l = []

        for filename in mpfiles:
            mult_l.extend(self._multipoles_from_h5file(filename))

        return MultipoleAllDets(mult_l)

    def __getitem__(self, key):
        """Read data associated to variable ``key``.

        HDF5 files are preferred over ASCII ones.

        :returns: Multipolar data.
        :rtype: :py:class:`~.MultipoleAllDets`

        """
        k = str(key).lower()

        # If we don't have already cached the result, we first read it.

        if k not in self._vars:
            # We prefer h5
            if k in self._vars_h5_files:
                self._vars[k] = self._multipoles_from_h5files(
                    self._vars_h5_files[k]
                )
            elif k in self._vars_ascii_files:
                self._vars[k] = self._multipoles_from_textfiles(
                    self._vars_ascii_files[k]
                )
            else:
                raise KeyError

        return self._vars[k]

    def get(self, key, default=None):
        """Return a the multipolar data for the given variable if available, else return
        the default value.

        :param key: Requested variable.
        :type key: str
        :param default: Returned value if ``key`` is not available.
        :type default: any

        :returns: Collection of all the multipolar data for the given variable.
        :rtype: :py:class:`~.MultipoleAllDets`

        """

        if key not in self:
            return default
        return self[key]

    def keys(self):
        """Return available variables with multipolar data.

        :returns: Available variables that have multipolar data.
        :rtype: dict_keys
        """
        # We merge the dictionaries and return the keys.
        # This automatically takes care of making sure that they keys are unique.
        return {**self._vars_h5_files, **self._vars_ascii_files}.keys()

    def __str__(self):
        """NOTE: __str__ requires opening all the h5 files! This can be slow!"""
        ret = f"Variables available: {self.keys()}\n"
        for variable in self.keys():
            ret += f"For variable {variable}:\n\n"
            ret += f"{self[variable]}\n"
        return ret
