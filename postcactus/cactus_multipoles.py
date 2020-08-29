#!/usr/bin/env python3

"""This module provides access to data saved by the multipoles thorn.

   The main class is :py:class:`CactusMultipoleDir`, which is typically
   accessed through :py:class:`~.SimDir` instances.

"""

from postcactus import timeseries
# import numpy as np
# import h5py
# import re
# import os


class MultipoleDet:
    """This class collects multipole components of a specific variable
    a given spherical surface.

    Multipoles are tightly connected with gravitational waves, so morally a
    sphere where multipoles are computed is a "detector". Hence, the name
    of the class.

    It works as a dictionary in terms of the component as a tuple (l,m),
    returning a :py:class:`~.TimeSeries` object. Alternatively, it can be
    called as a function(l,m). Iteration is supported and yields tuples
    (l, m, data), which can be used to loop through all the multipoles
    available.

    :ivar dist: Radius of the sphere
    :vartype dist: float
    :ivar radius: Radius of the sphere
    :vartype dist: float
    :ivar available_l: Available l values
    :vartype available_l: list
    :ivar available_m: Available m values
    :ivar available_m: list
    :ivar available_lm: Available (l, m) values
    :ivar available_lm: list of tuples

    """

    def __init__(self, dist, data):
        """ Data is a list of tuples with (l, m, timeseries).
        """

        self.dist = float(dist)
        self.radius = self.dist  # This is just an alias
        # Associate multipoles to list of timeseries
        multipoles_list_ts = {}

        # Now we populate the multipoles_ts_list dictionary. This is a
        # dictionary with keys (l, m) and with values lists of timeseries. If
        # the key is not already present, we create it with value an empty
        # list, then we append the timeseries to this list
        for mult_l, mult_m, ts in data:  # mult = "multipole"
            lm_list = multipoles_list_ts.setdefault((mult_l, mult_m), [])
            # This means: multipoles[(mult_t, mult_m)] = []
            lm_list.append(ts)
            # At the end we have:
            # multipoles[(mult_t, mult_m)] = [ts1, ts2, ...]

        # Now self._multipoles is a dictionary in which all the timeseries are
        # collapse in a single one. So it is a straightforward map (l, m) -> ts
        self._multipoles = {lm: timeseries.combine_ts(ts)
                            for lm, ts in multipoles_list_ts.items()}
        self.available_l = {mult_l for mult_l, _ in self._multipoles.keys()}
        self.available_m = {mult_m for _, mult_m in self._multipoles.keys()}
        self.available_lm = list(self._multipoles.keys())

    def __contains__(self, key):
        return key in self._multipoles

    def __getitem__(self, key):
        return self._multipoles[key]

    def __call__(self, mult_l, mult_m):
        return self[(mult_l, mult_m)]

    def __eq__(self, other):
        if (not isinstance(other, type(self))):
            return False
        return (self.dist == other.dist and
                self._multipoles == other._multipoles)

    def __iter__(self):
        for (mult_l, mult_m), ts in self._multipoles.items():
            yield mult_l, mult_m, ts

    def __len__(self):
        return len(self._multipoles)

    def keys(self):
        return self.available_lm


class MultipoleAllDets:
    """This class collects available surfaces with multipole data.

    It works as a dictionary in terms of spherical surface radius,
    returning a :py:class:`MultipoleDet` object. Iteration is supported,
    sorted by ascending radius. You can iterate over all the radii and
    all the available l and m with a nested loop.

    :ivar radii:        Available surface radii.
    :vartype radii: float
    :ivar r_outer:      Radius of the outermost detector
    :vartype r_outer: float
    :ivar outermost:    Outermost detector
    :vartype outermost: :py:class:`~MultipoleDet`
    :ivar available_lm: Available components as tuple (l,m).
    :vartype available_lm: list of tuples
    :ivar available_l:  List of available "l".
    :vartype available_l: list
    :ivar available_m:  List of available "m".
    :vartype available_m: list
    """

    def __init__(self, data):
        """Data is a list of tuples with structure
        (multipole_l, multipole_m, extraction_radius, [timeseries])
        """
        detectors = {}
        self.available_lm = set()

        # Populate detectors
        # detectors is a dictionary with keys the radii and
        # items that look like ([l, m, [timeseries], ...]).
        # We accumulate in the list all the ones for the same
        # radius
        for mult_l, mult_m, radius, ts in data:
            # If we don't have the radius yet, let's create an empty list
            d = detectors.setdefault(radius, [])
            # Add the new values
            d.append((mult_l, mult_m, ts))
            # Tally the available l and m
            self.available_lm.add((mult_l, mult_m))

        self._detectors = {radius: MultipoleDet(radius, multipoles)
                           for radius, multipoles in detectors.items()}

        # In Python3 .keys() is not a list
        self.radii = sorted(list(self._detectors.keys()))

        if len(self.radii) > 0:
            self.r_outer = self.radii[-1]
            self.outermost = self._detectors[self.r_outer]
        #
        self.available_l = {mult_l for mult_l, _ in self.available_lm}
        self.available_m = {mult_m for _, mult_m in self.available_lm}

        # Alias
        self._dets = self._detectors

    def __contains__(self, key):
        return key in self._dets

    def __getitem__(self, key):
        return self._dets[key]

    def __iter__(self):
        for r in self.radii:
            yield self[r]

    def __eq__(self, other):
        if (not isinstance(other, type(self))):
            return False
        return (self.radii == other.radii and
                self._dets == other._dets)

    def __len__(self):
        return len(self._dets)

    def keys(self):
        return self._dets.keys()


# class CactusMultipoleDir:
#     """This class provides acces to various types of multipole data in a given
#     simulation directory.

#     Files are lazily loaded.

#     If both h5 and ASCII are present, h5 are preferred. There's no attempt to
#     combine the two.
#     """

#     def __init__(self, sd):
#         # self._vars is a dictionary. The keys are the variables,
#         # and the items are lists of tuples of the form
#         # (multipole_l, multipole_m, radius, filename) for text
#         # files and
#         # (0, 0, -1, filename) for h5 files (since we have to read
#         # the content to find the l and m)
#         self._vars = {}

#         # First, we need to find the multipole files.
#         # There are text files and h5 files
#         #
#         # We use a regular expressions on the name
#         # The structure is like mp_Psi4_l_m2_r110.69.asc
#         #
#         # Let's understand the regexp:
#         # 0. ^ and $ means that we match the entire name
#         # 1. We match mp_ followed my the variable name, which is
#         #    any combination of characters
#         # 2. We match _l with a number
#         # 3. We match _m with possibly a minus sign and a number
#         # 4. We match _r with any combination of numbers with possibly
#         #    dots
#         # 5. We possibly match a compression
#         rx_txt = re.compile(r"""^
#         mp_([a-zA-Z0-9\[\]_]+)
#         _l(\d+)
#         _m([-]?\d+)
#         _r([0-9.]+)
#         .asc
#         (?:.bz2|.gz)?
#         $""", re.VERBOSE)

#         # For h5 files is easy: it is just the var name
#         rx_h5 = re.compile(r'^mp_([a-zA-Z0-9\[\]_]+).h5$')

#         for f in sd.allfiles:
#             filename = os.path.split(f)[1]
#             matched_h5 = rx_h5.match(filename)
#             matched_txt = rx_txt.match(filename)
#             if (matched_h5 is not None):
#                 variable_name = matched_h5.group(1).lower()
#                 var_list = self._vars.setdefault(variable_name, [])
#                 # We are flagging that this h5
#                 var_list.append((0, 0, -1, filename))
#             elif (matched_txt is not None):
#                 variable_name = matched_txt.group(1).lower()
#                 mult_l = int(matched_txt.group(2))
#                 mult_m = int(matched_txt.group(3))
#                 radius = float(matched_txt.group(4))
#                 var_list = self._vars.setdefault(variable_name, [])
#                 var_list.append((mult_l, mult_m, radius,
#                                  filename))

#     def __contains__(self, key):
#         return str(key).lower() in self._vars

#     def _multipole_from_textfile(self, path):
#         a = np.loadtxt(path, unpack=True, ndmin=2)
#         if ((len(a) != 3)):
#             raise RuntimeError('Wrong format')
#         mp = a[1] + 1j * a[2]
#         return timeseries.remove_duplicate_iters(a[0], mp)

#     def _multipoles_from_textfiles(self, mpfiles):
#         amp = [(mult_l, mult_m, radius, self._multipole_from_textfile(filename))
#                for mult_l, mult_m, radius, filename in mpfiles]

#         return MultipoleAllDets(amp)

#     def _multipoles_from_h5file(self, mpfile):
#         mpfiles = h5py.File(mpfile, 'r')
#         amp = []
#         # This regexp matched : l(number)_m(-number)_r(number)
#         fieldname_pattern = re.compile(r'l(\d+)_m([-]?\d+)_r([0-9.]+)')

#         for n in mpfile.keys():
#             matched = fieldname_pattern.match(n)
#             if not matched:
#                 continue

#             mult_l = int(matched.group(1))
#             mult_m = int(matched.group(2))
#             radius = float(matched.group(3))
#             a = mpfile[n][()]
#             ts = timeseries.TimeSeries(a[:, 0], a[:, 1] + 1j*a[:, 2])
#             amp.append((mult_l, mult_m, radius, ts))

#     def _multipoles_from_h5files(self, mpfiles):
#         mult_l = []

#         for filename in mpfiles:
#             mult_l.extend(self._mp_from_h5file(filename))

#         return MultipoleAllDets(mult_l)

#     def __getitem__(self, key):
#         k = str(key).lower()
#         if self._vars[k][2] == -1:  # When the radius is -1 it's h5
#             return self._multipoles_from_h5files(self._vars[k])
#         else:
#             return self._multipoles_from_textfiles(self._vars[k])

#     def get(self, key, default=None):
#         if key not in self:
#             return default
#         return self[key]

#     def keys(self):
#         return list(self._vars.keys())

#     def __str__(self):
#         pass
