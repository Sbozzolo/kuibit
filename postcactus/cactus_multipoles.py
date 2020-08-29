#!/usr/bin/env python3

"""This module provides access to data saved by the multipoles thorn.

   The main class is :py:class:`CactusMultipoleDir`, which is typically
   accessed through :py:class:`~.SimDir` instances.

"""

from postcactus import timeseries


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

    def __iter__(self):
        for (mult_l, mult_m), ts in self._multipoles.items():
            yield mult_l, mult_m, ts

    def __len__(self):
        return len(self._multipoles)

    def keys(self):
        return self.available_lm
