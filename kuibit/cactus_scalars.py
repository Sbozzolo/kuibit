#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
#
# Based on code originally developed by Wolfgang Kastaun. See, GitHub,
# wokast/PyCactus/PostCactus/cactus_scalars.py
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

"""The :py:mod:`~.cactus_scalars` module provides simple interfaces to access
time series data as output by CarpetASCII, including all the reductions.

There are multiple classes defined in this module:

- :py:class`~.ScalarsDir` interfaces with :py:class:`~.SimDir` and organizes the
  data according to their reduction. This is a dictionary-like object with keys
  the possible reduction (e.g., ``max``, ``average``, ``norm2``). -
- :py:class`~.AllScalars` takes all the files that correspond to a given reduction
  and organize them according to the variables they contain.
- :py:class`~.OneScalar` represents one single scalar variable, with data that
  is represented as :py:class:`~.TimeSeries` objects. :py:class`~.AllScalars` contains
  many :py:class`~.OneScalar` objects.

These are hierarchical classes, one containing the others, so one typically ends
up with a series of brackets or dots to access the actual data. For example, if
``sim`` is a :py:class:`~.SimDir`, ``sim.ts.max['rho_b']`` is maximum of
``rho_b`` represented as :py:class:`~.TimeSeries`.

"""

import os
import re
from bz2 import open as bopen
from functools import lru_cache
from gzip import open as gopen

import numpy as np

from kuibit import simdir
from kuibit import timeseries as ts
from kuibit.attr_dict import pythonize_name_dict
from kuibit.cactus_ascii_utils import scan_header


class OneScalar:
    """Read scalar data produced by CarpetASCII.

    CactusScalarASCII is a dictionary-like object with keys the variables and
    values the :py:class:`~.TimeSeries`.

    Single variable per file or single file per group are supported. In the
    latter case, the header is inspected to understand the content of the file.
    Compressed files (gz and bz2) are supported too.

    :py:class:`~.OneScalar` represents one scalar file, there can be multiple
    variables inside, (if it was output with ``one_file_per_group = yes``).

    :ivar path: Path of the file.
    :type path: str
    :ivar folder: Path of the folder that contains the file.
    :type folder: str
    :ivar reduction_type: Type of reduction.
    :type reduction_type: str

    """

    # What is this pattern?
    # Let's understand it. We have ^ and $, so we match the entire string and
    # we have seven capturing groups.
    # 1: (\w+) matches any number of characters greater than 0 (w = word)
    # 2: ((-(\w+))|(\[\d+\]))? optionally match one of the two
    # 3: Matched - with followed by 4: any word
    # 5: Matches brackets with a number inside
    # In between match a dot (\.)
    # 6: (minimum|maximum|norm1|norm2|norm_inf|average|scalars)? optionally match one
    #    of those
    # In between match .asc (\.asc)
    # 7: (\.(gz|bz2))? optionally match .gz or .bz2

    # We want to match file names like hydrobase-press.maximum.asc or
    # hydrobase-vel[0].maximum.asc
    #
    # The .scalars. file is the one generated with the option
    # all_reductions_in_one_file

    _pattern_filename = r"""
    ^(\w+)
    ((-(\w+))|(\[\d+\]))?
    \.(minimum|maximum|norm1|norm2|norm_inf|average|scalars)?
    \.asc
    (\.(gz|bz2))?$"""

    _rx_filename = re.compile(_pattern_filename, re.VERBOSE)

    _reduction_types = {
        "minimum": "min",
        "maximum": "max",
        "norm1": "norm1",
        "norm2": "norm2",
        "norm_inf": "infnorm",
        "average": "average",
        None: "scalar",
    }

    # What function to use to open the file?
    # What mode?
    _decompressor = {
        None: (open, "r"),
        "gz": (gopen, "rt"),
        "bz2": (bopen, "rt"),
    }

    def __init__(self, path):
        """Constructor.

        Here we understand what the file contains.

        :param path: Path of the file.
        :type path: str
        """
        self.path = str(path)
        # The _vars dictionary contains a mapping between the various variables
        # and the column numbers in which they are stored.
        self._vars = {}
        self.folder, filename = os.path.split(self.path)

        filename_match = self._rx_filename.match(filename)

        if filename_match is None:
            raise RuntimeError(f"Name scheme not recognized for {filename}")

        # variable_name1 may be a thorn name (e.g. hydrobase-press)
        # Or the actual variable name (e.g. H)
        (
            variable_name1,
            _0,
            _1,
            variable_name2,
            index_in_brackets,
            reduction_type,
            _2,
            compression_method,
        ) = filename_match.groups()

        self._compression_method = compression_method

        self.reduction_type = (
            reduction_type if reduction_type is not None else "scalar"
        )

        # If the file contains multiple variables, we will scan the header
        # immediately to understand the content. If not, we scan the header
        # only when needed
        self._is_one_file_per_group = variable_name2 is not None
        self._was_header_scanned = False

        if self._is_one_file_per_group:
            # We need the variable names
            self._scan_header()
        else:
            variable_name = variable_name1
            if index_in_brackets is not None:
                variable_name += index_in_brackets
            self._vars = {variable_name: None}

    def _scan_header(self):
        # Call scan_header with the right argument

        extended_format = self.reduction_type == "scalar"

        # What method to we need to use to open the file?
        # opener can be open, gopen, or bopen depending on the extension
        # of the file
        opener, opener_mode = self._decompressor[self._compression_method]

        self._time_column, columns_info = scan_header(
            self.path,
            self._is_one_file_per_group,
            extended_format,
            opener=opener,
            opener_mode=opener_mode,
        )

        if self._is_one_file_per_group:
            self._vars.update(columns_info)
        else:
            # There is only one data_column
            self._vars = {list(self._vars.keys())[0]: columns_info}

        self._was_header_scanned = True

    @lru_cache(128)
    def load(self, variable):
        """Read file and return a TimeSeries with the requested variable.

        :param variable: Requested variable.
        :type variable: str

        :returns: :py:class:`~.TimeSeries` with requested variable as read from
                  file
        :rtype: :py:class:`~.TimeSeries`

        """
        if not self._was_header_scanned:
            self._scan_header()

        if variable not in self:
            raise ValueError(f"{variable} not available")

        column_number = self._vars[variable]
        t, y = np.loadtxt(
            self.path,
            unpack=True,
            ndmin=2,
            usecols=(self._time_column, column_number),
        )

        return ts.remove_duplicated_iters(t, y)

    def __getitem__(self, key):
        return self.load(key)

    def __contains__(self, key):
        return key in self._vars

    def keys(self):
        """Return the list of variables available.

        :returns: Variables in the file
        :rtype:   dict_keys

        """
        return self._vars.keys()


class AllScalars:
    """Helper class to read various types of scalar data in a list of files and
    properly order them. The core of this object is the ``_vars`` dictionary
    which contains the location of all the files for a specific variable and
    reduction (as :py:class:`~.OneScalar`).

    :py:class:`~.AllScalars` is a dictionary-like object, using the bracket notation
    you can access values with as TimeSeries. Alternatively, you can access the
    data as attributes of the ``fields`` attribute.

    :ivar reduction_type: Type of reduction.
    :type reduction_type: str

    """

    def __init__(self, allfiles, reduction_type):
        """Constructor.

        :param allfiles: List of all the files
        :type allfiles: list of str
        :param reduction_type: Type of reduction.
        :type reduction_type: str

        """
        self.reduction_type = str(reduction_type)

        # TODO: Is it necessary to have the folder level?
        # Probably not, so remove it

        # _vars is like _vars['variable']['folder'] -> CactusScalarASCII(f)
        # _vars['variable'] is a dictionary with as keys the folders where
        # to find the files associated to the variable and the reduction
        # reduction_type
        self._vars = {}
        for file_ in allfiles:
            # We only save those that variables are well-behaved
            try:
                cactusascii_file = OneScalar(file_)
                if cactusascii_file.reduction_type == reduction_type:
                    for var in list(cactusascii_file.keys()):
                        # We add to the _vars dictionary the mapping:
                        # [var][folder] to OneScalar(f)
                        folder = cactusascii_file.folder
                        self._vars.setdefault(var, {})[
                            folder
                        ] = cactusascii_file
            except RuntimeError:
                pass

        # What pythonize_name_dict does is to make the various variables
        # accessible as attributes, e.g. self.fields.rho
        self.fields = pythonize_name_dict(list(self.keys()), self.__getitem__)

    @lru_cache(128)
    def __getitem__(self, key):
        # We read all the files associated to variable key
        folders = self._vars[key]
        series = [f.load(key) for f in folders.values()]
        return ts.combine_ts(series)

    def __contains__(self, key):
        return key in self._vars

    def keys(self):
        """Return the available variables corresponding to the given reduction.

        :returns: Variables with given reduction
        :rtype:   dict_keys

        """
        return self._vars.keys()

    def get(self, key, default=None):
        """Return variable if available, else return the default value.

        :param key: Requested variable.
        :type key: str
        :param default: Returned value if ``variable`` is not available.
        :type default: any

        :returns: :py:class:`~.TimeSeries` of the requested variable
        :rtype: :py:class:`~.TimeSeries`

        """
        if key in self:
            return self[key]

        return default

    def __str__(self):
        ret = f"Available {self.reduction_type} timeseries:\n"
        ret += f"{list(self.keys())}\n"
        return ret


class ScalarsDir:
    """This class provides acces to various types of scalar data in a given
    simulation directory. Typically used from a :py:class:`~.SimDir` instance.
    The different scalars are available as attributes:

    :ivar scalar:    access to grid scalars.
    :ivar minimum:   access to minimum reduction.
    :ivar maximum:   access to maximum reduction.
    :ivar norm1:     access to norm1 reduction.
    :ivar norm2:     access to norm2 reduction.
    :ivar average:   access to average reduction.
    :ivar infnorm:   access to inf-norm reduction.

    Each of those works as a dictionary mapping variable names to
    :py:class:`~.TimeSeries` instances.

    """

    # TODO: Implement the following, possibly in a clean way

    # .. note::
    #    infnorm is reconstructed from min and max if infnorm
    #    itself is not available.

    def __init__(self, sd):
        """The constructor is not intended for direct use.

        :param sd: Simulation directory
        :type sd:  :py:class:`~.SimDir` instance.
        """
        if not isinstance(sd, simdir.SimDir):
            raise TypeError("Input is not SimDir")

        self.path = sd.path
        self.point = AllScalars(sd.allfiles, "scalar")
        self.scalar = AllScalars(sd.allfiles, "scalar")
        self.minimum = AllScalars(sd.allfiles, "minimum")
        self.maximum = AllScalars(sd.allfiles, "maximum")
        self.norm1 = AllScalars(sd.allfiles, "norm1")
        self.norm2 = AllScalars(sd.allfiles, "norm2")
        self.average = AllScalars(sd.allfiles, "average")
        self.infnorm = AllScalars(sd.allfiles, "infnorm")

        # Aliases
        self.max = self.maximum
        self.min = self.minimum

    def __getitem__(self, reduction):
        return getattr(self, reduction)

    def get(self, key, default=None):
        """Return a reduction if available, else return the default value.

        :param key: Requested reduction.
        :type key: str
        :param default: Returned value if ``reduction`` is not available.
        :type default: any

        :returns: Collection of all the variables with a given reduction.
        :rtype: :py:class:`~.AllScalars`

        """
        if key in [
            "point",
            "scalar",
            "minimum",
            "maximum",
            "norm1",
            "norm2",
            "average",
            "infnorm",
        ]:
            return self[key]

        return default

    def __str__(self):
        return "Folder %s\n%s\n%s\n%s\n%s\n%s\n%s\n" % (
            self.path,
            self.scalar,
            self.minimum,
            self.maximum,
            self.norm1,
            self.norm2,
            self.average,
        )
