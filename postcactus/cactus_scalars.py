#!/usr/bin/env python3
"""The :py:mod:`~.cactus_scalars` module provides functions to load
timeseries in Cactus formats and a class :py:class:`ScalarsDir` for easy
access to all timeseries in a Cactus simulation directory. This module
is normally not used directly, but from the :py:mod:`~.simdir` module.
The data loaded by this module is represented as
:py:class:`~.TimeSeries` objects.
"""

import os
import re
from bz2 import open as bopen
from functools import lru_cache
from gzip import open as gopen

import numpy as np

from postcactus import simdir
from postcactus import timeseries as ts
from postcactus.attr_dict import pythonize_name_dict


class OneScalar:
    """Read scalar data produced by CarpetASCII.

    CactusScalarASCII is a dictionary-like object: it has keys() and you can
    TimeSeries using the ['key'] syntax.

    Single variable per file or single file per group are supported. In the
    latter case, the header is inspected to understand the content of the file.

    Compressed files (gz and bz2) are supported.

    """

    # What is this pattern?
    # Let's understand it. We have ^ and $, so we match the entire string and
    # we have seven capturing groups.
    # 1: (\w+) matches any number of characters greater than 0 (w = word)
    # 2: ((-(\w+))|(\[\d+\]))? optionally match one of the two
    # 3: Matched - with followed by 4: any word
    # 5: Matches brackets with a number inside
    # In between match a dot (\.)
    # 6: (minimum|maximum|norm1|norm2|norm_inf|average)? optionally match one
    #    of those
    # In between match .asc (\.asc)
    # 7: (\.(gz|bz2))? optionally match .gz or .bz2

    # We want to match file names like hydrobase-press.maximum.asc or
    # hydrobase-vel[0].maximum.asc

    _pattern_filename = r"""
    ^(\w+)
    ((-(\w+))|(\[\d+\]))?
    \.(minimum|maximum|norm1|norm2|norm_inf|average)?
    \.asc
    (\.(gz|bz2))?$"""

    # Example of data column:
    # 1 2 3 are always: 1:iteration 2:time 3:data
    # data columns: 3:kxx 4:kxy 5:kxz 6:kyy 7:kyz 8:kzz
    _pattern_data_columns = r"^# data columns: (.+)$"

    # Example of column format:
    # column format: 1:it 2:tl 3:rl 4:c 5:ml 6:ix 7:iy 8:iz 9:time 10:x 11:y
    # 12:z 13:data
    _pattern_column_format = r"^# column format: (.+)$"

    # Here we match (number):(word[number])
    # We are matching expressions like 3:kxx
    _pattern_columns = r"^(\d+):(\w+(\[\d+\])?)$"

    _rx_filename = re.compile(_pattern_filename, re.VERBOSE)
    _rx_data_columns = re.compile(_pattern_data_columns)
    _rx_column_format = re.compile(_pattern_column_format)
    _rx_columns = re.compile(_pattern_columns)

    _reduction_types = {
        'minimum': 'min',
        'maximum': 'max',
        'norm1': 'norm1',
        'norm2': 'norm2',
        'norm_inf': 'infnorm',
        'average': 'average',
        None: 'scalar'
    }

    # How many lines do we read as header?
    _header_line_number = 20

    # What function to use to open the file?
    # What mode?
    _decompressor = {None: (open, 'r'),
                     'gz': (gopen, 'rt'),
                     'bz2': (bopen, 'rt')}

    def __init__(self, path):
        self.path = str(path)
        # The _vars dictionary contains a mapping between the various variable
        # and the column numbers in which they are stored.
        self._vars = {}
        self.folder, filename = os.path.split(self.path)

        filename_match = self._rx_filename.match(filename)

        if filename_match is None:
            raise RuntimeError(f"Name scheme not recognized for {filename}")

        # variable_name1 may be a thorn name (e.g. hydrobase-press)
        # Or the actual variable name (e.g. H)
        (variable_name1, _0, _1, variable_name2, index_in_brackets,
         reduction_type, _2, compression_method) = filename_match.groups()

        self._compression_method = compression_method

        self.reduction_type = reduction_type if reduction_type is not None \
            else "scalar"

        # If the file contains multiple variables, we will scan the header
        # immediately to understand the content. If not, we scan the header
        # only when needed
        self._is_one_file_per_group = (variable_name2 is not None)
        self._was_header_scanned = False

        if self._is_one_file_per_group:
            # We need the variable names
            self._scan_header()
        else:
            variable_name = variable_name1
            if (index_in_brackets is not None):
                variable_name += index_in_brackets
            self._vars = {variable_name: None}

    def _scan_strings_for_columns(self, strings, pattern):
        """Match each string in strings against pattern and each matching
        result against _pattern_columns. Then, return a dictionary that maps
        variable to column number.

        """

        # We scan these lines and see if any matches with the regexp for the
        # column format
        for line in strings:
            matched_pattern = pattern.match(line)
            if (matched_pattern is not None):
                break
        # Here we are using an else clause with a for loop. This else
        # branch is reached only if the break in the for loop is never
        # called. In this case, this happens if we never match the
        # column format
        else:
            raise RuntimeError(f"Missing column information in {self.path}")

        # Here we should have matched the column format. It should be
        # like:
        # column format: 1:it 2:tl 3:rl 4:c 5:ml 6:ix 7:iy 8:iz 9:time
        # 10:x 11:y 12:z 13:data

        # Let's make sure that this is the case.
        #
        # In matched_column_format.groups()[0] we have the matched
        # expression (there is only one group). The different column
        # meanings are separated by a space, so
        # matched_column_format.groups()[0].split() is a list with the
        # various subgroups. Each has to match against self.rx_columns.
        # So, to check that they all match we apply rx_columns.match()
        # to each element and see if they are all not None with the
        # all() function

        columns = list(map(self._rx_columns.match,
                           matched_pattern.groups()[0].split()))

        are_real_columns = all(columns)

        if not are_real_columns:
            raise RuntimeError(f"Bad header in {self.path}")

        # Columns are good. Let's create a dictionary to map the number
        # to the description. Columns are indexed starting from 1.
        columns_description = {
            variable_name: int(column_number) - 1
            for column_number, variable_name, _ in (c.groups()
                                                    for c in columns)
        }

        return columns_description

    def _scan_header(self):
        """Use regular expressions to understand the content of a file.
        In particular, we look for column format and data columns.
        """
        # What method to we need to use to open the file?
        # opener can be open, gopen, or bopen depending on the extension
        # of the file
        opener, opener_mode = self._decompressor[self._compression_method]

        with opener(self.path, mode=opener_mode) as f:
            # Read the first 20 lines
            header = [f.readline() for i in range(self._header_line_number)]

            # Column format is relevant only for scalar output, so we start
            # from that
            if self.reduction_type == 'scalar':
                columns_description = \
                    self._scan_strings_for_columns(header,
                                                   self._rx_column_format)

                time_column = columns_description.get('time', None)

                if time_column is None:
                    raise RuntimeError(f"Missing time column in {self.path}")

                data_column = columns_description.get('data', None)

                if data_column is None:
                    raise RuntimeError(f"Missing data column in {self.path}")

                self._time_column = time_column
            else:
                # The reductions have always the same form:
                # 1:iteration 2:time 3:data
                self._time_column = 1
                data_column = 2

            # When we have one file per group we have many data columns
            # Se update _vars to add all the new ones
            if self._is_one_file_per_group:
                columns_description = \
                    self._scan_strings_for_columns(header,
                                                   self._rx_data_columns)

                self._vars.update(columns_description)
            else:
                # There is only one data_column
                self._vars = {list(self._vars.keys())[0]: data_column}

        self._was_header_scanned = True

    @lru_cache(128)
    def load(self, variable):
        """Read file and return a TimeSeries with the requested variable.

        :param variable: Requested variable
        :type variable: str

        :returns: TimeSeries with requested variable as read from file
        :rtype:        :py:class:`~.TimeSeries`

        """
        if (not self._was_header_scanned):
            self._scan_header()

        if (variable not in self):
            raise ValueError(f"{variable} not available")

        column_number = self._vars[variable]
        t, y = np.loadtxt(self.path, unpack=True, ndmin=2,
                          usecols=(self._time_column, column_number))

        return ts.remove_duplicate_iters(t, y)

    def __getitem__(self, key):
        return self.load(key)

    def __contains__(self, key):
        return key in self._vars

    def keys(self):
        """Return the list of variables available.

        :returns: List of variables in the file
        :rtype:   list

        """
        return list(self._vars.keys())


class AllScalars:
    """Helper class to read various types of scalar data in a list of files and
    properly order them. The core of this object is the _vars dictionary which
    contains the location of all the files for a specific variable and
    reduction.

    AllScalars is a dictionary-like object.

    Using the [] notation you can access values with as TimeSeries.

    Not intended for direct use.

    """

    def __init__(self, allfiles, reduction_type):
        """sd has to be a SimDir object, reduction_type has to be a reduction
        or scalar.

        allfiles is a list of files
        """
        self.reduction_type = str(reduction_type)
        # _vars is like _vars['variable']['folder'] -> CactusScalarASCII(f)
        # _vars['variable'] is a dictionary with as keys the folders where
        # to find the files associated to the variable and the reduction
        # reduction_type
        self._vars = {}
        for f in allfiles:
            # We only save those that variables are well-behaved
            try:
                cactusascii_file = OneScalar(f)
                if cactusascii_file.reduction_type == reduction_type:
                    for v in list(cactusascii_file.keys()):
                        # We add to the _vars dictionary the mapping:
                        # [v][folder] to CactusScalarASCII(f)

                        self._vars.setdefault(v, {})[cactusascii_file.folder] \
                            = cactusascii_file
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
        """Return the list of available variables corresponding to the given
        reduction.

        :returns: List of variables with given reduction
        :rtype:   list

        """
        return list(self._vars.keys())

    def get(self, key, default=None):
        """Return variable if available, else return the default value.

        :param key: Requested variable
        :type key: str
        :param default: Returned value if variable is not available
        :type default: any

        :returns: Timeseries of the requested variable
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
    simulation directory. Typically used from simdir instance. The different
    scalars are available as attributes:

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
        if (not isinstance(sd, simdir.SimDir)):
            raise TypeError("Input is not SimDir")

        self.path = sd.path
        self.point = AllScalars(sd.allfiles, 'scalar')
        self.scalar = AllScalars(sd.allfiles, 'scalar')
        self.minimum = AllScalars(sd.allfiles, 'minimum')
        self.maximum = AllScalars(sd.allfiles, 'maximum')
        self.norm1 = AllScalars(sd.allfiles, 'norm1')
        self.norm2 = AllScalars(sd.allfiles, 'norm2')
        self.average = AllScalars(sd.allfiles, 'average')
        self.infnorm = AllScalars(sd.allfiles, 'infnorm')

    def __getitem__(self, reduction):
        return getattr(self, reduction)

    def get(self, key, default=None):
        """Return a reduction if available, else return the default value.

        :param key: Requested reduction
        :type key: str
        :param default: Returned value if reduction is not available
        :type default: any

        :returns: Timeseries of the requested variable
        :rtype: :py:class:`~.AllScalars`

        """
        if key in ["point", "scalar", "minimum", "maximum", "norm1",
                   "norm2", "average", "infnorm"]:
            return self[key]

        return default

    def __str__(self):
        return "Folder %s\n%s\n%s\n%s\n%s\n%s\n%s\n"\
            % (self.path, self.scalar, self.minimum,
               self.maximum, self.norm1, self.norm2, self.average)
