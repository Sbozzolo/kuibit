#!/usr/bin/env python3
"""The :py:mod:`~.cactus_scalars` module provides functions to load
timeseries in Cactus formats and a class :py:class:`ScalarsDir` for easy
access to all timeseries in a Cactus simulation directory. This module
is normally not used directly, but from the :py:mod:`~.simdir` module.
The data loaded by this module is represented as
:py:class:`~.TimeSeries` objects.
"""

from bz2 import BZ2File as bopen
from gzip import open as gopen
import os
import re


class CactusScalarASCII:
    """Read scalar data produced by CarpetASCII.

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

    _pattern_datacolumns = r"^# data columns: (.+)$"
    _pattern_columnformat = r"^# column format: (.+)$"
    # Here we match (number):(word[number])
    _pattern_columns = r"^(\d+):(\w+(\[\d+\])?)$"

    _rx_filename = re.compile(_pattern_filename, re.VERBOSE)
    _rx_datacolumns = re.compile(_pattern_datacolumns)
    _rx_columnformat = re.compile(_pattern_columnformat)
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

    # What function to use to open the file?
    _decompr = {None: open, 'gz': gopen, 'bz2': bopen}

    def __init__(self, path):
        self.path = str(path)
        self._vars = {}
        self.folder, filename = os.path.split(self.path)

        filename_match = self._rx_filename.match(filename)

        if filename_match is None:
            raise RuntimeError(f"Name scheme not recognized for {filename}")

        # variable_name1 may be a thorn name (e.g. hydrobase-press)
        # Or the actual variable name (e.g. H)
        (variable_name1, _0, _1, variable_name2, index_in_brackets,
         reduction_type, _2, compression) = filename_match.groups()

        self._compression = compression

        self.reduction_type = reduction_type if reduction_type is not None \
            else "scalar"

        # If the file contains multiple variables, we will scan the header
        # immediately to understand the content. If not, we scan the header
        # only when needed
        self._is_one_file_per_group = (variable_name2 is not None)
        self._was_header_scanned = False

        if self._is_one_file_per_group:
            pass
            # self._scan_column_header()
        else:
            self._time_column = None
            variable_name = variable_name1
            if (index_in_brackets is not None):
                variable_name += index_in_brackets
            self._vars = {variable_name: None}
