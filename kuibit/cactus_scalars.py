#!/usr/bin/env python3

# Copyright (C) 2020-2025 Gabriele Bozzola
#
# Based on code originally developed by Wolfgang Kastaun. This file may contain
# algorithms and/or structures first implemented in
# GitHub:wokast/PyCactus/PostCactus/cactus_scalars.py
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
time series data as output by CarpetASCII or CarpetX, including all the
reductions.

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
import warnings
from bz2 import open as bopen
from functools import lru_cache
from gzip import open as gopen

import numpy as np

from kuibit import simdir
from kuibit import timeseries as ts
from kuibit.attr_dict import pythonize_name_dict
from kuibit.cactus_ascii_utils import (
    get_carpetx_variable_columns,
    scan_carpetx_header,
    scan_header,
)


class OneScalar:
    """Read scalar data produced by CarpetASCII or CarpetX.

    :py:class:`~.OneScalar` is a dictionary-like object with keys the variables
    and values the :py:class:`~.TimeSeries`.

    Single variable per file or single file per group are supported. In the
    latter case, the header is inspected to understand the content of the file.
    CarpetASCII files compressed with gzip or bzip2 are also supported.

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
    # we have eight capturing groups.
    # 1: (\w+) matches any number of characters greater than 0 (w = word)
    # 2: ((-(\w+))|(\[\d+\]))? optionally match one of the two
    # 3: Matches - with followed by 4: any word
    # 5: Matches brackets with a number inside
    # 6: (\.0)? optionally match .0 (see WARNING below)
    # In between match a dot (\.)
    # 7: (minimum|maximum|norm1|norm2|norm_inf|average|sum|scalars)? optionally match one
    #    of those
    # In between match .asc (\.asc)
    # 8: (\.(gz|bz2))? optionally match .gz or .bz2

    # We want to match file names like hydrobase-press.maximum.asc or
    # hydrobase-vel[0].maximum.asc
    #
    # The .scalars. file is the one generated with the option
    # all_reductions_in_one_file

    # For runs with Llama, there is an additional .N before the extension,
    # where N is the number of the patch (eg rho.0..asc).
    # WARNING: For now, only patch 0 files are detected. For patch systems
    # 'Thornburg04' and 'Thornburg13', it corresponds to the central cartesian
    # patch, giving the expected behavior. This may not apply to other patch systems.

    _pattern_filename = r"""
    ^(\w+)
    ((-(\w+))|(\[\d+\]))?
    (\.0)?
    \.(minimum|maximum|norm1|norm2|norm_inf|average|sum|scalars)?
    \.asc
    (\.(gz|bz2))?$"""

    _rx_filename = re.compile(_pattern_filename, re.VERBOSE)

    # CarpetX TSV files are named by thorn and group and can combine iterations
    # or use one file per iteration, e.g. hydrobasex-rho.tsv or
    # quasilocalmeasures-qlm_scalars.it000004.tsv. Norm files are always
    # norms/<thorn>-<group>.tsv.
    _pattern_filename_carpetx = r"^\w+-\w+(?P<iteration>\.it\d+)?\.tsv$"
    _rx_filename_carpetx = re.compile(_pattern_filename_carpetx)

    _reduction_types = {
        "minimum": "minimum",
        "maximum": "maximum",
        "norm1": "norm1",
        "norm2": "norm2",
        "norm_inf": "infnorm",
        "average": "average",
        "sum": "sum",
        "scalars": "scalars",
        None: "scalar",
    }

    # Map CarpetX norm header reduction names to kuibit reduction names.
    _reduction_types_carpetx = {
        "min": "minimum",
        "max": "maximum",
        "L1norm": "norm1",
        "L2norm": "norm2",
        "maxabs": "infnorm",
        "avg": "average",
        "sum": "sum",
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
        # The _vars_columns dictionary contains a mapping between the various variables
        # and the column numbers in which they are stored.
        self._vars_columns = {}

        # The _vars dictionary maps variables with the TimeSeries. It is used to
        # cached loaded data.
        self._vars = {}

        # Defaults overwritten by file-specific initialization.
        self._compression_method = None
        self.reduction_type = None
        self._all_reductions_in_one_file = False
        self._reduction_vars_columns = None
        self._is_one_file_per_group = False
        self._was_header_scanned = False
        self._time_column = None
        self._file_per_iteration = False

        self.folder, filename = os.path.split(self.path)

        filename_match = self._rx_filename.match(filename)
        carpetx_filename_match = self._rx_filename_carpetx.match(filename)
        if filename_match is not None:
            self._init_carpetascii_file(filename_match)
        elif carpetx_filename_match is not None:
            self._init_carpetx_tsv_file(carpetx_filename_match)
        else:
            raise RuntimeError(f"Name scheme not recognized for {filename}")

    def _init_carpetx_tsv_file(self, filename_match):
        """Initialize a CarpetX scalar or combined norm file."""

        self._file_format = "TSV"
        # Skip the header line, which starts with "#" or (plain) "iteration"
        self._comments = ("#", "iteration")
        self._is_one_file_per_group = True
        if os.path.basename(self.folder) == "norms":
            self.reduction_type = "scalars"
            self._all_reductions_in_one_file = True
        else:
            self.reduction_type = "scalar"
        self._file_per_iteration = (
            filename_match.group("iteration") is not None
        )
        try:
            self._scan_header()
        except RuntimeError as error:
            # The name looks like CarpetX output, so tell the user why the
            # file cannot be read
            warnings.warn(f"Skipping file: {error}", RuntimeWarning)
            raise

    def _init_carpetascii_file(self, filename_match):
        self._file_format = "ASCII"
        self._comments = "#"
        # variable_name1 may be a thorn name (e.g. hydrobase-press)
        # Or the actual variable name (e.g. H)
        (
            variable_name1,
            _0,
            _1,
            variable_name2,
            index_in_brackets,
            _multipatch,
            reduction_type,
            _2,
            compression_method,
        ) = filename_match.groups()

        self._compression_method = compression_method

        self.reduction_type = self._reduction_types[reduction_type]
        self._all_reductions_in_one_file = reduction_type == "scalars"

        # If the file contains multiple variables, we will scan the header
        # immediately to understand the content. If not, we scan the header
        # only when needed
        self._is_one_file_per_group = variable_name2 is not None

        if self._is_one_file_per_group:
            # We need the variable names
            self._scan_header()
        else:
            variable_name = variable_name1
            if index_in_brackets is not None:
                variable_name += index_in_brackets
            self._vars_columns = {variable_name: None}

    def _scan_header(self):
        # Normalize each format to variable -> column, or, for combined
        # reductions, reduction -> {variable: column}.

        if self._file_format == "ASCII":
            opener, opener_mode = self._decompressor[self._compression_method]
            self._time_column, columns_info = scan_header(
                self.path,
                self._is_one_file_per_group,
                extended_format=self.reduction_type == "scalar",
                all_reductions_in_one_file=self._all_reductions_in_one_file,
                opener=opener,
                opener_mode=opener_mode,
            )
            if self._all_reductions_in_one_file:
                columns_info = {
                    self._reduction_types[name]: columns
                    for name, columns in columns_info.items()
                    if name in self._reduction_types
                }
            elif not self._is_one_file_per_group:
                variable = next(iter(self._vars_columns))
                columns_info = {variable: columns_info}
        elif self._file_format == "TSV":
            self._time_column, columns_info, is_comment = scan_carpetx_header(
                self.path
            )
            if self._all_reductions_in_one_file:
                columns_info = self._get_carpetx_reduction_columns(
                    columns_info
                )
            else:
                columns_info = get_carpetx_variable_columns(
                    columns_info, {"iteration", "time"}, is_comment=is_comment
                )

        if self._all_reductions_in_one_file:
            if not columns_info:
                raise RuntimeError(
                    f"No supported reductions found in {self.path}"
                )
            self._reduction_vars_columns = columns_info
            columns_info = next(iter(columns_info.values()))
        self._vars_columns = columns_info

        self._was_header_scanned = True

    def _get_carpetx_reduction_columns(self, columns_description):
        """Group norm columns by kuibit reduction name and variable."""

        # Norm labels have the form hydrobasex::rho.min.
        rx_norm_column = re.compile(r"^\w+::(\w+)\.([\w]+(?:\[\d+\])?)$")
        reductions = {}
        for column_name, column_number in columns_description.items():
            if column_name in ("iteration", "time"):
                continue
            matched = rx_norm_column.match(column_name)
            if matched is None:
                raise RuntimeError(
                    f"Bad CarpetX norm header found in file {self.path}"
                )
            variable_name, reduction_name = matched.groups()
            reduction = self._reduction_types_carpetx.get(reduction_name)
            if reduction is not None:
                columns = reductions.setdefault(reduction, {})
                columns[variable_name] = column_number
        return reductions

    def _get_vars_columns_allreds(self, reduction_type):
        if not self._was_header_scanned:
            self._scan_header()

        columns = self._reduction_vars_columns.get(reduction_type)
        if columns is None:
            raise KeyError(f"{reduction_type} not available in {self.path}")

        self._vars_columns = columns
        self.reduction_type = reduction_type

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
            raise KeyError(f"{variable} not available")

        data_columns = self._vars_columns[variable]
        if not isinstance(data_columns, tuple):
            data_columns = (data_columns,)

        columns = np.loadtxt(
            self.path,
            comments=self._comments,
            unpack=True,
            ndmin=2,
            usecols=(self._time_column,) + data_columns,
        )
        t = columns[0]
        y = columns[1]
        if len(data_columns) == 2:
            y = y + 1j * columns[2]

        return ts.remove_duplicated_iters(t, y)

    def __getitem__(self, key):
        if key not in self._vars:
            # Error checking is done in __load__
            self._vars[key] = self.load(key)
        return self._vars[key]

    def __contains__(self, key):
        return key in self._vars_columns

    def keys(self):
        """Return the list of variables available.

        :returns: Variables in the file
        :rtype:   dict_keys

        """
        return self._vars_columns.keys()


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

        # _vars_readers is like _vars_readers['variable']['folder'][file_key]
        # -> OneScalar(f). Each folder is one run. CarpetX can write one file
        # per iteration, so these files are keyed by their path and combined
        # within the run. Other files have file_key None, as there is one file
        # per variable in each run.
        #
        # OneScalar objects act as readers.
        self._vars_readers = {}
        for file_ in allfiles:
            # We only save those that variables are well-behaved
            try:
                cactusascii_file = OneScalar(file_)
                if cactusascii_file.reduction_type != reduction_type:
                    if not cactusascii_file._all_reductions_in_one_file:
                        continue
                    try:
                        cactusascii_file._get_vars_columns_allreds(
                            reduction_type
                        )
                    except KeyError:
                        continue

                file_key = (
                    cactusascii_file.path
                    if cactusascii_file._file_per_iteration
                    else None
                )
                for var in list(cactusascii_file.keys()):
                    # We add to the _vars_readers dictionary the mapping:
                    # [var][folder][file_key] to OneScalar(f)
                    folder = cactusascii_file.folder
                    readers = self._vars_readers.setdefault(
                        var, {}
                    ).setdefault(folder, {})
                    if file_key in readers:
                        warnings.warn(
                            f"Overwriting {var} {reduction_type} from "
                            f"{readers[file_key].path} with "
                            f"{cactusascii_file.path}",
                            RuntimeWarning,
                        )
                    readers[file_key] = cactusascii_file
            except RuntimeError:
                pass

        # We cache the results in _vars, a dictionary with keys the variables
        # and values the timeseries.
        self._vars = {}

        # What pythonize_name_dict does is to make the various variables
        # accessible as attributes, e.g. self.fields.rho
        self.fields = pythonize_name_dict(list(self.keys()), self.__getitem__)

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(f"{key} not available")

        if key not in self._vars:
            # Combine files within each run, then combine the restarts.
            series = [
                ts.combine_ts([reader[key] for reader in run.values()])
                for run in self._vars_readers[key].values()
            ]
            self._vars[key] = ts.combine_ts(series)

        return self._vars[key]

    def __contains__(self, key):
        return key in self._vars_readers

    def keys(self):
        """Return the available variables corresponding to the given reduction.

        :returns: Variables with given reduction
        :rtype:   dict_keys

        """
        return self._vars_readers.keys()

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
    :ivar sum:       access to sum reduction.
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
        self.scalar = AllScalars(sd.allfiles, "scalar")
        self.minimum = AllScalars(sd.allfiles, "minimum")
        self.maximum = AllScalars(sd.allfiles, "maximum")
        self.norm1 = AllScalars(sd.allfiles, "norm1")
        self.norm2 = AllScalars(sd.allfiles, "norm2")
        self.average = AllScalars(sd.allfiles, "average")
        self.sum = AllScalars(sd.allfiles, "sum")
        self.infnorm = AllScalars(sd.allfiles, "infnorm")

        # Aliases
        self.max = self.maximum
        self.min = self.minimum
        self.point = self.scalar

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
            "sum",
            "infnorm",
        ]:
            return self[key]

        return default

    def __str__(self):
        str_ = [f"Folder {self.path}"]
        for key in (
            self.path,
            self.scalar,
            self.minimum,
            self.maximum,
            self.norm1,
            self.norm2,
            self.average,
            self.sum,
        ):
            str_.append(str(key))
        return "\n".join(str_)
