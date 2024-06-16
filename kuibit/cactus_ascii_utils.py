#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
#
# Inspired by code originally developed by Wolfgang Kastaun. This file may
# contain algorithms and/or structures first implemented in
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

"""This module provides helper functions to extract information from Cactus
ASCII files.

The functions available are:

- :py:func:`~.scan_header`: Takes the path of a Cactus ASCII file, a bool to
                            indicate if the file contains one or multiple
                            variables (if Carpet was set with
                            ``one_file_per_group = yes``), a bool to indicate if
                            the file has several columns and a line that
                            describe them. ``scan_header`` then returns the
                            number of the column with the time, and a dictionary
                            (or a single number) with the description of the
                            content of the other columns.

- :py:func:`~.total_filesize`: Takes a list of files are return the total
                               filesize with a given unit. This also works for
                               non-ASCII files.

"""

import os
import re
from collections.abc import Iterable


def _scan_strings_for_columns(strings, pattern, path=None):
    """Match each string in strings against pattern and each matching result
    against _pattern_columns, which matches expressions like "3:kxx". Then,
    return a dictionary that maps variable to column number.

    This is specialized function used by :py:meth:`~.scan_header` to go through
    headers of CarpetASCII files.

    :param strings: List of strings to match against the given pattern.
    :type strings: list of str
    :param pattern: Pattern to match against strings.
    :type pattern: ``re.Pattern``
    :param path: Path of the file, used only for producing useful error messages.
    :type path: str

    :returns: Dictionary with the mapping between columns numbers and their
              variable.
    :rtype: dict

    """

    # Here we match (number):(word[number])
    # We are matching expressions like 3:kxx
    pattern_columns = r"^(\d+):(\w+(\[\d+\])?)$"
    rx_columns = re.compile(pattern_columns)

    # We scan these lines and see if any matches with the regexp for the
    # column format
    for line in strings:
        matched_pattern = pattern.match(line)
        if matched_pattern is not None:
            break
    # Here we are using an else clause with a for loop. This else
    # branch is reached only if the break in the for loop is never
    # called. In this case, this happens if we never match the
    # given pattern
    else:
        raise RuntimeError(f"Unrecognized header in file {path}")

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

    columns = list(
        map(
            rx_columns.match,
            matched_pattern.groups()[0].split(),
        )
    )

    are_real_columns = all(columns)

    if not are_real_columns:
        raise RuntimeError(f"Bad header found in file {path}")

    # Columns are good. Let's create a dictionary to map the number
    # to the description. Columns are indexed starting from 1.
    columns_description = {
        variable_name: int(column_number) - 1
        for column_number, variable_name, _ in (c.groups() for c in columns)
    }

    return columns_description


def scan_header(
    path,
    one_file_per_group,
    extended_format=True,
    opener=open,
    opener_mode="r",
):
    """Use regular expressions to understand the content of a CarpetASCII file.
    In particular, we look for column format and data columns by reading the
    header, as defined as the lines that start with ``#``.

    This function is used by :py:mod:`~.cactus_grid_functions` and
    :py:mod:`~.cactus_scalars`.

    Some files, like the scalars output by CarpetASCII, have an additional row
    "column format" that describes the various columns. If that is available, we
    scan it. However, some files do not have that (e.g., the reductions).

    :param path: Path of the file to be scanned.
    :type path: str
    :param one_file_per_group: Was this file generated with the option
                               ``one_file_per_group``? This can be understood by
                               looking at the filename.
    :type one_file_per_group: bool
    :param extended_format: Does this file have many columns, and a line that
                            explains all the columns?
    :type extended_format: bool

    :param opener: Function that has to be used to open the file. The default is
                   ``open``, but it has to be different if the file is compressed.
    :type opener: callable
    :param opener_mode: Mode to open the file with (e.g., ``r`` as in ``read``).
    :type opener_mode: str

    :returns: time_column and either the data column (if it is one variable per
              group), or a dictionary with column: variable.
    :rtype: tuple with int, another int or a dictionary.

    """

    # TODO (REFACTORING): This function really wants to be refactored!
    #
    # This function is a little bit convoluted for what it does, and it has four
    # possible branching. It should be possible to simplify the function and
    # deal with only one case. Instead of returning two possible return values,
    # we should be able to return only one.

    # We are going to match some regular expressions in the header to
    # understand what variables are there

    # Example of column format:
    # column format: 1:it 2:tl 3:rl 4:c 5:ml 6:ix 7:iy 8:iz 9:time 10:x 11:y
    # 12:z 13:data
    pattern_column_format = r"^# column format: (.+)$"
    rx_column_format = re.compile(pattern_column_format)

    # Example of data column:
    # 1 2 3 are always: 1:iteration 2:time 3:data
    # data columns: 3:kxx 4:kxy 5:kxz 6:kyy 7:kyz 8:kzz
    pattern_data_columns = r"^# data columns: (.+)$"
    rx_data_columns = re.compile(pattern_data_columns)

    with opener(path, mode=opener_mode) as fil:
        header = []
        for line in fil:
            # We read the header, which starts with #. The first line
            # that doesn't start with # we assume is the first line of
            # actual data
            if line.startswith("#"):
                header.append(line)
            else:
                break

        if extended_format:
            columns_description = _scan_strings_for_columns(
                header, rx_column_format, path=path
            )

            time_column = columns_description.get("time", None)

            if time_column is None:
                raise RuntimeError(f"Missing time column in {path}")

            data_column = columns_description.get("data", None)

            if data_column is None:
                raise RuntimeError(f"Missing data column in {path}")
        else:
            # The reductions have always the same form:
            # 1:iteration 2:time 3:data
            time_column = 1
            # There is only one data_column
            data_column = 2

    # When we have one file per group we have many data columns
    # Se update _vars to add all the new ones
    if one_file_per_group:
        columns_description = _scan_strings_for_columns(
            header, rx_data_columns, path=path
        )
        return time_column, columns_description
    # This is not one file per group
    return time_column, data_column


def total_filesize(allfiles: Iterable, unit="MB") -> float:
    """Return the total size of the given files.

    Available units B, KB, MB and GB.

    :param allfiles: List of the full paths of the files.
    :type allfiles: list
    :param unit: Unit to use (in powers of 1024 bytes).
    :type unit: str among: ``B``, ``KB``, ``MB``, ``GB``.
    :returns: Total size of the given files.
    :rtype: float

    """
    directories = [path for path in allfiles if not os.path.isfile(path)]

    if len(directories) > 0:
        raise ValueError(f"Given list contains directories: {directories}")

    # This function is here, but it could be anywhere, it doesn't really
    # apply only to ASCII files, nor only to Cactus files...
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    if unit not in units:
        raise ValueError(f"Invalid unit: expected one of {list(units.keys())}")
    return sum(os.path.getsize(path) for path in set(allfiles)) / units[unit]
