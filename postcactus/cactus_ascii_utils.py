#!/usr/bin/env python3

# Copyright (C) 2020 Gabriele Bozzola, Wolfgang Kastaun
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

""" This module provides helper functions to read Cactus ASCII files.
"""

import os
import re


def _scan_strings_for_columns(strings, pattern, path=None):
    """Match each string in strings against pattern and each matching result
    against _pattern_columns, which match expressions like 3:kxx. Then, return
    a dictionary that maps variable to column number.

    path here is given only for the error message

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
        raise RuntimeError("Bad header found")

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
    file_has_column_format=True,
    opener=open,
    opener_mode="r",
):
    """Use regular expressions to understand the content of a file.
    In particular, we look for column format and data columns.

    This function is also used by cactus_grid_function.

    Some files, like the scalars ouput by CarpetASCII, have an additional row
    "column format" that describes the various columns. If that is available, we
    should scan it. However, some files do not have that (e.g., the reductions).

    :returns: time_column and either the data column (if it is one variable per
    group, or a dictionary with column: variable)

    :rtype: int, another int or a dictionary

    """

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

        if file_has_column_format:
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


def total_filesize(allfiles, unit="MB"):
    """Return the total size of the given files.
    Available units B, KB, MB and GB

    :param allfiles: list of the full paths of the files
    :type allfiles: list

    :returns: Total size of the given files.
    :rtype: float

    """

    # This function is here, but it could be anywhere, it doesn't really
    # apply only to ASCII files, nor only to CACTUS files...
    units = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
    if unit not in units.keys():
        raise ValueError(f"Invalid unit: expected one of {list(units.keys())}")
    return sum(os.path.getsize(path) for path in set(allfiles)) / units[unit]
