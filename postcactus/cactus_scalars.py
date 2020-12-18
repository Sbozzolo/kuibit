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
from postcactus.cactus_ascii_utils import scan_header


class OneScalar:
    """Read scalar data produced by CarpetASCII.

    CactusScalarASCII is a dictionary-like object: it has keys() and you can
    TimeSeries using the ['key'] syntax.

    Single variable per file or single file per group are supported. In the
    latter case, the header is inspected to understand the content of the file.

    Compressed files (gz and bz2) are supported.

    OneScalar represents one scalar file, there can be multiple variables inside,
    (if it was one_file_per_group).

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

        file_has_column_format = self.reduction_type == "scalar"

        # What method to we need to use to open the file?
        # opener can be open, gopen, or bopen depending on the extension
        # of the file
        opener, opener_mode = self._decompressor[self._compression_method]

        self._time_column, columns_info = scan_header(
            self.path,
            self._is_one_file_per_group,
            file_has_column_format,
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

        :param variable: Requested variable
        :type variable: str

        :returns: TimeSeries with requested variable as read from file
        :rtype:        :py:class:`~.TimeSeries`

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

class TwoScalar:
    """Read scalar data produced by the IL code BHNS diagnostics suite.

    TwoScalar is a dictionary-like object: it has keys() and you can
    TimeSeries using the ['key'] syntax.

    Single variable per file or single file per group are supported. In the
    latter case, the header is inspected to understand the content of the file.

    Compressed files (gz and bz2) are supported.

    TwoScalar represents one scalar file, there can be multiple variables inside,
    (if it was one_file_per_group).

    """

    # What is this pattern?
    # See the comment in OneScalar for a general explanation.
    # In this case, I have changed things so that it recognizes
    # the set of files outputted by the BHNS diagnostics instead.
    
    _pattern_filename = r"""^bhns(?:\.(don|mon|xon)|-(dens_mode\.con|emf\.con|gam\.con|ham\.con|max_tor_pol\.mon|mom\.con)|(?:\.(jon)\.([0-9])))(?:\.(gz|bz2))?$"""

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
            file_name_segment1,
            file_name_segment2,
            file_name_segment3,
            extraction_radius_number,
            compression_method,
        ) = filename_match.groups()

        self._compression_method = compression_method

        self.reduction_type = self._reduction_types[None] # All bhns files are of the 'scalar' reduction type
        self._time_column = 0 # All bhns files have colmun 0 hold the times

        if (file_name_segment1 is not None):
            if (file_name_segment1=="don"):
                self._vars = {"M0":1, "M0 inside AH":2, "M0_esc(r>30M)":3, "M0_esc(r>50M)":4,
                              "M0_esc(r>70M)":5, "M0_esc(r>100M)":6, "Eint":7, "T":8, "M_ADM_Vol":9,
                              "M_Komar Surf":10, "M0dot_BH":11, "int_M0dot":12, "Tfluid0_0_VolInt":13}
            elif (file_name_segment1=="mon"):
                self._vars = {"Phicent":1, "Kcent":2, "lapsecent":3, "psicent":4, "lapsemin":5,
                              "phimax":6, "rhocent":7, "rho_b_max":8, "Jsurf_BH_1.1":9, "J_1.1exVol":10,
                              "M_ADMsurf_BH_1.1":11, "M_ADM1.1exVol":12, "rhostar_min":13,
                              "J_1.1exVolr":14, "J_1.1exVolr/2":15, "rho_b_max_x":16, "rho_b_max_y":17,
                              "rho_b_max_z":18}
            elif (file_name_segment1=="xon"):
                self._vars = {"Box1x":1, "Box1y":2, "Box2x":3, "Box2y":4, "M0_star_1":5, "M0_star_2":6}
            else:
                raise RuntimeError("BHNSDiagnosticASCII: naming scheme not recognized for %s" % fn)
        elif (file_name_segment2 is not None):
            if (file_name_segment2=="dens_mode.con"):
                # Ultimately there should be a special structure for the variables from
                # this diagnostic file, but for now this will have to do.
                self._vars = {"m=0":1, "m=1 real":2, "m=1 im":3, "m=2 real":4, "m=2 im":5, "m=3 real":6,
                              "m=3 im":7, "m=4 real":8, "m=4 im":9, "m=5 real":10, "m=5 im":11,
                              "m=6 real":12, "m=6 im":13}
            elif (file_name_segment2=="emf.con"):
                self._vars = {"Eem":1, "Eem_outsideAH":2, "int B_phi":3, "Bx_max":4, "By_max":5,
                              "Monopole":6, "Monop.t=0":7, "Mpole_outAH":8, "b2_max":9, "b2_max_x":10,
                              "b2_max_y":11, "b2_max_z":12, "Eemoutside_r1":13, "Eemoutside_r2":14,
                              "b2_maxv2":15}
            elif (file_name_segment2=="gam.con"):
                 self._vars = {"Gamx Res":1, "Gamy Res":2, "Gamz Res":3}
            elif (file_name_segment2=="ham.con"):
                 self._vars = {"Haml ResN":1, "Haml ResD":2, "Ham Res2P1.1N":3, "Ham Res2P1.1D":4,
                               "HR1.1-NoD":5, "HRes_outAH_N":6, "HRes_outAH_D":7, "HRes_outAHNoD":8,
                               "HRes2P1.1NNR":9, "HRes2P1.1DNR":10}
            elif (file_name_segment2=="max_tor_pol.mon"):
                self._vars = {"B_tor_rhostarN_3":1, "B_pol_rhostarN_3":2, "B^2/(2P)rhostarN_3":3,
                              "rhostar_D_3":4, "B_tor_max":5, "B_pol_max":6}
            elif (file_name_segment2=="mom.con"):
                self._vars = {"Momx ResN":1, "Momy ResN":2, "Momz ResN":3, "Mom ResD":4, "Momx Res1.1N":5, "Momy Res1.1N":6, "Momz Res1.1N":7, "Mom Res1.1D":8, "MRx 1.1NoD":9}
            else:
                raise RuntimeError("Naming scheme not recognized for %s" % filename)
        elif ((file_name_segment3=="jon") and (extraction_radius_number is not None)):
            # Ultimately there should probably be a special structure for the
            # variables from this set of diagnostic files that treats them as lists
            # of values at different extraction radii.
            # For now the extraction_radius_number will just be appended to the variable name string.
            tmp = {"fisheye radius":1, "phys radius":2, "Mass_sur":3, "Ang_mom_surf":4,
                   "Komar Mass":5, "P_x2":6, "P_y2":7, "P_z2":8, "F_M0":9, "F_E_fluid":10,
                   "F_E_em":11, "F_J_fluid":12, "F_J_em":13}
            for key, val in tmp.items():
                self._vars[key+"_"+str(extraction_radius_number)] = val
        else:
            raise RuntimeError("Naming scheme not recognized for %s" % filename)
    
    @lru_cache(128)
    def load(self, variable):
        """Read file and return a TimeSeries with the requested variable.

        :param variable: Requested variable
        :type variable: str

        :returns: TimeSeries with requested variable as read from file
        :rtype:        :py:class:`~.TimeSeries`

        """
        if variable not in self:
            raise ValueError(f"{variable} not available")

        column_number = self._vars[variable]
        t, y = np.loadtxt(
            self.path,
            unpack=True,
            ndmin=2,
            usecols=(self._time_column, column_number),
        )

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
        """allfiles is a list of files, reduction_type has to be a reduction or
        scalar.

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
                try:
                    cactusascii_file = TwoScalar(file_)
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

        :param key: Requested reduction
        :type key: str
        :param default: Returned value if reduction is not available
        :type default: any

        :returns: Timeseries of the requested variable
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
