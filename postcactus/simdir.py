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

""" This module provides easy access to CACTUS data files.

A simulation directory is represented by an instance of the
:py:class:`~.SimDir` class, which provides access to all supported
data types.
"""

import os

# We ideally would like to use cached_property, but it is in Python 3.8
# which is quite new
from functools import lru_cache

from postcactus import (
    cactus_grid_functions,
    cactus_horizons,
    cactus_multipoles,
    cactus_scalars,
    cactus_waves,
)


class SimDir:
    """This class represents a CACTUS simulation directory.

    Data is searched recursively in all subfolders. No particular folder
    structure (e.g. SimFactory style) is assumed. The following attributes
    allow access to the supported data types:

    :ivar path:           Top level path of simulation directory.
    :ivar dirs:           All directories in which data is searched.
    :ivar logfiles:       The locations of all log files (.out).
    :ivar errfiles:       The location of all error log files (.err).
    :ivar ts:             Scalar data of various type, see
                          :py:class:`~.ScalarsDir`
    :ivar gf:              Access to grid function data, see
                          :py:class:`~.GridFunctionsDir`.
    :ivar gws:            GW signal from the Weyl scalar multipole
                          decomposition, see
                          :py:class:`~.GravitationalWavesDir`.
    :ivar emws:           EM signal from the Weyl scalar multipole
                          decomposition, see
                          :py:class:`~.ElectromagneticWavesDir`.
    :ivar horizons:       Apparent horizon information, see
                          :py:class:`~.HorizonsDir`.
    :ivar multipoles:     Multipole components, see
                          :py:class:`~.CactusMultipoleDir`.
    """

    def _sanitize_path(self, path):
        # Make sure to have complete paths with respect to the current folder
        self.path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(self.path):
            raise RuntimeError(f"Folder does not exist: {path}")

    def _scan_folders(self, max_depth):
        """Scan all the folders in self.path up to depth max_depth
        and categorize all the files.
        """

        self.dirs = []
        self.parfiles = []
        self.logfiles = []
        self.errfiles = []
        self.allfiles = []

        def listdir_no_symlinks(path):
            """Return a list of files in path that are not symlink"""
            dir_content = [os.path.join(path, p) for p in os.listdir(path)]
            return [p for p in dir_content if not os.path.islink(p)]

        def filter_ext(files, ext):
            """Return a list from the input list of file that
            has file extension ext."""
            return [f for f in files if os.path.splitext(f)[1] == ext]

        def walk_rec(path, level=0):
            """Walk_rec is a recursive function that steps down all the
            subdirectories (except the ones with name defined in self.ignore)
            up to max_depth and add to self.allfiles the files found in the
            directories.

            """
            if level >= max_depth:
                return

            self.dirs.append(path)

            all_files_in_path = listdir_no_symlinks(path)

            files_in_path = list(filter(os.path.isfile, all_files_in_path))
            self.allfiles += files_in_path

            directories_in_path = list(
                filter(os.path.isdir, all_files_in_path)
            )

            # We ignore the ones in self.ignore
            directories_to_scan = [
                p
                for p in directories_in_path
                if (os.path.basename(p) not in self.ignore)
            ]

            # Apply walk_rec to all the subdirectory, but with level increased
            for p in directories_to_scan:
                walk_rec(p, level + 1)

        walk_rec(self.path)

        self.logfiles = filter_ext(self.allfiles, ".out")
        self.errfiles = filter_ext(self.allfiles, ".err")
        self.parfiles = filter_ext(self.allfiles, ".par")

        # Sort by time
        self.parfiles.sort(key=os.path.getmtime)
        self.logfiles.sort(key=os.path.getmtime)
        self.errfiles.sort(key=os.path.getmtime)

        simfac = os.path.join(self.path, "SIMFACTORY", "par")

        # Simfactory has a folder SIMFATORY with a subdirectory for par files
        # Even if SIMFACTORY is excluded, we should include that par file
        if os.path.isdir(simfac):
            mainpar = filter_ext(listdir_no_symlinks(simfac), ".par")
            self.parfiles = mainpar + self.parfiles

        self.has_parfile = bool(self.parfiles)

        # TODO: Add this when cactus_parfile is ready

        # if self.has_parfile:
        #     self.initial_params = cpar.load_parfile(self.parfiles[0])
        # else:
        #     self.initial_params = cpar.Parfile()

    def __init__(self, path, max_depth=8, ignore=None):
        """Constructor.

        :param path:      Path to simulation directory.
        :type path:       string
        :param max_depth: Maximum recursion depth for subfolders.
        :type max_depth:  int
        :param ignore: Folders to ignore
        :type ignore:  set

        Parfiles (``*.par``) will be searched in all data directories and the
        top-level SIMFACTORY/par folder, if it exists. The parfile in the
        latter folder, if available, or else the oldest parfile in any of
        the data directories, will be used to extract the simulation
        parameters. Logfiles (``*.out``) and errorfiles (``*.err``) will be
        searched for in all data directories.
        """
        if ignore is None:
            ignore = {"SIMFACTORY", "report", "movies", "tmp", "temp"}

        self.ignore = ignore
        self._sanitize_path(str(path))
        self._scan_folders(int(max_depth))

    @property
    # We only need to keep it 1 in memory: it is the only possible!
    @lru_cache(1)
    def ts(self):
        return cactus_scalars.ScalarsDir(self)

    timeseries = ts

    @property
    @lru_cache(1)
    def multipoles(self):
        return cactus_multipoles.MultipolesDir(self)

    @property
    @lru_cache(1)
    def gravitationalwaves(self):
        return cactus_waves.GravitationalWavesDir(self)

    gws = gravitationalwaves

    @property
    @lru_cache(1)
    def electromagneticwaves(self):
        return cactus_waves.ElectromagneticWavesDir(self)

    emws = electromagneticwaves

    @property
    @lru_cache(1)
    def gridfunctions(self):
        return cactus_grid_functions.GridFunctionsDir(self)

    gf = gridfunctions

    @property
    @lru_cache(1)
    def horizons(self):
        return cactus_horizons.HorizonsDir(self)

    def __str__(self):
        header = f"Indexed {len(self.allfiles)} files"
        header += f" and {len(self.dirs)} subdirectories\n"

        ts_ret = f"{self.ts}"
        mp_ret = f"{self.multipoles}"
        gf_ret = f"{self.gf}"

        if len(self.gravitationalwaves) > 0:
            gw_ret = "Available gravitational wave data"
        else:
            gw_ret = ""

        if len(self.electromagneticwaves) > 0:
            em_ret = "Available electromagnetic wave data"
        else:
            em_ret = ""

        hor_ret = f"{self.horizons}"

        return header + ts_ret + mp_ret + gw_ret + em_ret + gf_ret + hor_ret
