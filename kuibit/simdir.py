#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
#
# Based on code originally developed by Wolfgang Kastaun. See, GitHub,
# wokast/PyCactus/PostCactus/simdir.py
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

"""This module provides easy access to Cactus data files.

A simulation directory is represented by an instance of the :py:class:`~.SimDir`
class, which provides access to all supported data types.

This is the main entry point into ``kuibit``. When a :py:class:`~.SimDir` is
initialized, the simulation directory is scanned and all the data is organized.
:py:class:`~.SimDir` objects have attributes that are interfaces to the data:
each attribute is a dictionary-like object that indexes the relevant data in
some way. For example, :py:meth:`~.timeseries` contains all the time series in
the output, indexed by the type of reduction that produced them (for example,
``norm2``, ``max``, ...).

In case of uncertainty, it is always possible to print :py:class:`~.SimDir`,
or any of its attributes, to obtain a message with the available content of
such attribute.

"""

import os

# TODO (FUTURE): cached_property is the decorator we are looking for.
#
# We ideally would like to use cached_property, but it is in Python 3.8,
# and currently we only support 3.6.
from functools import lru_cache

from kuibit import (
    cactus_grid_functions,
    cactus_horizons,
    cactus_multipoles,
    cactus_scalars,
    cactus_waves,
)


class SimDir:
    """This class represents a Cactus simulation directory.

    Data is searched recursively in all subfolders. No particular folder
    structure (e.g. ``simfactory`` style) is assumed. The following attributes
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
        """Scan all the folders in self.path up to depth ``max_depth``
        and categorize all the files.

        :param max_depth: Maximum recursion depth to scan.
        :type max_depth: int
        """

        self.dirs = []
        self.parfiles = []
        self.logfiles = []
        self.errfiles = []
        self.allfiles = []

        def listdir_process_symlinks(path):
            """Return a list of files in path. If self.ignore_symlinks, exclude the
            symlinks, otherwise keep them around.

            """
            dir_content = [os.path.join(path, p) for p in os.listdir(path)]
            if not self.ignore_symlinks:
                return dir_content
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

            all_files_in_path = listdir_process_symlinks(path)

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

        # Simfactory has a folder SIMFACTORY with a subdirectory for par files
        # Even if SIMFACTORY is excluded, we should include that par file
        if os.path.isdir(simfac):
            mainpar = filter_ext(listdir_process_symlinks(simfac), ".par")
            self.parfiles = mainpar + self.parfiles

        self.has_parfile = bool(self.parfiles)

    def __init__(self, path, max_depth=8, ignore=None, ignore_symlinks=True):
        """Constructor.

        :param path:      Path to output of the simulation.
        :type path:       str
        :param max_depth: Maximum recursion depth for subfolders.
        :type max_depth:  int
        :param ignore: Names of folders to ignore (e.g. SIMFACTORY).
        :type ignore:  set
        :param ignore_symlink: If True, do not consider symlinks.
        :type ignore_symlink: bool

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
        self.ignore_symlinks = ignore_symlinks
        self._sanitize_path(str(path))
        self._scan_folders(int(max_depth))

    @property
    # We only need to keep it 1 in memory: it is the only possible!
    @lru_cache(1)
    def ts(self):
        """Return all the available timeseries in the data.

        :returns: Interface to all the timeseries in the directory.
        :rtype: :py:class:`~.ScalarsDir`
        """
        return cactus_scalars.ScalarsDir(self)

    timeseries = ts

    @property
    @lru_cache(1)
    def multipoles(self):
        """Return all the available multipole data.

        :returns: Interface to all the multipole data in the directory.
        :rtype: :py:class:`~.MultipolesDir`
        """
        return cactus_multipoles.MultipolesDir(self)

    @property
    @lru_cache(1)
    def gravitationalwaves(self):
        """Return all the available ``Psi4`` data.

        :returns: Interface to all the ``Psi4`` data in the directory.
        :rtype: :py:class:`~.GravitationalWavesDir`
        """
        return cactus_waves.GravitationalWavesDir(self)

    gws = gravitationalwaves

    @property
    @lru_cache(1)
    def electromagneticwaves(self):
        """Return all the available ``Phi2`` data.

        :returns: Interface to all the ``Phi2`` data in the directory.
        :rtype: :py:class:`~.ElectromagneticWavesDir`
        """
        return cactus_waves.ElectromagneticWavesDir(self)

    emws = electromagneticwaves

    @property
    @lru_cache(1)
    def gridfunctions(self):
        """Return all the available grid data.

        :returns: Interface to all the grid data in the directory.
        :rtype: :py:class:`~.GridFunctionsDir`
        """
        return cactus_grid_functions.GridFunctionsDir(self)

    gf = gridfunctions

    @property
    @lru_cache(1)
    def horizons(self):
        """Return all the available horizon data.

        :returns: Interface to all the horizon data in the directory.
        :rtype: :py:class:`~.HorizonsDir`
        """
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
