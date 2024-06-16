#!/usr/bin/env python3

# Copyright (C) 2020-2024 Gabriele Bozzola
#
# Based on code originally developed by Wolfgang Kastaun. This file may contain
# algorithms and/or structures first implemented in
# GitHub:wokast/PyCactus/PostCactus/simdir.py
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

The function :py:func:`~.load_SimDir` can be used to load a :py:class:`~.SimDir`
saved with the method :py:meth:`~.save`.

"""
import os
import pickle

from kuibit import (
    cactus_grid_functions,
    cactus_horizons,
    cactus_multipoles,
    cactus_scalars,
    cactus_timers,
    cactus_twopunctures,
    cactus_waves,
)


def load_SimDir(path):
    """Load file produced with :py:meth:`~.SimDir.save`.

    Pickles have to be regenerated if the version of ``kuibit`` changes.

    :param path: Pickle file as produced by :py:meth:`~.SimDir.save`.
    :type path: str

    :returns: SimDir
    :rtype: :py:class:`~.SimDir`

    """
    with open(path, "rb") as file_:
        sim = pickle.load(file_)

    if not isinstance(sim, SimDir):
        raise RuntimeError(f"File {path} does not contain a SimDir")

    return sim


class SimDir:
    """This class represents a Cactus simulation directory.

    :py:class:`~.SimDir` can be used as a context manager. For instance:

    .. code-block

       with SimDir(sim_path) as sim:
            print(sim)

    By itself, this is not very useful. It becomes more useful in conjunction
    with using pickles. Pickles are used to save the work done and resume it
    later. Since ``kuibit`` does a lot of lazy-loading, it can be useful to save
    the operations performed to disk and restart from them. For example, when
    initializing a :py:class:`~.SimDir`, the files have to be scanned and
    organized. It is pointless to this all the times if the simulation has not
    changed. For this, we use pickles.

    .. code-block

       with SimDir(sim_path, pickle_file="sim.pickle") as sim:
            print(sim)

    What happens here is that if ``pickle_file`` exists, it will be loaded. If
    it does not exist, it will be created. Using context managers here is useful
    because it automatically saves all the progress done. Alternatively, one
    has to call the :py:meth:`~.save` method manually.

    .. warning::

       When using pickles, it is important to make sure that the underlying data
       does not change, as the new/changed data will be not considered. To
       refresh :py:class:`~.SimDir`, you can always use the :py:meth:`~.rescan`
       method. Pickles have to be regenerated from scratch if the version of
       ``kuibit`` changes.

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
    :ivar timers:         Timer information, see
                          :py:class:`~.TimersDir`.
    :ivar twopunctures:   Metadata information from TwoPunctures.
                          :py:class:`~.TwoPuncturesDir`.
    :ivar multipoles:     Multipole components, see
                          :py:class:`~.CactusMultipoleDir`.

    """

    @staticmethod
    def _sanitize_path(path):
        # Make sure to have complete paths with respect to the current folder
        abs_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(abs_path):
            raise RuntimeError(f"Folder does not exist: {path}")

        return abs_path

    def _scan_folders(self, max_depth):
        """Scan all the folders in self.path up to depth ``max_depth``
        and categorize all the files.

        :param max_depth: Maximum recursion depth to scan.
        :type max_depth: int
        """

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
            subdirectories (except the ones with name defined in
            self.ignored_dirs) up to max_depth and add to self.allfiles the
            files found in the directories.

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

            # We ignore the ones in self.ignored_dirs
            directories_to_scan = [
                p
                for p in directories_in_path
                if (os.path.basename(p) not in self.ignored_dirs)
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

    def __init__(
        self,
        path,
        max_depth=8,
        ignored_dirs=None,
        ignore_symlinks=True,
        pickle_file=None,
    ):
        """Constructor.

        :param path:      Path to output of the simulation.
        :type path:       str
        :param max_depth: Maximum recursion depth for subfolders.
        :type max_depth:  int
        :param ignored_dirs: Names of folders to ignore (e.g. SIMFACTORY).
        :type ignored_dirs:  set
        :param ignore_symlink: If True, do not consider symlinks.
        :type ignore_symlink: bool
        :param pickle_file: If ``pickle_file`` is not None, do not scan the
                            folders and load the pickle file. All the other
                            parameters are ignored.
        :type pickle_file: bool

        Parfiles (``*.par``) will be searched in all data directories and the
        top-level SIMFACTORY/par folder, if it exists. The parfile in the latter
        folder, if available, or else the oldest parfile in any of the data
        directories, will be used to extract the simulation parameters. Logfiles
        (``*.out``) and errorfiles (``*.err``) will be searched for in all data
        directories.

        """
        if ignored_dirs is None:
            ignored_dirs = {"SIMFACTORY", "report", "movies", "tmp", "temp"}

        # We update self.path in _sanitize_path to make sure it is an absolute
        # path
        self.path = self._sanitize_path(str(path))

        self.max_depth = int(max_depth)
        self.ignored_dirs = ignored_dirs
        self.ignore_symlinks = ignore_symlinks

        self.dirs = []
        self.parfiles = []
        self.logfiles = []
        self.errfiles = []
        self.allfiles = []
        self.has_parfile = False
        self.__timeseries = None
        self.__multipoles = None
        self.__gravitationalwaves = None
        self.__electromagneticwaves = None
        self.__gridfunctions = None
        self.__horizons = None
        self.__timers = None
        self.__twopunctures = None

        if (pickle_file is None) or (not os.path.exists(pickle_file)):
            self._populate()
        else:
            sim = load_SimDir(pickle_file)
            # Overwrite all the local variables
            self.__dict__ = sim.__dict__

        # We set this later, so that if it was read from the pickle, we override
        # it in such a way that we have consistency.
        self.pickle_file = pickle_file

    def _populate(self):
        """Scan the folders and populate basic attributes."""

        self._scan_folders(self.max_depth)

        self.__timeseries = None
        self.__multipoles = None
        self.__gravitationalwaves = None
        self.__electromagneticwaves = None
        self.__gridfunctions = None
        self.__horizons = None
        self.__timers = None
        self.__twopunctures = None

    def rescan(self):
        """Reset the SimDir and rescan all the files."""
        self._populate()

    @property
    def timeseries(self):
        """Return all the available timeseries in the data.

        :returns: Interface to all the timeseries in the directory.
        :rtype: :py:class:`~.ScalarsDir`
        """
        if self.__timeseries is None:
            self.__timeseries = cactus_scalars.ScalarsDir(self)
        return self.__timeseries

    ts = timeseries

    @property
    def multipoles(self):
        """Return all the available multipole data.

        :returns: Interface to all the multipole data in the directory.
        :rtype: :py:class:`~.MultipolesDir`
        """
        if self.__multipoles is None:
            self.__multipoles = cactus_multipoles.MultipolesDir(self)
        return self.__multipoles

    @property
    def gravitationalwaves(self):
        """Return all the available ``Psi4`` data.

        :returns: Interface to all the ``Psi4`` data in the directory.
        :rtype: :py:class:`~.GravitationalWavesDir`
        """
        if self.__gravitationalwaves is None:
            self.__gravitationalwaves = cactus_waves.GravitationalWavesDir(
                self
            )
        return self.__gravitationalwaves

    gws = gravitationalwaves

    @property
    def electromagneticwaves(self):
        """Return all the available ``Phi2`` data.

        :returns: Interface to all the ``Phi2`` data in the directory.
        :rtype: :py:class:`~.ElectromagneticWavesDir`
        """
        if self.__electromagneticwaves is None:
            self.__electromagneticwaves = cactus_waves.ElectromagneticWavesDir(
                self
            )
        return self.__electromagneticwaves

    emws = electromagneticwaves

    @property
    def gridfunctions(self):
        """Return all the available grid data.

        :returns: Interface to all the grid data in the directory.
        :rtype: :py:class:`~.GridFunctionsDir`
        """
        if self.__gridfunctions is None:
            self.__gridfunctions = cactus_grid_functions.GridFunctionsDir(self)
        return self.__gridfunctions

    gf = gridfunctions

    @property
    def horizons(self):
        """Return all the available horizon data.

        :returns: Interface to all the horizon data in the directory.
        :rtype: :py:class:`~.HorizonsDir`
        """
        if self.__horizons is None:
            self.__horizons = cactus_horizons.HorizonsDir(self)
        return self.__horizons

    @property
    def timers(self) -> cactus_timers.TimersDir:
        """Return all the available timertree data.

        :returns: Interface to all the timertree data in the directory.
        :rtype: :py:class:`~.TimertreeDir`
        """
        if self.__timers is None:
            self.__timers = cactus_timers.TimersDir(self)
        return self.__timers

    @property
    def twopunctures(self):
        """Return the metadata for TwoPunctures.

        :returns: Interface to the metadata in TwoPunctures.
        :rtype: :py:class:`~.TwoPuncturesDir`
        """
        if self.__twopunctures is None:
            self.__twopunctures = cactus_twopunctures.TwoPuncturesDir(self)
        return self.__twopunctures

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

        tim_ret = f"\n{self.timers}"

        return (
            header
            + ts_ret
            + mp_ret
            + gw_ret
            + em_ret
            + gf_ret
            + hor_ret
            + tim_ret
        )

    def __enter__(self):
        """This is classed when the object is used as a context manager."""
        # All the work in done in __init__
        return self

    def __exit__(self, _1, _2, _3):
        """Save the SimDir to disk as pickle.

        This is called when the object is used as a context manager.

        """
        if self.pickle_file is not None:
            self.save(self.pickle_file)

    def save(self, path):
        """Save this object as a pickle.

        The object can be loaded with the function :py:func:`~.load_SimDir`.

        :param path: Path where to save the file.
        :type path: str

        """
        with open(path, "wb") as file_:
            pickle.dump(self, file_, protocol=pickle.HIGHEST_PROTOCOL)
