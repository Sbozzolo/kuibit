#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
#
# Inspired by code originally developed by Wolfgang Kastaun. See, GitHub,
# wokast/PyCactus/PostCactus/cactus_grid_ascii.py, cactus_grid_h5.py,
# cactus_grid_omni.py
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

"""The :py:mod:`~.cactus_grid` module provides functions to load grid function
in Cactus formats.

There are multiple classes defined in this module:

- :py:class`~.GridFunctionsDir` interfaces with :py:class:`~.SimDir` and
  organizes the grid functions by dimensionality. This is a dictionary-like
  object with keys the possible dimensions (e.g., ``x``, ``yz``, ``xyz``).
- :py:class`~.AllGridFunctions` takes all the files in SimDir and sort them
  according the different grid functions they contain.
- There are two :py:class`~.OneGridFunction` classes, one for HDF5 files and one
  for ASCII files. They describe one single grid function and they contains the
  files associated to that grid function. Both the classes are derived from the
  same abstract base class :py:class`~.OneGridFunctionBase`, which implements
  the shared methods.

These are hierarchical classes, one containing the others, so one typically ends
up with a series of brackets to access the actual data. For example, if ``sim``
is a :py:class:`~.SimDir`, ``sim.gf.xy['rho_b'][0]`` is ``rho_b`` at iteration 0
on the equatorial plane represented as :py:class:`~.HierarchicalGridData`.

"""

import os
import re
import warnings
from abc import ABC, abstractmethod
from bz2 import open as bopen
from contextlib import contextmanager
from functools import lru_cache
from gzip import open as gopen

import h5py
import numpy as np

from kuibit import grid_data, simdir
from kuibit.attr_dict import pythonize_name_dict
from kuibit.cactus_ascii_utils import scan_header, total_filesize


class BaseOneGridFunction(ABC):
    """Abstract class that implements capabilities to handle grid functions.

    This class is the parent class of :py:class:`~.OneGridFunctionASCII` and
    :py:class:`~.OneGridFunctionH5`. :py:class:`~.BaseOneGridFunction`
    implements most methods, except the readers.

    The derived classes have to specify:

    - How to read a file, populating the ``self.alldata`` dictionary
      (method ``_parse_file``).
    - How to populate the last level of ``self.alldata`` by returning
      a :py:class:`~.UniformGridData` for a given iteration and component
      (method ``_read_componenent_as_uniform_grid``)
    - How to associate an iteration with a time (method
      ``time_at_iteration``).

    The simplest way to access data at a given iteration as
    :py:class:`~.HierarchicalGridData` is using the ``[]`` notation.

    :ivar allfiles: Paths of files associated to the variable.
    :type allfiles: list of str
    :ivar alldata: Dictionary that organizes files and iterations available.
    :type alldata: nested dictionary
    :ivar restarts_data: How iterations are distributed across files.
    :type restarts_data: tuple of str
    :ivar var_name: Variable name.
    :type var_name: str

    """

    def __init__(self, allfiles, var_name):
        """Constructor.

        :param allfiles: Paths of files associated to the variable.
        :type allfiles: list of str
        :param var_name: Variable name.
        :type var_name: str

        """

        self.allfiles = list(allfiles)

        # self.alldata is a nested dictionary
        # 1. At the first level, we have the file
        # 2. self.alldata[filename] is a dictionary with keys the various
        #    available iterations in filename
        # 3. self.alldata[filename][iteration] is another dictionary with keys the
        #    various refinement levels available in filename at the iteration
        #    and as values another dictionary
        # 4. This last dictionary has as keys the available components and as
        #    values None if the data has not been read yet, or UniformGridData
        #    for that component if the data has been read
        self.alldata = {}

        # We use this to extract only the information related to the specific
        # variable
        self.var_name = var_name

        for path in self.allfiles:
            self._parse_file(path)

        # Here we are going to save the restart information
        self.restarts_data = None

    # The derived classes have to specify:
    # 1. How to read a file, populating the self.alldata dictionary up to the
    #    last level (excluded or included) (_parse_file). This should also
    # 2. How to populate the last level of self.alldata by returning
    #    a UniformGridData for a given iteration and component
    #    (_read_componenent_as_uniform_grid)
    # 3. How to associate an iteration with a time (time_at_iteration)

    @abstractmethod
    def _parse_file(self, path):
        """Read file at path and populate ``self.alldata``."""
        raise NotImplementedError

    @abstractmethod
    def _read_component_as_uniform_grid_data(
        self, path, iteration, ref_level, component
    ):
        """Read specific component."""
        raise NotImplementedError

    @abstractmethod
    def time_at_iteration(self, iteration):
        """Return the time at a given iteration.

        :param iteration: Iteration.
        :type iteration: int
        :returns: Time.
        :rtype: float
        """
        raise NotImplementedError

    @lru_cache(128)
    def _iterations_in_file(self, path):
        """Return the (sorted) available iterations in file path.

        Use this if you need to ensure that you are looping over iterations in
        order!

        :param path: File to inspect.
        :type path: str

        :returns: Sorted list of iterations available in a given file.
        :rtype: list

        """
        return sorted(list(self.alldata[path].keys()))

    def _min_iteration_in_file(self, path):
        """Return the minimum available iterations in the given file.

        :param path: File to inspect.
        :type path: str

        :returns: First iteration available.
        :rtype: int

        """
        return self._iterations_in_file(path)[0]

    def _max_iteration_in_file(self, path):
        """Return the maximum available iterations in the given file.

        :param path: File to inspect.
        :type path: str

        :returns: Last iteration available.
        :rtype: int

        """
        return self._iterations_in_file(path)[-1]

    def _get_restarts(self):
        """Return a list of tuples of the form ``(iteration_min, iteration_max, paths)``
        with the minimum iterations and maximum iterations in each file
        associated to this variable. ``paths`` is a list of all the files with
        same ``min_iteration`` and ``max_iteration`` (as when we have 3d
        xyz_file files).

        This routine is used to identify which files to use for any given
        iteration.

        :returns: List of tuples with first iteration, last iteration, and
                  path for every file in ``self.allfiles``.
        :rtype: list of tuple (int, int, list of str)

        """
        restarts = [
            (
                self._min_iteration_in_file(path),
                self._max_iteration_in_file(path),
                [path],
            )
            for path in self.allfiles
        ]
        # We sort by first the minimum iteration. If the minimum iteration is
        # the same, we sort by the -maximum iteration, which ensures that where
        # we have more iterations is placed before. Consider the example of
        # list to be sorted [(1, 2), (0, 3), (1, 5)], this would be sorted in
        # ascending order with respect as if it was [(1, -2), (0, -3), (1, -5)],
        # so [(0, 3), (1, 5), (1, 2)]
        restarts.sort(key=lambda x: (x[0], -x[1]))

        # Next, we check that there is no overlap. If there's overlap, we
        # ignore some folders, unless min_iteration and max_iteration are
        # exactly the same, in which case we combine the two
        first, *others = restarts
        # We assemble a return list, starting with the first element. Then,
        # we loop over the other elements and we keep expanding the return
        # list only if we find that the maximum iteration is larger than
        # the previous maximum iteration. Since we sorted as described above,
        # this will ignore repeated runs in which the iterations are a subset
        # of the previous one.
        ret = [first]
        for min_iteration, max_iteration, path in others:
            # If we have exactly the same iterations, we are looking at ones
            # of those 3D files, so we collect them
            if (min_iteration, max_iteration) == (ret[-1][0], ret[-1][1]):
                ret[-1][2].append(path)
                continue

            max_iteration_in_ret = ret[-1][1]
            if max_iteration > max_iteration_in_ret:
                ret.append((min_iteration, max_iteration, path))
            else:
                warnings.warn(f"Unused (redundant) file: {path}")

        return ret

    @property
    def restarts(self):
        """Return a list of tuples of the form ``(iteration_min, iteration_max, paths)``
        with the minimum iterations and maximum iterations in each file
        associated to this variable. ``paths`` is a list of all the files with
        same ``min_iteration`` and ``max_iteration`` (as when we have 3d
        xyz_file files).

        :returns: List of tuples with first iteration, last iteration, and
                  path for every file in ``self.allfiles``.
        :rtype: list of tuple (int, int, list of str)
        """
        if self.restarts_data is None:
            self.restarts_data = self._get_restarts()
        return self.restarts_data

    @property
    def min_iteration(self):
        """Return the minimum available iteration in the all the files.

        :returns: First iteration available.
        :rtype: int

        """
        # restarts is a ordered list of tuples with three elements:
        # (iteration_min, iteration_max, path)

        # The minimum iteration is in restarts[0][0]
        return self.restarts[0][0]

    @property
    def max_iteration(self):
        """Return the maximum available iteration in the all the files.

        :returns: Latest iteration available.
        :rtype: int

        """
        # restarts is a ordered list of tuples with three elements:
        # (iteration_min, iteration_max, path)

        # The maximum iteration is in restarts[-1][1]
        return self.restarts[-1][1]

    @property
    @lru_cache(128)
    def available_iterations(self):
        """Return the available iterations.

        :returns: List with all the available iterations.
        :rtype: list

        """
        iterations_in_files = set()
        for path in self.allfiles:
            iterations_in_files.update(self._iterations_in_file(path))

        # Next we merge everything to make a set and we sort it
        return sorted(list(iterations_in_files))

    @property
    @lru_cache(128)
    def available_times(self):
        """Return the available times.

        :returns: List with all the available times.
        :rtype: list

        """
        return [
            self.time_at_iteration(iteration)
            for iteration in self.available_iterations
        ]

    times = available_times
    iterations = available_iterations

    def __iter__(self):
        for iteration in self.available_iterations:
            yield self[iteration]

    def iteration_at_time(self, time):
        """Return the iteration that corresponds to the given time.

        :param time: Time.
        :type time: float

        :returns: Iteration corresponding to the given time.
        :rtype: int

        """

        # TODO (FEATURE): Add tolerance in time.
        #
        # Often different outputs are not synced, so we need to allow for some
        # tolerance.

        if time not in self.available_times:
            raise ValueError(f"Time {time} not available")

        index = np.searchsorted(self.available_times, time)
        return self.available_iterations[index]

    def _ref_levels_in_file(self, path, iteration):
        """Return the available refinement levels in the given path at the
        specified iteration.

        :param path: Path of the file.
        :type path: str
        :param iteration: Iteration.
        :type iteration: int

        :returns: Available refinement levels in ``path`` at ``iteration``.
        :rtype: list
        """
        # In case we don't have the iteration, we want to return the empty
        # list, so we use the .get dictionary method. This return the
        # value, if the key is available, otherwise it returns the second
        # argument. Here, we fix as second argument the empty dictionary, so
        # when we take .keys() we get the empty list
        return list(self.alldata[path].get(iteration, {}).keys())

    def _components_in_file(self, path, iteration, ref_level):
        """Return the available components in the given file at the specified iteration
        and refinement level.

        :param path: Path of the file.
        :type path: str
        :param iteration: Iteration.
        :type iteration: int
        :param ref_level: Refinement level.
        :type ref_level: int

        :returns: Available components in ``path`` at ``iteration`` and ``ref_level``.
        :rtype: list

        """
        # Same comment as _ref_levels_in_file, but with an
        # additional level
        return list(
            self.alldata[path].get(iteration, {}).get(ref_level, {}).keys()
        )

    def _files_with_iteration(self, iteration):
        """Return all the files that contain the given iteration.

        :param path: Path of the file.
        :type path: str
        :param iteration: Iteration.
        :type iteration: int

        :returns: Files that contain the given iteration.
        :rtype: list

        """
        # Using self.get_restarts(), find the file that the given iteration
        # between iteration_min and iteration_max

        if (iteration < self.min_iteration) or (
            iteration > self.max_iteration
        ):
            raise ValueError(f"Iteration {iteration} not in available range")

        # restarts is a ordered list of tuples with three elements:
        # (iteration_min, iteration_max, path)
        max_iterations_in_files = np.array([i for _, i, _ in self.restarts])
        # This returns the index of the first element in max_iterations_in_files
        # that is greater than the given iteration.
        # E.g. max_iterations_in_files = [1, 5, 10] and iteration = 6, since
        # max_iterations_in_files are the maximum iterations in the various files,
        # the function has to return 2, which is the index of the first element
        # in max_iterations_in_files that is larger than 6. This is what
        # np.searchsorted does.
        index = np.searchsorted(max_iterations_in_files, iteration)

        # The second element is the list of path
        return self.restarts[index][2]

    def total_filesize(self, unit="MB"):
        """Return the total size of the files with this variable.
        Available units B, KB, MB and GB (in power of 1024 bytes).

        :param unit: Unit to use (in powers of 1024 bytes).
        :type unit: str among: ``B``, ``KB``, ``MB``, ``GB``.
        :returns: Total size of the files associated to this variable.
        :rtype: float

        """
        return total_filesize(self.allfiles, unit=unit)

    @lru_cache(128)
    def _read_iteration_as_HierarchicalGridData(self, iteration):
        """Return the data at the given iteration as a :py:class:`~.HierarchicalGridData`.

        :param iteration: Iteration.
        :type iteration: int

        :returns: Variable at the given iteration as a
                  :py:class:`~.HierarchicalGridData`.
        :rtype: :py:class:`~.HierarchicalGridData`

        """

        uniform_grid_data_components = []

        for path in self.allfiles:
            for ref_level in self._ref_levels_in_file(path, iteration):
                for comp in self._components_in_file(
                    path, iteration, ref_level
                ):
                    uniform_grid_data_components.append(
                        self._read_component_as_uniform_grid_data(
                            path, iteration, ref_level, comp
                        )
                    )

        return (
            grid_data.HierarchicalGridData(uniform_grid_data_components)
            if uniform_grid_data_components
            else None
        )

    @lru_cache(128)
    def get_iteration(self, iteration, default=None):
        """Return the data at the given iteration as a :py:class:`~.HierarchicalGridData`.
        If the iteration is not available, return ``default``.

        :param iteration: Iteration.
        :type iteration: int
        :param default: What to return if iteration is not available.
        :type default: anything

        :returns: Variable at the given iteration as a
                  :py:class:`~.HierarchicalGridData`.
        :rtype: :py:class:`~.HierarchicalGridData`

        """

        if iteration not in self.available_iterations:
            return default
        return self[iteration]

    @lru_cache(128)
    def get_time(self, time, default=None):
        """Return the data at the given time as a :py:class:`~.HierarchicalGridData`.
        If the time is not available, return ``default``.

        :param iteration: Iteration.
        :type iteration: int
        :param default: What to return if time is not available.
        :type default: anything

        :returns: Variable at the given time as a
                  :py:class:`~.HierarchicalGridData`.
        :rtype: :py:class:`~.HierarchicalGridData`

        """
        if time not in self.available_times:
            return default
        return self[self.iteration_at_time(time)]

    def __getitem__(self, iteration):
        if iteration not in self.available_iterations:
            raise KeyError(f"Iteration {iteration} not present")

        return self._read_iteration_as_HierarchicalGridData(iteration)

    def read_on_grid(self, iteration, grid, resample=False):
        """Read an iteration and resample the output on the specified grid.

        Warning: this can be computationally expensive!

        :param iteration: requested iteration
        :type iteration: time
        :param grid:
        :type grid: UniformGrid
        :param resample: Whether to use multilinear interpolation
        :type resample: bool
        """
        return self[iteration].to_UniformGridData_from_grid(
            grid, resample=resample
        )

    # def read_evolution_on_grid(
    #     self,
    #     grid,
    #     read_every=None,
    #     min_iteration=None,
    #     max_iteration=None,
    #     **kwargs,
    # ):
    #     """Read multiple iterations at once on the specified grid and return the result
    #     as RegularGridData in which the first index is the time and the other indices
    #     are the spatial indices of grid.

    #     """
    #     # TODO: IMPLEMENT THIS

    #     iterations = self.available_iterations
    #     if min_iteration is not None:
    #         iterations = iterations[iterations >= min_iteration]
    #     if max_iteration is not None:
    #         iterations = iterations[iterations <= max_iteration]
    #     if read_every is None:
    #         read_every = np.diff(iterations).max()

    #     iterations = [
    #         i for i in iterations if ((i - iterations[0]) % read_every == 0)
    #     ]
    #     times = [self.time_at_iteration(i) for i in iterations]

    #     dt = np.diff(times).min()
    #     if dt <= 0:
    #         raise RuntimeError("Non-positive timesteps detected.")
    #     #
    #     if abs(np.diff(times).max() - dt).max() > dt * 1e-5:
    #         raise RuntimeError("Timestep not constant enough")

    #     data = np.asarray(
    #         [self.read_on_grid(i, grid).data for i in iterations]
    #     )

    #     new_x0 = [times[0]] + list(grid.x0)
    #     new_dx = [dt] + list(grid.dx)

    #     return grid_data.UniformGridData.from_grid_structure(
    #         data, x0=new_x0, dx=new_dx
    #     )


class OneGridFunctionASCII(BaseOneGridFunction):
    """Read grid data produced by CarpetASCII.

    This class is derived from :py:class:`~.BaseOneGridFunction` and implements
    the reading facilities.

    :py:class:`~.OneGridFunctionASCII` can read 1D, 2D, and 3D ASCII files, even
    when they are compressed with bzip2 or gzip.

    ASCII files do not contain information about the ghost zones, but this can be
    set "by hand".

    """

    # TODO (REFACTORING): Avoid reading files twice.
    #
    # This class has to read the all files. When there are multiple variables in
    # one file, it would be better to avoid reading again the various files (if
    # we have already read them). Maybe we can add another level in the class
    # hierarchy that contains all the information for a given file, and produces
    # transparently OneGridFunctionASCII objects upon request.

    # What function to use to open the file?
    # What mode?
    _decompressor = {
        None: (open, "r"),
        "gz": (gopen, "rt"),
        "bz2": (bopen, "rt"),
    }

    def __init__(self, allfiles, var_name, num_ghost=None):
        """Constructor.

        :param allfiles: Paths of files associated to the variable.
        :type allfiles: list of str
        :param var_name: Variable name.
        :type var_name: str
        :param num_ghost: Number of ghost zones in each direction.
        :type num_ghost: 1d NumPy array

        """

        self._iterations_to_times = {}
        self.num_ghost = num_ghost

        super().__init__(allfiles, var_name)

    def _parse_file(self, path):
        """Read the content of the given file.

        At the moment, we read the entire file line by line, which is very
        inefficient.

        :param path: Path of the file to read.
        :type path: str

        """

        # First we parse the header to find the column description, then we read
        # the ENTIRE file. This is very inefficient, but it is not too hard to
        # implement.

        # This regex is meant to understand if we have one variable per file or
        # one group per file, and to understand if we have compression. To see a
        # detailed explanation, see AllGridFunctions. The only difference here
        # is that we don't care about the extension, so we have an addition (*)?
        rx_filename = re.compile(
            r"^(([a-zA-Z0-9_]+)-)?([a-zA-Z0-9\[\]_]+).([xyz]+)?.asc(\.(gz|bz2))?$"
        )

        filename = os.path.split(path)[1]
        matched = rx_filename.match(filename)

        if matched is None:
            raise RuntimeError(f"Found file with unusual name: {path}")

        is_one_file_per_group = matched.group(1) is not None

        compression_method = matched.group(6)
        opener, opener_mode = OneGridFunctionASCII._decompressor[
            compression_method
        ]

        # These files always have the column format line, and have the data
        # format line only if they are "one file per group"
        _, column_description = scan_header(
            path,
            one_file_per_group=is_one_file_per_group,
            extended_format=True,
            opener=opener,
            opener_mode=opener_mode,
        )
        # We have two possibilities, one is that the file only contains one
        # variable, column_description will be the column number. If the
        # file contains many variables, column_description is a dictionary
        # that maps variables to their column.
        if isinstance(column_description, dict):
            # The variable we work with is column_description so we overwrite it
            # to be the number of column with the data we are interested in
            column_description = column_description[self.var_name]

        # Now we read the entire file, line by line. This is the most
        # inefficient way possible. But at least, it is reasonably
        # straightforward to implement. If you are reading this comment and you
        # want to improve this, feel free to do it.

        def current_data_to_UniformGridData(
            current_x,
            current_y,
            current_z,
            current_data,
            current_time,
            current_iteration,
            current_component,
            current_ref_level,
        ):
            # First, we compute x0 and x1
            x0_3d = np.asarray(
                [
                    np.amin(current_x),
                    np.amin(current_y),
                    np.amin(current_z),
                ]
            )
            x1_3d = np.array(
                [
                    np.amax(current_x),
                    np.amax(current_y),
                    np.amax(current_z),
                ]
            )

            # Now we find the interesting dimensions
            dimensions_in_data = x0_3d != x1_3d

            # With unique we find the real data
            shape_3d = [
                len(np.unique(current_x)),
                len(np.unique(current_y)),
                len(np.unique(current_z)),
            ]

            shape = np.asarray(shape_3d)[dimensions_in_data]
            x0 = np.asarray(x0_3d)[dimensions_in_data]
            x1 = np.asarray(x1_3d)[dimensions_in_data]

            var_data = np.array(current_data).reshape(tuple(shape[::-1]))

            grid = grid_data.UniformGrid(
                shape,
                x0=x0,
                x1=x1,
                num_ghost=self.num_ghost,
                component=current_component,
                ref_level=current_ref_level,
                time=current_time,
                iteration=current_iteration,
            )

            return grid_data.UniformGridData(grid, np.transpose(var_data))

        # We are going to assume that the iteration column is the first
        with opener(path, opener_mode) as fil:
            # We use these variables as local variables. We are going to aggregate
            # the data read here, and reset it when we find a blank line
            current_iteration = None
            current_ref_level = None
            current_component = None
            current_x = []
            current_y = []
            current_z = []
            current_data = []
            # We scan the entire file line by line
            for line in fil:
                # Skip header
                if not line[0].isdigit():
                    # We don't care about lines that don't start with a number
                    continue

                # Here are can assume that this is a line with data
                line_data = line.split()
                line_data = list(map(float, line_data))
                if current_iteration is None:
                    current_iteration = line_data[0]

                if current_ref_level is None:
                    current_ref_level = line_data[2]

                if current_component is None:
                    current_component = line_data[3]

                # If iteration, component, or refinement level changes, we
                # write the data, else we continue reading
                if (
                    line_data[0] != current_iteration
                    or line_data[2] != current_ref_level
                    or line_data[3] != current_component
                ):
                    alldata_file = self.alldata.setdefault(path, {})
                    alldata_iteration = alldata_file.setdefault(
                        int(current_iteration), {}
                    )
                    alldata_ref_level = alldata_iteration.setdefault(
                        int(current_ref_level), {}
                    )

                    current_time = line_data[8]

                    uniform_grid_data = current_data_to_UniformGridData(
                        current_x,
                        current_y,
                        current_z,
                        current_data,
                        current_time,
                        current_iteration,
                        current_component,
                        current_ref_level,
                    )

                    alldata_ref_level.setdefault(
                        int(current_component), uniform_grid_data
                    )

                    # Write iterations_to_time
                    if current_iteration not in self._iterations_to_times:
                        self._iterations_to_times[
                            current_iteration
                        ] = current_time

                    # Reset everything
                    current_iteration = line_data[0]
                    current_ref_level = line_data[2]
                    current_component = line_data[3]
                    current_x = []
                    current_y = []
                    current_z = []
                    current_data = []

                # We still have to read the data on this line even if we
                # "are done with a group"
                current_x.append(line_data[9])
                current_y.append(line_data[10])
                current_z.append(line_data[11])

                current_data.append(line_data[column_description])

            # Here we take care of the last piece of data
            if len(current_data) > 0:
                if current_iteration not in self._iterations_to_times:
                    current_time = line_data[8]
                    self._iterations_to_times[current_iteration] = current_time

                alldata_file = self.alldata.setdefault(path, {})
                alldata_iteration = alldata_file.setdefault(
                    int(current_iteration), {}
                )
                alldata_ref_level = alldata_iteration.setdefault(
                    int(current_ref_level), {}
                )

                uniform_grid_data = current_data_to_UniformGridData(
                    current_x,
                    current_y,
                    current_z,
                    current_data,
                    current_time,
                    current_iteration,
                    current_component,
                    current_ref_level,
                )

                alldata_ref_level.setdefault(
                    int(current_component), uniform_grid_data
                )

    def _read_component_as_uniform_grid_data(
        self, path, iteration, ref_level, component
    ):
        """Return the component at the given iteration, refinement level, and component.

        :param path: Path of the file.
        :type path: str
        :param iteration: Iteration.
        :type iteration: int
        :param ref_level: Refinement level.
        :type ref_level: int
        :param component: Component.
        :type component: int

        :returns: Component as a :py:class:`~.UniformGridData`.
        :rtype: :py:class:`~.UniformGridData`

        """

        # We have already read the files, so we just return it
        return self.alldata[path][iteration][ref_level][component]

    def time_at_iteration(self, iteration):
        """Return the time at a given iteration.

        :param iteration: Iteration.
        :type iteration: int

        :returns: Time at given iteration.
        :rtype: float

        """

        if iteration not in self.available_iterations:
            raise ValueError("Iteration {iteration} not available")

        return self._iterations_to_times[iteration]


class OneGridFunctionH5(BaseOneGridFunction):
    """Read grid data produced by CarpetHDF5 files.

    This class is derived from :py:class:`~.BaseOneGridFunction` and implements
    the reading facilities.

    """

    # This class implements the details on how to read the data, most of the
    # functionalities of the class are in OneGridFunctionBase.

    # Let's unpack the regex, we have 7 capturing groups
    #
    # 1. [^:] matches any number of characters (at least one), with the
    #    exception of ':'
    # 2. \S+ matches any number of non-whitespace characters (at least
    #    one).
    #
    # 1. and 2. are thorn and variable name, then we have
    # 3, 4. \d+ matches a number (iteration and time level)
    # 5. " m=0" matches this exactly, if it is present
    # 6. \d+ matches a number (refinement level) if present
    #    (grid arrays don't have this)
    # Then we have two nested capturing groups
    # ( c=(\d+))? checked whether c=NUMBER is matched,
    # and inside we have that the component number is matched
    #
    # All the [ ] are meaningful blank spaces. Note, we don't have $^ because
    # there is more that we do not want to match
    _pattern_group_name = r"""
    ([^:]+)             # Thorn name
    ::
    (\S+)               # Variable name
    [ ]
    it=(\d+)            # Iteration
    [ ]
    tl=(\d+)            # Timelevel (almost always 0)
    ([ ]m=0)?           # Map
    ([ ]rl=(\d+))?      # Refinement level
    ([ ]c=(\d+))?       # Component
    """

    def __init__(self, allfiles, var_name):
        """Constructor.

        :param allfiles: Paths of files associated to the variable.
        :type allfiles: list of str
        :param var_name: Variable name.
        :type var_name: str

        """

        # We need these variables to properly find what dataset to look at in
        # the HDF5 file.
        self.thorn_name = None
        self.map = None

        self.rx_group_name = re.compile(self._pattern_group_name, re.VERBOSE)

        super().__init__(allfiles, var_name)

        # super() will fill the other variables that we need for dataset_format
        if self.map is None:
            self.map = ""

        # TODO (FEATURE): Make separator customizable
        #
        # Technically the separator :: is customizable, so we should be more
        # flexible.

        self.dataset_format = (
            f"{self.thorn_name}::{self.var_name} it=%d tl=0{self.map}%s%s"
        )

        # HDF5 files can contain ghostzones or not. Here, we can that all the
        # files have the same behavior (they all contain, or they all don't)
        #
        # self._are_ghostzones_in_file(path) returns True or False, so this
        # is a set with True, False or a mix
        ghost_in_files = {
            self._are_ghostzones_in_file(path) for path in self.allfiles
        }

        # Here we check that we only have True or False
        if len(ghost_in_files) != 1:
            raise ValueError(
                "Inconsistent IOHDF5::output_ghost_points across files"
            )

        # We know that ghost_in_files has only one element (either True or
        # False), so we pick that (with tuple unpacking)
        (self.are_ghostzones_in_files,) = ghost_in_files

    def _parse_file(self, path):
        """Read the content of the given file (without reading the data).

        :param path: Path of the file.
        :type path: str

        """
        # This will give us an overview of what is available in the provided
        # file. We keep a collection of all these in the variable self.alldata
        with h5py.File(path, "r") as f:
            for group in f.keys():
                matched = self.rx_group_name.match(group)
                # If this is not an interesting group, just skip it
                if not matched:
                    continue

                (
                    thorn_name,
                    var_name,
                    iteration,
                    time_level,
                    map_,
                    _,
                    ref_level,
                    _,
                    component,
                ) = matched.groups()

                if var_name != self.var_name:
                    continue

                time_level = int(time_level)

                # We only care about the current timelevel
                if time_level != 0:
                    continue

                if self.thorn_name is None:
                    self.thorn_name = thorn_name

                if self.map is None:
                    self.map = map_

                component = -1 if matched.group(9) is None else int(component)
                # This is important to support grid arrays, which do not have a
                # refinement level
                ref_level = -1 if matched.group(7) is None else int(ref_level)

                # Here is where we prepare are nested alldata dictionary
                alldata_file = self.alldata.setdefault(path, {})
                alldata_iteration = alldata_file.setdefault(int(iteration), {})
                alldata_ref_level = alldata_iteration.setdefault(ref_level, {})

                # We set the actual data to None, and we will read it in
                # _read_component_as_uniform_grid_data upon request
                alldata_ref_level.setdefault(int(component), None)

    def _grid_from_dataset(self, dataset, iteration, ref_level, component):
        """Return a :py:class:`~.UniformGrid` from a given HDF5 dataset.

        :param dataset: Dataset to model the grid after.
        :type dataset: H5py.dataset
        :param iteration: Iteration.
        :type iteration: int
        :param ref_level: Refinement level.
        :type ref_level: int
        :param component: Component.
        :type component: int

        :returns: Grid corresponding to the dataset.
        :rtype: :py:class:`~.UniformGrid`

        """

        # NOTE: Why are we taking the reverse?
        shape = np.array(dataset.shape[::-1])

        x0 = dataset.attrs["origin"]
        dx = dataset.attrs["delta"]
        time = dataset.attrs["time"]
        # With the .get we ensure that if "cctk_nghostzones" cannot be read, we
        # have returned None, which we can test later
        num_ghost = dataset.attrs.get("cctk_nghostzones", None)
        # If we do not have the ghostzones in the file, then it is as if we
        # have ghostzones of size zero.
        if not self.are_ghostzones_in_files or num_ghost is None:
            num_ghost = np.zeros_like(shape, dtype=int)

        return grid_data.UniformGrid(
            shape,
            x0=x0,
            dx=dx,
            ref_level=ref_level,
            num_ghost=num_ghost,
            time=time,
            iteration=iteration,
            component=component,
        )

    # What is a context manager?
    #
    # Context managers are useful ways to handle resources in Python. With a
    # context manager, we do not have to worry about releasing resources. Here,
    # we wrap reading the h5 file with another context manager so that we can
    # easily get the dataset.
    @contextmanager
    def _get_dataset(self, path, iteration, ref_level, component):
        """Context manager to read an HDF5 file.

        :param path: Path of the file.
        :type path: str
        :param iteration: Iteration.
        :type iteration: int
        :param ref_level: Refinement level.
        :type ref_level: int
        :param component: Component.
        :type component: int

        """
        ref_level_str = f" rl={ref_level}" if (ref_level >= 0) else ""
        component_str = f" c={component}" if (component >= 0) else ""
        with h5py.File(path, "r") as f:
            try:
                yield f[
                    self.dataset_format
                    % (iteration, ref_level_str, component_str)
                ]
            finally:
                # All the hard work is done by the other 'with' statement.
                # We don't need to do anything here.
                pass

    def _read_component_as_uniform_grid_data(
        self, path, iteration, ref_level, component
    ):
        """Return the component at the given iteration, refinement level, and component.

        :param path: Path of the file.
        :type path: str
        :param iteration: Iteration.
        :type iteration: int
        :param ref_level: Refinement level.
        :type ref_level: int
        :param component: Component.
        :type component: int

        :returns: Component as a :py:class:`~.UniformGridData`.
        :rtype: :py:class:`~.UniformGridData`

        """

        if self.alldata[path][iteration][ref_level][component] is None:
            with self._get_dataset(
                path, iteration, ref_level, component
            ) as dataset:
                grid = self._grid_from_dataset(
                    dataset, iteration, ref_level, component
                )
                data = np.transpose(dataset[()])

                self.alldata[path][iteration][ref_level][
                    component
                ] = grid_data.UniformGridData(grid, data)

        return self.alldata[path][iteration][ref_level][component]

    @staticmethod
    def _are_ghostzones_in_file(path):
        """Return whether the ghostzones were output or not.

        :param path: File to inspect.
        :type path: str

        :returns: Whether ``path`` contains ghost zones.
        :rtype: bool

        """
        # This is a tricky and important function to stitch together all the
        # different components. Carpet has an option (technically two) to output
        # the ghostzones in the files. These are: output_ghost_points and
        # out3D_ghosts (which is deprecated). When they are both set to yes,
        # the ghostzones are output in the h5 files. When one of the two is set
        # to no, the ghostzones are not output.

        # The default value of these parameters is yes
        with h5py.File(path, "r") as f:
            parameters = f["Parameters and Global Attributes"]
            all_pars = parameters["All Parameters"][()].decode().split("\n")
            # We make sure that everything is lowercase, we are case insensitive
            iohdf5_pars = [
                param.lower()
                for param in all_pars
                if param.lower().startswith("carpetiohdf5")
                or param.lower().startswith("iohdf5")
            ]

            def is_param_true(name):
                param = [p for p in iohdf5_pars if name.lower() in p]
                # When the parameters are not set, they are set to yes by
                # default
                if len(param) == 0:
                    return True

                # The parameter is set
                return (
                    ("true" in param[0])
                    or ("yes" in param[0])
                    or ("1" in param[0])
                )

            return is_param_true("out3D_ghosts") and is_param_true(
                "output_ghost_points"
            )

    def time_at_iteration(self, iteration):
        """Return the time corresponding to the provided iteration.

        :param iteration: Iteration.
        :type iteration: int

        :returns: Time corresponding to ``iteration``.
        :rtype: float

        """
        # If there are multiple files, we take the first.
        # A case in which there are multiple files is with 3D data
        path = self._files_with_iteration(iteration)[0]

        ref_levels = self._ref_levels_in_file(path, iteration)
        components = self._components_in_file(path, iteration, ref_levels[-1])
        with self._get_dataset(
            path, iteration, ref_levels[-1], components[-1]
        ) as dataset:
            return dataset.attrs["time"]


class AllGridFunctions:
    """Helper class to read various types of grid data in a list of files and
    properly order them. The core of this object is the ``_vars`` dictionary
    which contains the location of all the files for a specific variable and
    reduction.

    :py:class:`~.AllGridFunction` is a dictionary-like object with keys the
    various variables and values :py:class:`~.BaseOneGridFunction` (or derived).

    You can access data with the bracket operator or as attributes of the
    ``fields`` attribute.

    Not intended for direct initialization.

    :ivar dimension: Dimension associated to this object (e.g. (0, 1) would be the
                     xy plane).
    :type dimension: tuple
    :ivar num_ghost: Number of ghost zones in each dimension.
    :type num_ghost: 1d NumPy array.

    """

    # Different "cuts" have different extensions in their filenames, here we
    # save all the possible ones. In the instance of the class, we fix the
    # specific pattern corresponding to the dimension (which are the keys on
    # the following dictionary). In general, the file name will be:
    # variable-name.ext.h5, eg rho.xy.h5.
    filename_extensions = {
        (0,): ".x",
        (1,): ".y",
        (2,): ".z",
        (0, 1): "(.0)?.xy",
        (0, 2): "(.0)?.xz",
        (1, 2): "(.0)?.yz",
        (0, 1, 2): r"(.xyz)?(.file_[\d]+)?(.xyz)?",
    }

    _dim_names = {
        (0,): "x",
        (1,): "y",
        (2,): "z",
        (0, 1): "xy",
        (0, 2): "xz",
        (1, 2): "yz",
        (0, 1, 2): "xyz",
    }

    def __init__(self, allfiles, dimension, num_ghost=None):
        """Constructor.

        :param allfiles: List of all the files.
        :type allfiles: list
        :param dimension: Dimension associated to this object.
        :type dimension: tuple
        :param num_ghost: Number of ghost zones in the data for each dimension.
                          This is used only for ASCII data.
        :type num_ghost: list or tuple of the same length as the number of dimension

        """

        # Here we save what kind of file we are looking at
        # We assume that dimension is already sanitized (that is, is in tuple
        # form and not in string form)
        self.dimension = dimension

        # If we are using ASCII files, we have to know how many ghost zones are
        # in the data. At the moment we ask the user to provide the data, but
        # in the future we will parse the paramter file and find this value.
        #
        # We don't use this value for HDF5 data, as it is more reliable to just
        # read it from the files.
        # Here we are using a setter for num_ghost, see below
        self.num_ghost = num_ghost

        # This is a simple regex:
        # 1. ^ and $ mean that we have to match the entire string
        # 2. ([a-zA-Z0-9_]+) means that we match any combination of letters
        #    and numebrs. This is the thorn name when we output one group
        #    per file. We wrap this into another capturing group:
        #    (([a-zA-Z0-9_]+)-). Here we also try to match the literal '-'.
        #    This separates the thron name from the group name. If we match
        #    this larger capturing group, it means that the file was output
        #    with the option "one_group_per_file".
        # 3. ([a-zA-Z0-9\[\]_]+) means that we match any character any number
        #    of times, this is a capturing group, and is the variable name,
        #    or the group name if we output one group per file.
        # 4. We have the extension, which identifies the dimension and is
        #    saved in the class instance
        # 5. Finally, we have the filename extension which can be either h5
        #    or txt
        #
        # Example of filenames are:
        # admbase-metric.xyz.file_158.h5 (one group per file)
        # alp.xy.h5 (one variable per file)
        # filename_pattern = r"^([a-zA-Z0-9_]+)(-)?([a-zA-Z0-9\[\]_]+)%s.%s$"
        filename_pattern = r"^(([a-zA-Z0-9_]+)-)?([a-zA-Z0-9\[\]_]+)%s.%s$"
        h5_pattern = filename_pattern % (
            self.filename_extensions[self.dimension],
            "h5",
        )
        ascii_pattern = filename_pattern % (
            self.filename_extensions[self.dimension],
            r"asc(\.(gz|bz2))?",
        )

        # Variable files is a dictionary, the keys are the variables, the
        # values the set of files associated to that variable
        self._vars_ascii = {}
        self._vars_h5 = {}

        rx_h5 = re.compile(h5_pattern)
        rx_ascii = re.compile(ascii_pattern)

        # Here we scan all the files and find those with a name that match
        # one of our regular expressions.

        for f in allfiles:
            filename = os.path.split(f)[1]
            matched_h5 = rx_h5.match(filename)
            matched_ascii = rx_ascii.match(filename)
            # If matched_pattern is not None, this is a Carpet h5 file
            if matched_h5 is not None:
                # First, we understand if the file was output with
                # "one_group_per_file". In this case, the file contains
                # multiple variables. Files output in this way have names:
                # thorname-groupname.dim.h5 (possibly with also file_NUM).
                #
                # Group1 contains thorname- (notice the -). If we match
                # group1 it means that the file contains one group,
                # the thorn name is in group2, and the group name in group3.
                #
                # In case group1 is not matched, then the variable name is
                # in group3.
                if matched_h5.group(1) is None:
                    variable_name = matched_h5.group(3)
                    var_list = self._vars_h5.setdefault(variable_name, set())
                    var_list.add(f)
                else:
                    # We have to open the file to understand which variables
                    # are available
                    #
                    # We use the pattern name in OneGridFunctionH5
                    rx_group_name = re.compile(
                        OneGridFunctionH5._pattern_group_name, re.VERBOSE
                    )
                    with h5py.File(f, "r") as h5f:
                        # Here group is in the sense of HDF5 group
                        for group in h5f.keys():
                            group_matched = rx_group_name.match(group)
                            # If this is not an interesting group, just skip it
                            if not group_matched:
                                continue
                            variable_name = group_matched.group(2)
                            var_list = self._vars_h5.setdefault(
                                variable_name, set()
                            )
                            var_list.add(f)
            elif matched_ascii is not None:
                # As in the case of H5 files, we first need to understand if
                # the output is with "one_group_per_file". If yes, we have to
                # open all the files and read the headers to find what variables
                # are available.
                if matched_ascii.group(1) is None:
                    variable_name = matched_ascii.group(3)
                    var_list = self._vars_ascii.setdefault(
                        variable_name, set()
                    )
                    var_list.add(f)
                else:
                    # In this case we need to open the file and scan the
                    # header, for this we use the scan_header function in
                    # cactus_scalars.
                    #
                    # Here we have to pay attention to the output of the Thorns
                    # VolumeIntegralsGRMHD and VolumeIntegralsVacuum as they
                    # produce output files with names:
                    # volume_integrals-vacuum and volume_integrals-GRMHD.
                    #
                    # These would be matched here, so we will exclude the case in
                    # which the thorn name is volume_integrals and the var name
                    # is vacuum or GRMHD.
                    thorn_name = matched_ascii.group(2)
                    var_name = matched_ascii.group(3)
                    if thorn_name == "volume_integrals" and (
                        var_name in ("GRMHD", "vacuum")
                    ):
                        continue

                    # TODO (FEATURE): Avoid reading headers twice
                    #
                    # Here we scan the headers, we should not do this work again
                    # when we deal with the single variables.

                    # The last group is where compression information is. It
                    # could be None.
                    compression_method = matched_ascii.groups()[-1]
                    opener, opener_mode = OneGridFunctionASCII._decompressor[
                        compression_method
                    ]
                    _, column_description = scan_header(
                        f,
                        one_file_per_group=True,
                        extended_format=True,
                        opener=opener,
                        opener_mode=opener_mode,
                    )
                    for variable_name in column_description.keys():
                        var_list = self._vars_ascii.setdefault(
                            variable_name, set()
                        )
                        var_list.add(f)

        # What pythonize_name_dict does is to make the various variables
        # accessible as attributes, e.g. self.fields.rho
        self.fields = pythonize_name_dict(list(self.keys()), self.__getitem__)

    @lru_cache(128)
    def __getitem__(self, key):
        var_name = str(key)
        # We prefer h5
        if var_name in self._vars_h5:
            return OneGridFunctionH5(self._vars_h5[var_name], var_name)

        if var_name in self._vars_ascii:
            if self.num_ghost is None:
                warnings.warn(
                    "You are using ASCII files, which have no information"
                    " about ghost zone information. Set the attribute num_ghost"
                    " of this object to properly account for the ghost zones. "
                )
            return OneGridFunctionASCII(
                self._vars_ascii[var_name], var_name, num_ghost=self.num_ghost
            )

        raise KeyError(f"Variable {key} not present in simulation data")

    @property
    def num_ghost(self):
        """Return the number of ghost zones along each direction.

        :returns: Number of ghost zones along each direction.
        :rtype: 1d NumPy array
        """
        return self.__num_ghost

    @num_ghost.setter
    def num_ghost(self, num_ghost):
        if num_ghost is not None:
            if len(num_ghost) != len(self.dimension):
                raise ValueError(
                    "Number of ghost zones {len(num_ghost)} is inconsistent "
                    " with dimesionality (len(self.dimension))"
                )
            # We copy num_ghost, in case it was a mutable object
            self.__num_ghost = np.array(num_ghost)
        else:
            self.__num_ghost = None

    def __contains__(self, var):
        return var in self.keys()

    def get(self, key, default=None):
        """Return variable ``key``.

        :param key: Variable to read.
        :type key: str
        :param default: Value to return is variable is not available.
        :type default: anything
        """
        if key not in self:
            return default
        return self[key]

    def keys(self):
        """Return the list of all the available variables."""
        # We merge the dictionaries and return the keys.
        # This automatically takes care of making sure that they keys are unique.
        return {**self._vars_h5, **self._vars_ascii}.keys()

    def __str__(self):
        ret = "\nAvailable grid data of dimension "
        ret += f"{len(self.dimension)}D ({self._dim_names[self.dimension]}): "
        ret += f"\n{list(self.keys())}\n"
        return ret

    @property
    def allfiles(self):
        """Return a set of all the files that have variables of the
        given dimension

        :return: Collection of all the unique files with variables of
                 this dimension.
        :rtype: set
        """
        allfiles = set()
        # We collect all the files by merging the lists into a set. The
        # set will automatically remove repeated entries.
        for file_list in self._vars_h5.values():
            allfiles.update(file_list)
        for file_list in self._vars_ascii.values():
            allfiles.update(file_list)
        return allfiles

    def total_filesize(self, unit="MB"):
        """Return the total size of the files with this dimension.
        Available units B, KB, MB and GB (in power of 1024 bytes).

        :param unit: Unit to use (in powers of 1024 bytes).
        :type unit: str among: ``B``, ``KB``, ``MB``, ``GB``.
        :returns: Total size of all the files associated with this dimension.
        :rtype: float

        """
        return total_filesize(self.allfiles, unit=unit)


class GridFunctionsDir:
    """This class provides access to all grid data.

    This includes 1D-3D data in HDF5 and ASCII formats. Data of the required
    dimensionality is read from any format available (HDF5 preferred over
    ASCII). If you need lower dimensional data, read the higher dimensional one
    and slice the data.

    :ivar x:           Access to 1D data along x-axis.
    :ivar y:           Access to 1D data along y-axis.
    :ivar z:           Access to 1D data along z-axis.
    :ivar xy:          Access to 2D data along xy-plane.
    :ivar xz:          Access to 2D data along xz-plane.
    :ivar yz:          Access to 2D data along yz-plane.
    :ivar xyz:         Access to 3D data.

    """

    # Usually we think in terms of dimensions xyz, but it is much more
    # convenint to index them with numbers. This dictionary provides a way
    # to go from one notation to the other. Internally, we always use the
    # index notation.
    _dim_indices = {
        "x": (0,),
        "y": (1,),
        "z": (2,),
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
        "xyz": (0, 1, 2),
    }

    def __init__(self, sd):
        """Constructor.

        :param sd: Simulation directory.
        :type sd: :py:class:`~.SimDir`
        """

        if not isinstance(sd, simdir.SimDir):
            raise TypeError("Input is not SimDir")

        # _all_griddata is a dictionary that maps dimension to an object
        # AllGridFunctions, which contains all the variables for which that
        # dimension is available
        self._all_griddata = {
            dim: AllGridFunctions(sd.allfiles, dim)
            for dim in self._dim_indices.values()
        }

    def _string_or_tuple_to_dimension_index(self, dimension):
        """Internally, we always refer to the different dimensions with their
        numerical index. However, it is more convenient to have public
        interfaces with xyz. This method takes a dimension that can be either
        a string or a tuple and returns the corresponding dimension in the
        index notation.

        E.g.: 'x' -> (0, ), or (1, 2) -> (1, 2), or 'xy' -> (0, 1)

        :returns:  tuple of dimensions, e.g. (0,1) for xy-plane.
        or string with the name, e.g. 'xy'
        """
        # If the input is a recognized tuple, just return it
        if dimension in self._dim_indices.values():
            return dimension

        # If the input is a recognized string, return the corresponding tuple
        if dimension in self._dim_indices.keys():
            return self._dim_indices[dimension]

        raise ValueError(f"{dimension} is not a recognized dimension")

    def __getitem__(self, dimension):
        """Get data with given dimensionality.

        :param dimension:  tuple of dimensions, e.g. (0,1) for xy-plane.
        or string with the name, e.g. 'xy'
        """
        return self._all_griddata[
            self._string_or_tuple_to_dimension_index(dimension)
        ]

    def __getattr__(self, attr):
        # This allows to call self.x, self.xy and so on
        if attr in self._dim_indices.keys():
            # We retrieve the data with __getitem__
            return self[attr]

        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute {attr}"
        )

    def __contains__(self, dimension):
        return (
            self._string_or_tuple_to_dimension_index(dimension)
            in self._all_griddata
        )

    def __str__(self):
        """String representation"""
        return "\n".join([str(self[dim]) for dim in self._all_griddata])

    def total_filesize(self, unit="MB"):
        """Return the total size of the grid data files.
        Available units B, KB, MB and GB (in power of 1024 bytes).

        :param unit: Unit to use (in powers of 1024 bytes).
        :type unit: str among: ``B``, ``KB``, ``MB``, ``GB``.
        :returns: Total size of the grid data files.
        :rtype: float

        """
        # First we find all the unique files
        allfiles = set()
        for dim in self._all_griddata.keys():
            allfiles.update(self[dim].allfiles)
        return total_filesize(allfiles, unit=unit)
