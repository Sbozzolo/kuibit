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

"""The :py:mod:`~.cactus_grid` module provides functions to load
grid function in Cactus formats.
"""
# import os
# import re
# import warnings
# from contextlib import contextmanager
# from functools import lru_cache

# import h5py
# import numpy as np

from postcactus import simdir
# from postcactus import grid_data
# from postcactus.attr_dict import pythonize_name_dict


# class OneGridFunctionH5:
#     """Read grid data produced by CarpetASCII or CarpetHDF5."""

#     # TODO: Allow for one file per group

#     # Let's unpack the regex, we have 7 capturing groups
#     #
#     # 1. [^:] matches any number of characters (at least one), with the
#     #    exception of ':'
#     # 2. \S+ matches any number of non-whitespace characters (at least
#     #    one).
#     #
#     # 1. and 2. are thorn and variable name, then we have
#     # 3, 4. \d+ matches a number (iteration and time level)
#     # 5. " m=0" matches this exactly, if it is present
#     # 6. \d+ matches a number (refinement level)
#     # Then we have two nested capturing groups
#     # ( c=(\d+))? checked whether c=NUMBER is matched,
#     # and inside we have that the component number is matched
#     #
#     # All the [ ] are meaningful blank spaces. Note, we don't have $^ because
#     # there is more that we do not want to match
#     _pattern_group_name = r"""
#     ([^:]+)             # Thorn name
#     ::
#     (\S+)               # Variable name
#     [ ]
#     it=(\d+)            # Iteration
#     [ ]
#     tl=(\d+)            # Timelevel (almost always 0)
#     ([ ]m=0)?           # Map
#     [ ]
#     rl=(\d+)            # Refinement level
#     ([ ]c=(\d+))?       # Component
#     """

#     def __init__(self, allfiles):
#         self.allfiles = allfiles

#         # self.toc is a nested dictionary
#         # 1. At the first level, we have the file
#         # 2. self.toc[filename] is a dictionary with keys the various
#         #    available iterations in filename
#         # 3. self.toc[filename][iteration] is another dictionary with keys the
#         #    various refinement levels available in filename at the iteration
#         #    and values a list of available components
#         self.toc = {}

#         self.thorn_name = None
#         self.var_name = None
#         self.map_ = None

#         for path in self.allfiles:
#             self._parse_group_names_from_file(path)

#         if self.map_ is None:
#             self.map_ = ""

#         self.dataset_format = (
#             f"{self.thorn_name}::{self.var_name} it=%d tl=0{self.map_} rl=%d%s"
#         )

#         # HDF5 files can contain ghostzones or not. Here, we can that all the
#         # files have the same behavior (they all contain, or they all don't)
#         ghost_in_files = {
#             self._are_ghostzones_in_file(path) for path in self.allfiles
#         }

#         if len(ghost_in_files) != 1:
#             raise ValueError(
#                 "Inconsistent IOHDF5::output_ghost_points across files"
#             )

#         self.are_ghostzones_in_files = list(ghost_in_files)[0]

#     def _parse_group_names_from_file(self, path):
#         # This will give us an overview of what is available in the provided
#         # file. We keep a collection of all these in the variable self.toc
#         rx_group_name = re.compile(self._pattern_group_name, re.VERBOSE)
#         with h5py.File(path, "r") as f:
#             for group in f.keys():
#                 matched = rx_group_name.match(group)
#                 # If this is not an interesting group, just skip it
#                 if not matched:
#                     continue

#                 time_level = int(matched.group(4))
#                 # TODO: Why is this?
#                 if time_level != 0:
#                     continue

#                 thorn_name = matched.group(1)
#                 var_name = matched.group(2)

#                 iteration = int(matched.group(3))
#                 ref_level = int(matched.group(6))

#                 component = (
#                     -1 if matched.group(8) is None else int(matched.group(8))
#                 )

#                 toc_file = self.toc.setdefault(path, {})
#                 toc_iteration = toc_file.setdefault(iteration, {})
#                 toc_ref_level = toc_iteration.setdefault(ref_level, [])
#                 toc_ref_level.append(component)

#             self.thorn_name = thorn_name
#             self.var_name = var_name

#     @lru_cache(128)
#     def _iterations_in_file(self, path):
#         """Return the (sorted) available iterations in file path.

#         Use this if you need to ensure that you are looping over iterations
#         in order!
#         """
#         return sorted(self.toc[path].keys())

#     def _min_iteration_in_file(self, path):
#         """Return the minimum available iterations in file path."""
#         return self._iterations_in_file(path)[0]

#     def _max_iteration_in_file(self, path):
#         """Return the maximum available iterations in file path."""
#         return self._iterations_in_file(path)[-1]

#     @property
#     def min_iteration(self):
#         """Return the minimum available iteration."""
#         # restarts is a ordered list of tuples with three elements:
#         # (iteration_min, iteration_max, path)
#         restarts = self.get_restarts()
#         # The minimum iteration is in restarts[0][0]
#         return restarts[0][0]

#     @property
#     def max_iteration(self):
#         """Return the maximum available iteration."""
#         # restarts is a ordered list of tuples with three elements:
#         # (iteration_min, iteration_max, path)
#         restarts = self.get_restarts()
#         # The maximum iteration is in restarts[-1][1]
#         return restarts[-1][1]

#     @property
#     @lru_cache(128)
#     def available_iterations(self):
#         """Return the available iterations."""
#         iteration_in_files = [
#             {it for it in self._iterations_in_file(path)}
#             for path in self.allfiles
#         ]
#         # Next we merge everything to make a set and we sort it
#         return sorted(set().union(*iteration_in_files))

#     @property
#     @lru_cache(128)
#     def available_times(self):
#         """Return the available times."""
#         return [
#             self.time_at_iteration(iteration)
#             for iteration in self.available_iterations
#         ]

#     def _ref_levels_in_file(self, path, iteration):
#         """Return the available refinement levels in file path at the
#         specified iteration
#         """
#         # In case we don't have the iteration, we want to return the empty
#         # list, so we use the .get dictionary method. This return the
#         # value, if the key is available, otherwise it returns the second
#         # argument. Here, we fix as second argument the empty dictionary, so
#         # when we take .keys() we get the empty list
#         return list(self.toc[path].get(iteration, {}).keys())

#     def _components_in_file(self, path, iteration, ref_level):
#         """Return the available components  in file path at the
#         specified iteration and refinement level
#         """
#         # Same comment as _ref_levels_in_file, but with an
#         # additional level
#         return self.toc[path].get(iteration, {}).get(ref_level, [])

#     def _grid_information_in_dataset(self, dataset):

#         # TODO: Why do we need to reverse the array?
#         shape = np.array(dataset.shape[::-1])

#         x0 = dataset.attrs["origin"]
#         dx = dataset.attrs["delta"]
#         time = dataset.attrs["time"]
#         # With the .get we ensure that if "cctk_nghostzones" cannot be read, we
#         # have returned None, which we can test later
#         num_ghost = dataset.attrs.get("cctk_nghostzones", None)
#         # If we do not have the ghostzones in the file, then it is as if we
#         # have ghostzones of size zero.
#         if not self.are_ghostzones_in_files or num_ghost is None:
#             num_ghost = np.zeros_like(shape, dtype=int)

#         return shape, x0, dx, num_ghost, time

#     def _grid_from_dataset(self, dataset, iteration, ref_level, component):
#         shape, x0, dx, num_ghost, time = self._grid_information_in_dataset(
#             dataset
#         )

#         return grid_data.UniformGrid(
#             shape,
#             x0=x0,
#             dx=dx,
#             ref_level=ref_level,
#             num_ghost=num_ghost,
#             time=time,
#             iteration=iteration,
#         )

#     # What is a context manager?
#     #
#     # Context managers are useful ways to handle resources in Python. With a
#     # context manager, we do not have to worry about releasing resources. Here,
#     # we wrap reading the h5 file with another context manager so that we can
#     # easily get the dataset.
#     @contextmanager
#     def _get_dataset(self, path, iteration, ref_level, component):
#         component_str = f" c={component}" if (component >= 0) else ""
#         with h5py.File(path, "r") as f:
#             try:
#                 yield f[
#                     self.dataset_format % (iteration, ref_level, component_str)
#                 ]
#             finally:
#                 # All the hard work is done by the other 'with' statement
#                 pass

#     def _read_dataset_in_file_as_uniform_grid_data(
#         self, path, iteration, ref_level, component
#     ):
#         """"""
#         with self._get_dataset(
#             path, iteration, ref_level, component
#         ) as dataset:
#             grid = self._grid_from_dataset(
#                 dataset, iteration, ref_level, component
#             )

#             # TODO: Why do we need to reverse the array?
#             data = np.transpose(dataset[()])

#         return grid_data.UniformGridData(grid, data)

#     def _are_ghostzones_in_file(self, path):
#         """"""
#         # This is a tricky and important function to stitch together all the
#         # different components. Carpet has an option (technically two) to output
#         # the ghostzones in the files. These are: output_ghost_points and
#         # out3D_ghosts (which is DEPRECATED). When they are both set to yes,
#         # the ghostzones are output in the h5 files. When one of the two is set
#         # to no, the ghostzones are not output.

#         # The default value of these parameters is yes
#         with h5py.File(path, "r") as f:
#             parameters = f["Parameters and Global Attributes"]
#             all_pars = parameters["All Parameters"][()].decode().split("\n")
#             # We make sure that everything is lowercase, we are case insensitive
#             iohdf5_pars = [
#                 param.lower()
#                 for param in all_pars
#                 if param.startswith("CarpetIOHDF5")
#                 or param.startswith("IOHDF5")
#             ]

#             def is_param_true(name):
#                 param = [p for p in iohdf5_pars if name.lower() in p]
#                 # When the parameters are not set, they are set to yes by
#                 # default
#                 if len(param) == 0:
#                     return True

#                 # The parameter is set
#                 return (
#                     ("true" in param[0])
#                     or ("yes" in param[0])
#                     or ("1" in param[0])
#                 )

#             return is_param_true("out3D_ghosts") and is_param_true(
#                 "output_ghost_points"
#             )

#     def _files_with_iteration(self, iteration):
#         # Using self.get_restarts(), find the file that the given iteration
#         # between iteration_min and iteration_max

#         if (iteration < self.min_iteration) or (
#             iteration > self.max_iteration
#         ):
#             raise ValueError(f"Iteration {iteration} not in available range")

#         # restarts is a ordered list of tuples with three elements:
#         # (iteration_min, iteration_max, path)
#         max_iterations_in_files = np.array([i for _, i, _ in self.restarts])
#         # This returns the index of the first element in max_iterations_in_files
#         # that is greater than the given iteration.
#         # E.g. max_iterations_in_files = [1, 5, 10] and iteration = 6, since
#         # max_iterations_in_files are the maximum iterations in the various files,
#         # the function has to return 2, which is the index of the first element
#         # in max_iterations_in_files that is larger than 6. This is what
#         # np.searchsorted does.
#         index = np.searchsorted(max_iterations_in_files, iteration)

#         # The second element is the list of path
#         return self.restarts[index][2]

#     def time_at_iteration(self, iteration):
#         """Return the time corresponding to the provided iteration"""
#         # If there are multiple files, we take the first
#         path = self._files_with_iteration(iteration)[0]

#         ref_levels = self._ref_levels_in_file(path, iteration)
#         if ref_levels:
#             components = self._components_in_file(
#                 path, iteration, ref_levels[-1]
#             )
#             if components:
#                 with self._get_dataset(
#                     path, iteration, ref_levels[-1], components[-1]
#                 ) as dataset:
#                     return dataset.attrs["time"]

#         return None

#     def dx_at_iteration_ref_level(self, iteration, ref_level):
#         """Return the time corresponding to the provided iteration"""
#         # If there are multiple files, we take the first
#         path = self._files_with_iteration(iteration)[0]

#         components = self._components_in_file(path, iteration, ref_level)
#         if components:
#             with self._get_dataset(
#                 path, iteration, ref_level, components[-1]
#             ) as dataset:
#                 return dataset.attrs["delta"]

#         return None

#     @lru_cache(128)
#     def get_restarts(self):
#         """Return a list of tuples (iteration_min, iteration_max, paths)
#         with the minimum iterations and maximum iterations in each file
#         associated to this variable. (Restarts and checkpoints)
#         paths is a list of all the files with same min_iteration and
#         max_iteration (as when we have 3d xyz_file files)
#         """
#         restarts = [
#             (
#                 self._min_iteration_in_file(path),
#                 self._max_iteration_in_file(path),
#                 [path],
#             )
#             for path in self.allfiles
#         ]
#         # We sort by first the minimum iteration. If the minimum iteration is
#         # the same, we sort by the -maximum iteration, which ensures that where
#         # we have more iterations is placed before. Consider the example of
#         # list to be sorted [(1, 2), (0, 3), (1, 5)], this would be sorted in
#         # ascending order with respect as if it was [(1, -2), (0, -3), (1, -5)],
#         # so [(0, 3), (1, 5), (1, 2)]
#         restarts.sort(key=lambda x: (x[0], -x[1]))

#         # Next, we check that there is no overlap. If there's overlap, we
#         # ignore some folders, unless min_iteration and max_iteration are
#         # exactly the same, in which case we combine the two
#         first, *others = restarts
#         # We assemble a return list, starting with the first element. Then,
#         # we loop over the other elements and we keep expanding the return
#         # list only if we find that the maximum iteration is larger than
#         # the previous maximum iteration. Since we sorted as described above,
#         # this will ignore repeated runs in which the iterations are a subset
#         # of the previous one.
#         ret = [first]
#         for min_iteration, max_iteration, path in others:
#             # If we have exactly the same iterations, we are looking at ones
#             # of those 3D files, so we collect them
#             if (min_iteration, max_iteration) == (ret[-1][0], ret[-1][1]):
#                 ret[-1][2].append(path)
#                 break

#             max_iteration_in_ret = ret[-1][1]
#             if max_iteration > max_iteration_in_ret:
#                 ret.append((min_iteration, max_iteration, path))
#             else:
#                 warnings.warn(f"Unused (redundant) file: {path}")

#         return ret

#     @property
#     def restarts(self):
#         return self.get_restarts()

#     def total_filesize(self, unit="MB"):
#         """Return the total size of the files with this variable.
#         Available units B, KB, MB and GB
#         """
#         units = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
#         if unit not in units.keys():
#             raise ValueError(
#                 f"Invalid unit: expected one of {list(units.keys())}"
#             )
#         return (
#             sum({os.path.getsize(path) for path in self.allfiles})
#             / units[unit]
#         )

#     @lru_cache(128)
#     def _read_raw(self, iteration):

#         uniform_grid_data_components = []

#         for path in self.allfiles:
#             for ref_level in self._ref_levels_in_file(path, iteration):
#                 for component in self._components_in_file(
#                     path, iteration, ref_level
#                 ):
#                     unif_grid_data = (
#                         self._read_dataset_in_file_as_uniform_grid_data(
#                             path, iteration, ref_level, component
#                         )
#                     )
#                     if unif_grid_data is None:
#                         continue
#                     uniform_grid_data_components.append(unif_grid_data)

#         return (
#             # grid_data.HierachicalGridData(uniform_grid_data_components)
#             uniform_grid_data_components
#             if uniform_grid_data_components
#             else None
#         )


# class AllGridFunctions:
#     """Helper class to read various types of grid data in a list of files and
#     properly order them. The core of this object is the _vars dictionary which
#     contains the location of all the files for a specific variable and
#     reduction.

#     AllGridFunction is a dictionary-like object.

#     Using the [] notation you can access values with as HierarchicalGridFunction.

#     Not intended for direct use.

#     """

#     # Different "cuts" have different extensions in their filenames, here we
#     # save all the possible ones. In the instance of the class, we fix the
#     # specific pattern corresponding to the dimension (which are the keys on
#     # the following dictionary). In general, the file name will be:
#     # variable-name.ext.h5, eg rho.xy.h5.
#     filename_extensions = {
#         (0,): ".x",
#         (1,): ".y",
#         (2,): ".z",
#         (0, 1): "(.0)?.xy",
#         (0, 2): "(.0)?.xz",
#         (1, 2): "(.0)?.yz",
#         (0, 1, 2): r"(.xyz)?(.file_[\d]+)?(.xyz)?",
#     }

#     def __init__(self, allfiles, dimension):
#         """allfiles is a list of files, dimension has to a tuple"""

#         # Here we save what kind of file we are looking at
#         # We assume that dimension is already sanitized (that is, is in tuple
#         # form and not in string form)
#         self.dimension = dimension

#         # This is a simple regex:
#         # 1. ^ and $ mean that we have to match the entire string
#         # 2. ([a-zA-Z0-9\[\]_]+) means that we match any character any number
#         #    of times, this is a capturing group, and is the variable name.
#         # 3. We have the extension, which identifies the dimension and is
#         #    saved in the class instance
#         # 4. Finally, we have the filename extension which can be either h5
#         #    or txt
#         filename_pattern = r"^([a-zA-Z0-9\[\]_]+)%s.%s$"
#         h5_pattern = filename_pattern % (
#             self.filename_extensions[self.dimension],
#             "h5",
#         )
#         ascii_pattern = filename_pattern % (
#             self.filename_extensions[self.dimension],
#             "txt",
#         )

#         # Variable files is a dictionary, the keys are the variables, the
#         # values a list of files associated to that variable
#         self._vars_ascii = {}
#         self._vars_h5 = {}

#         rx_h5 = re.compile(h5_pattern)
#         rx_ascii = re.compile(ascii_pattern)

#         for f in allfiles:
#             filename = os.path.split(f)[1]
#             matched_h5 = rx_h5.match(filename)
#             matched_ascii = rx_ascii.match(filename)
#             # If matched_pattern is not None, this is a Carpet h5 file
#             if matched_h5 is not None:
#                 # We normalize the names taking everything to be lowercase
#                 variable_name = matched_h5.group(1).lower()
#                 var_list = self._vars_h5.setdefault(variable_name, [])
#                 var_list.append(f)
#             elif matched_ascii is not None:
#                 variable_name = matched_ascii.group(1).lower()
#                 var_list = self._vars_ascii.setdefault(variable_name, [])
#                 var_list.append(f)

#         self._vars = {}
#         # What pythonize_name_dict does is to make the various variables
#         # accessible as attributes, e.g. self.fields.rho
#         self.fields = pythonize_name_dict(list(self.keys()), self.__getitem__)

#     @lru_cache(128)
#     def __getitem__(self, key):
#         k = str(key).lower()
#         # We prefer h5
#         if k in self._vars_h5:
#             # return self._load_from_h5files(self._vars_h5[k])
#             return OneGridFunctionH5(self._vars_h5[k])

#         if k in self._vars_ascii:
#             return self._load_from_textfiles(self._vars_ascii[k])

#         raise KeyError

#     def _load_from_h5files(self, h5files):
#         pass

#     def _load_from_textfiles():
#         pass

#     def get(self, key, default=None):
#         if key not in self:
#             return default
#         return self[key]

#     def keys(self):
#         # To find the unique keys we use transofrm the keys in sets, and then
#         # we unite them.
#         # TODO: In Python3, keys() should not be a list!
#         return list(set(self._vars_h5.keys()).union(set(self._vars_h5.keys())))

#     def __str__(self):
#         ret = f"\nAvailable grid data of dimension {self.dimension}: "
#         ret += f"\n{list(self.keys())}\n"
#         return ret


class GridFunctionsDir:
    """This class provides access to all grid data.

    This includes 1D-3D data in hdf5 format as well as 1D ASCII
    data. Data of the required dimensionality is read from any format
    available (hdf5 preferred over ascii). If necessary, cuts are applied
    to 2D/3D data to get requested 1D/2D data.

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

        if not isinstance(sd, simdir.SimDir):
            raise TypeError("Input is not SimDir")

        # self._all_griddata = {
        #     dim: AllGridFunctions(sd.allfiles, dim)
        #     for dim in self._dim_indices.values()
        # }

        # self.hdf5 = cgr.GridH5Dir(sd)
        # self.ascii = cgra.GridASCIIDir(sd)
        # rdr = [self.hdf5, self.ascii]
        # self._dims = {d: GridOmniReader(d, rdr) for d in self._alldims}

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

    # def __getattr__(self, attr):
    #     # This allows to call self.x, self.xy and so on
    #     if attr in self._dim_indices.keys():
    #         # We retrieve the data with __getitem__
    #         return self[attr]

    #     raise AttributeError(
    #         f"{self.__class__.__name__} object has no attribute {attr}"
    #     )

    # def __getitem__(self, dimension):
    #     """Get data with given dimensionality

    #     :param dimension:  tuple of dimensions, e.g. (0,1) for xy-plane.
    #     or string with the name, e.g. 'xy'
    #     """
    #     return self._all_griddata[
    #         self._string_or_tuple_to_dimension_index(dimension)
    #     ]

    # def __contains__(self, dimension):
    #     return (
    #         self._string_or_tuple_to_dimension_index(dimension)
    #         in self._all_griddata
    #     )

    # def __str__(self):
    #     """String representation"""
    #     return "\n".join([str(self[d]) for d in self._all_griddata])

    # def filesize(self):
    #     sizes = {d: self[d].filesize() for d in self._alldims}
    #     total = sum((s[0] for s in list(sizes.values())))
    #     return total, sizes
