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

"""The :py:mod:`~.cactus_ah` module provides classes to access the information
about apparent horizons from various thorns. The main class is
:py:class:`~.cactus_horizons` which collects all available data from a
simulation.

"""

import os
import re
import warnings

import numpy as np

from postcactus.attr_dict import pythonize_name_dict
from postcactus.timeseries import TimeSeries, combine_ts


class OneHorizon:
    r"""This class represents properties of an apparent horizon
    computed from the quasi-isolated horizon formalism.

    The following variables are provided as :py:class:`~.TimeSeries`:

    :ivar formation_time: First time at which the horizon has been found
                          (as from AHFinderDirect)

    Those are taken from the QuasilocalMeasures thorn output, if available. For
    each timeseries a boolean variable with prefix "has\_" reports if the
    information is available or not. Variables with suffix "\_final" provide the
    values at the end of the simulation. Printing this class (or conversion to
    string) results in human readable summary.

    """

    def __init__(self, qlm_vars, ah_vars, shape_files):
        """Constructor. No need to use this class directly, create a
        :py:class:`~.HorizonsDir` instance instead to collect all BH
        info.


        """

        self._qlm_vars = qlm_vars
        self._ah_vars = ah_vars

        # We turn the var_dictionary into attributes
        for var, timeseries in self._qlm_vars.items():
            # With this we can access properties in the following way
            # horizon.mass
            setattr(self, var, timeseries)

        # Here we compute some interesting and useful quantities, if we have
        # qlm data
        if self._qlm_vars:
            self.mass_final = self.mass.y[-1]
            self.spin_final = self.spin.y[-1]
            self.dimensionless_spin_final = (
                self.spin_final / self.mass_final ** 2
            )
        else:
            self.mass_final = None
            self.spin_final = None
            self.dimensionless_spin_final = None

        # We put the AH vars under the ah attribute, they are accessed in the
        # same way as qlm_vars (as attribute). This is achieved using
        # pythonize_name_dict. The second argument is how we access the data.
        # Here we use the method get_ah_property that peeks into self._ah_vars.
        self.ah = pythonize_name_dict(ah_vars, self.get_ah_property)

        # We read the formation time from a variable in AH
        if self._ah_vars:
            self.formation_time = self.ah.area.tmin
        else:
            self.formation_time = None

        # Now we deal with the shape. Shape files is a dictionary that maps
        # iteration to the associated file
        self._shape_files = shape_files

        self.shape_available = self._shape_files != {}

        if self.shape_available:
            # We sort the iterations
            self.shape_iterations = np.array(
                sorted(s for s in self._shape_files)
            )

            self.shape_iteration_min = self.shape_iterations[0]
            self.shape_iteration_max = self.shape_iterations[-1]

            # To convert between time and iteration, we need the AH data If we
            # don't have that, we will assume time = iteration.
            #
            if self._ah_vars:
                # Now we find the associated times using ah.cctk_iteration
                # self.ah.cctk_iteration is a function time vs iteration, we
                # want the opposite. We define a new timeseries in which we swap
                # t and y
                times_iterations = TimeSeries(
                    self.ah.cctk_iteration.y, self.ah.cctk_iteration.t
                )
                self.shape_times = times_iterations(self.shape_iterations)
                self.shape_time_min = self.shape_times[0]
                self.shape_time_max = self.shape_times[-1]
            else:
                warnings.warn(
                    "AH data not found, so it is impossible to convert"
                    " between iteration number to time.\nManually set"
                    " shape_times or methods involving shape and time"
                    " will not work"
                )
                self.shape_times = None
                self.shape_time_min = None
                self.shape_time_max = None

            # We will save all the shape patches and their origin that we read in
            # this dictionary
            self._patches = {}

    def __getitem__(self, key):
        if key not in self._qlm_vars.keys():
            raise KeyError(f"Quantity {key} does not exist")

        return self._qlm_vars[key]

    def get_ah_property(self, key):
        """Return a property from AHFinderDirect as timeseries."""
        return self._ah_vars[key]

    def __str__(self):
        """Conversion to string.
        :returns: Human readable summary
        """
        ret = ""
        if self._ah_vars:
            ret += f"Formation time: {self.formation_time:.4f}\n"
        if self.shape_available:
            ret += "Shape available\n"
        if self._qlm_vars:
            ret += f"Final Mass = {self.mass_final:.3e}\n"
            ret += f"Final Angular Momentum = {self.spin_final:.3e}\n"
            ret += f"Final Dimensionless Spin = {self.dimensionless_spin_final:.3e}"
        return ret

    def ah_origin_at_iteration(self, iteration):
        """Return the AH origin at the given iteration"""
        return self._patches_at_iteration(iteration)[1]

    def _patches_at_iteration(self, iteration):
        """Return the shape of the horizon as a dictionary of patches with
        3D coordinates."""
        if not self.shape_available:
            raise ValueError("Shape information not available")
        if iteration not in self.shape_iterations:
            raise ValueError(
                f"Shape information for iteration {iteration} not available"
            )

        if iteration not in self._patches:
            self._patches[iteration] = self._load_patches(
                self._shape_files[iteration]
            )

        return self._patches[iteration]

    def shape_at_iteration(self, iteration):
        """Return the shape of the horizon as 3 arrays with the
        coordinates of the points.

        :returns:
        :rtype: three lists of 2D numpy arrays, one for each
                coordinate. The list is over the different patches
        """

        # TODO: Add an option to merge all the patches while keeping an order.

        patches, _ = self._patches_at_iteration(iteration)

        # Patches is dictionary, each patch is a list of three coordiantes for
        # all the points along the two angular directions.
        coord_x = [p[0] for p in patches.values()]
        coord_y = [p[1] for p in patches.values()]
        coord_z = [p[2] for p in patches.values()]

        return coord_x, coord_y, coord_z

    @staticmethod
    def _load_patches(path):
        """AHFinderDirect uses a system of multipatches to avoid coordinates
        singularities. Each patch covers a portion around an axis (e.g., +z
        axis). We will load this data.

        """
        # TODO: Add support for HDF5 files

        # ASCII files are structured in this way:
        # * There is a header with the number of patches and the origin
        # * For each patch, there is a section taht starts with
        #   ### +z patch and a local header
        # * Each section has multiple groups with one angular coordinate
        #   fixed, the various groups are separated by a blank line

        # Here we match the patch name. This matches
        # 1. the entire string ^ #
        # 2. the literals ### ..... patch
        # 3. and then we find what is inside +/- and one between xyz
        rx_patch = re.compile(r"^### ([+-][xyz]) patch$")

        # Here we find the origin of the system.
        # 1. We match the entire string (^ $)
        # 2. The match the literal # origin =
        # 3. We match numbers with possibly +/- and dots, and spaces in between
        #    ([\s])
        rx_origin = re.compile(
            r"^# origin = ([+-eE\d.]+)[\s]+([+-eE\d.]+)[\s]+([+-eE\d.]+)[\s]*$"
        )
        origin = None
        with open(path, "r") as fil:
            # patches is a dictionary that maps the name of the patch to the
            # coordiantes of that patch
            patches = {}
            # We use these variables as local variables
            current_patch = None
            current_patch_data = []
            current_coordinates = []
            # We scan the entire file line by line
            for line in fil:
                # We read the header, which starts with #
                if line.startswith("#"):
                    # If we haven't found the origin yet, let's look for it
                    if origin is None:
                        matched_origin = rx_origin.match(line)
                        if matched_origin is not None:
                            origin = np.array(
                                [
                                    float(matched_origin.group(coord))
                                    for coord in (1, 2, 3)
                                ]
                            )
                            # If we know that this the origin line, we can
                            # skip the rest of the analysis
                            continue

                    matched_patch = rx_patch.match(line)
                    if matched_patch is not None:
                        # If current_patch is not None it means that we have
                        # already read an header, so here we are moving to the
                        # next patch. We save all the data to the dictionary
                        # patches, and move reset current_patch_data to an empty
                        # list.
                        if current_patch is not None:
                            patches[current_patch] = current_patch_data
                            current_patch_data = []
                        current_patch = matched_patch.group(1)
                elif (line == "") or (line.isspace()):
                    # Here we found the blank line, which means that we have
                    # to flush the data and move on to a new group in the same
                    # patch.
                    # If current_coordinates is empty, the following will test
                    # negative. If it is not empty, it means that we have read
                    # the current group, so we can reset the current_coordiantes
                    # variable
                    if current_coordinates:
                        current_patch_data.append(current_coordinates)
                        current_coordinates = []
                else:
                    # Data line should have 6 columns, the last three are x, y,
                    # and z.
                    data = line.split()
                    if len(data) != 6:
                        raise RuntimeError(f"Corrupt AH shape file {path}")
                    coordinates = list(map(float, data[3:]))
                    # coordinates is a list like:
                    # [5.198917752 -0.1549284014 0.1549284014],
                    # so we append the list
                    current_coordinates.append(coordinates)

            # Here we take care of the last group
            if current_coordinates:
                current_patch_data.append(current_coordinates)
            if current_patch is not None:
                patches[current_patch] = current_patch_data

        if origin is None:
            raise RuntimeError("Corrupt AH files, missing origin.")
        # Reorganize the data from the AHHorizonDirect format to a numpy matrix
        # Each patch is an array with three lists, one for each direction
        patches = {
            p: np.transpose(np.array(d), axes=(2, 0, 1))
            for p, d in patches.items()
        }
        return patches, origin

    def shape_outline_at_iteration(self, iteration, cut):
        """Return the cut of the 3D shape on a specified plane.

        cut has to be a 3D tuple or list with None on the dimensions
        you want to keep, and the value of the other coordinates. For
        example, if you want the outline at z = 3 on the xy plane,
        cut has to be (None, None, 3).

        No interpolation is performed, so results are not accurate when the cut
        is not along one of the major directions centered in the origin of the
        horizon.

        :returns:    Coordinates of AH outline.
        :rtype:      tuple of two 1D numpy arrays.

        """
        # TODO: Add interpolation for off-axis cuts

        if iteration not in self.shape_iterations:
            raise ValueError(f"Shape not available for iteration {iteration}")

        if not isinstance(cut, (tuple, list)):
            raise TypeError(
                f"Cut has to be a list or a tuple, not a {type(cut)}"
            )

        if len(cut) != 3:
            raise ValueError("Cut has to be three-dimensional")

        patches, origin = self._patches_at_iteration(iteration)

        # If cut is three None, we return the entire shape.
        # TODO: Merge patches here
        if cut.count(None) == 3:  # all None
            return self.shape_at_iteration(iteration)

        # There is at least one that is not None

        # If they are all not None, that's a point, so we cannot return anything
        # meaningful
        if cut.count(None) == 0:
            raise ValueError("Cut must has some entries set to None")

        # There is at least one that is None, but not all
        #
        # This can be 1D or 2D.

        # On what dimension(s) do we have to look at?
        dims = [index for index, value in enumerate(cut) if value is None]

        # Let's first consider the 1D case
        if cut.count(None) == 1:
            # Here we just have to find those points with the specified
            # coordinates

            # Here dim is the axis we are looking at, dim0 and dim1 the other
            # two
            dim = dims[0]
            (dim0, dim1) = [index for index in range(3) if index not in dims]

            # Here we collect the result
            points = []

            # We closest points up to tolerance of 0.1%, unless the cut is
            # outside the horizon
            for patch in patches.values():
                coords, coords0, coords1 = patch[dim], patch[dim0], patch[dim1]
                # We check we are inside the horizon
                if not (
                    np.min(coords0) <= cut[dim0] <= np.max(coords0)
                    and np.min(coords1) <= cut[dim1] <= np.max(coords1)
                ):
                    continue

                size0 = np.max(coords0) - np.min(coords0)
                size1 = np.max(coords1) - np.min(coords1)
                points_around_cut0 = abs(coords0 - cut[dim0]) < 1e-3 * size0
                points_around_cut1 = abs(coords1 - cut[dim1]) < 1e-3 * size1

                selected_points = np.logical_and(
                    points_around_cut0, points_around_cut1
                )
                points.append(coords[selected_points])

            return points

        # Finally the 2D case
        #
        # The catch here is that we want to merge the patches and make sure
        # that they are in order.
        if cut.count(None) == 2:
            (dim0, dim1) = dims
            # dim is the normal direction to the plane where we want the outline
            dim = [index for index in range(3) if index not in dims][0]
            # In points0 and points1 we collect the points on the dimensions 0
            # an 1, below we also use 0 and 1 to refer to these two dimensions.
            points0, points1 = [], []

            # We closest points up to tolerance of 0.1%, unless the cut is
            # outside the horizon
            for patch in patches.values():
                coords, coords0, coords1 = patch[dim], patch[dim0], patch[dim1]
                if not (np.min(coords) <= cut[dim] <= np.max(coords)):
                    continue
                size = np.max(coords) - np.min(coords)
                points_around_cut = abs(coords - cut[dim]) < 1e-3 * size
                points0.append(coords0[points_around_cut])
                points1.append(coords1[points_around_cut])

            if len(points0) == 0:
                return None

            points0 = np.hstack(points0)
            points1 = np.hstack(points1)
            # We compute angle to properly order the points
            phi = np.angle(
                points0 - origin[dim0] + 1j * (points1 - origin[dim1])
            )
            ordering = np.argsort(phi)
            return points0[ordering], points1[ordering]


class HorizonsDir:
    """Class to collect information on apparent horizons
    available from thorns AHFinderDirect and QuasiLocalMeasures.

    :ivar found_any:   True if at least one horizon was found.

    Iterating over HorizonsDir objects means iterating over
    the horizons. The length of this object is the number of
    horizons for which there's QLM and AH data (if there's any
    QLM data, otherwise is the number of AH horizons)

    AHFinderDirect and QuasiLocalMeasures have different indexing. You must
    provide both when accessing a file. In the future, the map between the
    two indexing systems will be inferred from the paramter file.

    """

    # What variables should not be passed to OneHorizon?
    _exclude_ah_vars = ["cctk_time"]

    def __init__(self, sd):
        """Constructor.

        :param sd:  SimDir object providing access to data directory.
        :type sd:   SimDir

        """
        # We organize the variables from QuasiLocalMeasures in the dictionary
        # _qlm_vars. This dictionary has as keys the number of qlm horizon and
        # as value another dictionary with as keys the name of the variables
        # stripped of the qlm_ prefix and of the number, and as values the
        # timeseries. We extract the number with a regular expression which
        # matches qlm_VARNAME[NUM]

        self._qlm_vars = {}
        self._populate_qlm_vars(sd)
        self._num_qlm_horizons = len(self._qlm_vars.keys())

        # ah_vars is a dictionary like qlm_vars with the difference that we
        # extract information from the files BH_diagnostics.ah(\d+).gp and
        # that the index here is the Apparent Horizon index.
        self._ah_vars = {}
        # self._num_ah_horizons is set inside the function
        self._populate_ah_vars(sd)

        # The next step is to find the files for the shape of the horizons, if
        # available. We scan all the files and find those with h.t*****.ah*.gp
        #
        # Once again we put all the files in a dictionary with index the AH
        # index and as values another dictionary with keys the iteration and
        # value the file
        self._shape_files = {}
        self._populate_shape_files(sd)

        # Here we align the ah_vars and shape_files so that they have the same
        # keys. We add an empty {} to the missing values.
        for ah_index in self._ah_vars:
            if ah_index not in self._shape_files:
                self._shape_files[ah_index] = {}

        for ah_index in self._shape_files:
            if ah_index not in self._ah_vars:
                self._ah_vars[ah_index] = {}

        # Okay, here we can be confident that shape_files and ah_vars have the
        # same keys.

        # Now there are multiple options:
        # 1. We don't have any qlm_vars, ah_vars, and _shape_files
        # 2. We qlm_vars, ah_vars, and shape_files
        # 3. We have a combination of the above
        #
        # The OneHorizon class is robust in taking partially empty data,
        # but we must clear the indexing system.
        #
        # Here we check that all the dictionaries are not empty
        self.found_any = any(
            [self._qlm_vars, self._ah_vars, self._shape_files]
        )

    def _populate_qlm_vars(self, sd):
        # Here we a regular expression with two capturing groups
        # 1. ^ $ means that we match the entire string
        # 2. qlm_ is matched to itself
        # 3. the first capturing group, (\w+), matches any word
        # 4. Then we match the brackets, and inside a number
        rx_qlm_number = re.compile(r"^qlm_(\w+)\[(\d+)\]$")

        for var_name in sd.ts.scalar.keys():
            matched = rx_qlm_number.match(var_name)
            if matched is not None:
                # Here we strip of qlm_ and of the number
                var_name_stripped = matched.group(1)
                horizon_number = int(matched.group(2))
                # For each horizon, we have a dictionary that maps variable
                # names to the timeseries
                horizon_vars = self._qlm_vars.setdefault(horizon_number, {})
                horizon_vars[var_name_stripped] = sd.ts.scalar[var_name]

    def _populate_ah_vars(self, sd):
        # First, we find all the files related to apparent horizons. These
        # have names like BH_diagnostics.ah1.gp
        self._ah_files = {}

        rx_ah_filename = re.compile(r"^BH_diagnostics.ah(\d+).gp$")
        for path in sd.allfiles:
            filename = os.path.split(path)[-1]
            matched = rx_ah_filename.search(filename)
            if matched is not None:
                ah_index = int(matched.group(1))
                self._ah_files.setdefault(ah_index, []).append(path)

        # Next, we find what variables they contain. This should be pretty
        # standard, but we can make our code more robust by not assuming too
        # much. We read one header and find the variables, then read all the
        # other files assuming the have the same variables. A complication is
        # that variables names have blank spaces. We turn them into underscores.

        self._num_ah_horizons = len(self._ah_files.keys())

        # We continue only if we find some files
        if self._num_ah_horizons > 0:

            first_ah_file = next(iter(self._ah_files.values()))[0]
            with open(first_ah_file, "r") as fil:
                # Here we read the first lines_to_read into header
                # We strip the new line
                header = []
                for line in fil:
                    # We read the header, which starts with #
                    if line.startswith("#"):
                        header.append(line.strip())
                    else:
                        break

            # Now, we parse the header and associate variable name with column
            # where the data is. The header looks like:
            #
            # # apparent horizon 1/3
            # #
            # # column  1 = cctk_iteration
            # # column  2 = cctk_time
            # # column  3 = centroid_x
            # # column  4 = centroid_y
            # # column  5 = centroid_z
            #
            # We scan the columns with a regex.
            # 1. ^ $ means that we match the entire string
            # 2. \#[\s]column[\s]+ matches the literal '# column ' with any
            #    number of spaces.
            # 3. Then we match the number
            # 4. We match another literal ' = ' with the sapces
            # 5. Finally we match the name of the variable matching letters
            #    and symbols
            rx_column = re.compile(
                r"\#[\s]column[\s]+(\d+)[\s]=[\s]([a-zA-Z_0-9\s()-/]+)$"
            )

            # Here is where we store the map
            self._ah_vars_columns = {}

            for line in header:
                matched = rx_column.match(line)
                if matched is not None:
                    # Columns counting start from 1, so we must subtract one to
                    # be with 0-based indexing
                    column_number = int(matched.group(1)) - 1
                    name = matched.group(2)

                    # We need to know where the time is
                    if name == "cctk_time":
                        time_column = column_number

                    # We exclude some variables we don't want in OneHorizon
                    # (e.g., we don't want cctk_time)
                    if name in self._exclude_ah_vars:
                        continue

                    # Spaces to underscores
                    name = name.replace(" ", "_")
                    # We remove parentheses
                    name = name.replace("(", "")
                    name = name.replace(")", "")
                    # We change / to -
                    name = name.replace("/", "-")
                    self._ah_vars_columns[name] = column_number

            # Now we are ready to populate, we read all the data first. Then, we
            # select all the columns
            for ah_index, files in self._ah_files.items():
                # We create an empty dictionary in self._ah_vars[ah_index]
                self._ah_vars.setdefault(ah_index, {})

                # We read all the data
                alldata = [np.loadtxt(f, unpack=True, ndmin=2) for f in files]
                for var_name, column_number in self._ah_vars_columns.items():
                    # Here we select the time column and the data column for all
                    # the data in each file and we convert them into TimeSeries
                    data_ts = combine_ts(
                        [
                            TimeSeries(data[time_column], data[column_number])
                            for data in alldata
                        ]
                    )
                    self._ah_vars[ah_index][var_name] = data_ts

    def _populate_shape_files(self, sd):
        # Here we match the files with a regular expression:
        # 1. ^ $ means that we match the entire string
        # 2. Then we match the literal h.t
        # 3. with a number (\d+)
        # 4. the literal .ah
        # 5. another number (\d+)
        # 6. and the file extension .gp
        rx_shape_filename = re.compile(r"^h.t(\d+).ah(\d+).gp$")
        for path in sd.allfiles:
            filename = os.path.split(path)[1]
            matched = rx_shape_filename.match(filename)
            if matched is not None:
                ah_index = int(matched.group(2))
                iteration = int(matched.group(1))
                ah_shape_dict = self._shape_files.setdefault(ah_index, {})
                ah_shape_dict[iteration] = path

    @property
    def available_qlm_horizons(self):
        """Horizons in QLM indexing with associated data"""
        return sorted(list(self._qlm_vars.keys()))

    @property
    def available_apparent_horizons(self):
        """Horizons in AH indexing with associated data"""
        return sorted(list(self._ah_vars.keys()))

    def __getitem__(self, key):
        if not isinstance(key, (tuple, list)):
            raise TypeError("You have to provide both the QLM and AH indices")

        if len(key) != 2:
            raise KeyError(f"{key} does not identify an horizon")

        # We ensure that we have ints because we store the keys as int.
        qlm_index, ah_index = int(key[0]), int(key[1])

        # We need at least of the two indices to be valid
        if (
            qlm_index not in self.available_qlm_horizons
            and ah_index not in self.available_apparent_horizons
        ):
            raise KeyError(f"Horizon {key} in not found")

        # With get we return an empty dictionary if we don't have the index
        return OneHorizon(
            self._qlm_vars.get(qlm_index, {}),
            self._ah_vars.get(ah_index, {}),
            self._shape_files.get(ah_index, {}),
        )

    def __str__(self):
        if not self.found_any:
            return "No horizon found"

        num_qlm = len(self.available_qlm_horizons)
        num_ah = len(self.available_apparent_horizons)

        ret = "Horizons found:\n"
        ret += f"{num_qlm} horizons from QuasiLocalMeasures\n"
        ret += f"{num_ah} horizons from AHFinderDirect"
        return ret

    # TODO: Implement the following functions, when we can infer the mapping
    #       form the paramter file

    # def __iter__(self):
    #     for horizon_number in self.available_horizons:
    #         yield self[horizon_number]

    # def __len__(self):
    #     return len(self.available_horizons)
