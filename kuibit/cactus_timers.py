#!/usr/bin/env python3

# Copyright (C) 2022-2024 Gabriele Bozzola
#
# Inspired by code originally developed by Wolfgang Kastaun. This file may
# contain algorithms and/or structures first implemented in
# GitHub:wokast/PyCactus/PostCactus/cactus_timertree.py
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

"""The :py:mod:`~.cactus_timers` module provides an interface to Cactus timers.

Currently, the only timers supported are from the XML output by Carpet with the
option ``output_xml_timer_tree = yes``.

The main class defined in this module is :py:class:`~.TimersDir`. This is a
dictionary-like object that contains most of the information available in the
timers. The keys of this class are the various MPI ranks (ie, process numbers)
found in the output. The value are :py:class:`~.Tree` objects. These trees
contain the reported call-stack of the simulation with the associated timing
information. Simulation restarts are summed up. Often, it is useful to know how
much a function took on average across different MPI ranks. This can be accessed
with the :py:meth:`TimerDir.average` method.

Example of usage:

.. code-block:: python

   # Assuming sim is a SimDir
   avg_timers = sim.timers.average

   # avg_timers.name contains the name of the top level function, main
   # avg_timers.value contains the time took on average by such function

   # avg_timers.children contains a tuple with all the functions called by main.
   # The type of these functions is Tree, so they have a name and value attribute.

   # By default, the units are second. If you want to normalize to 1 and get the
   # percentual value, you can use
   normalized_avg_timers = avg_timers / avg_timers.tot_value_leaves

   print(normalized_avg_timers.value)   # Should be 1

   # Another useful feature is to represent the timer tree with a dictionary or a
   # json file, for postprocessing
   print(normalized_avg_timers.to_json())

"""

import os
import re
from collections.abc import KeysView
from statistics import mean, median
from typing import TYPE_CHECKING, Dict, List, Optional
from xml.etree.ElementTree import Element  # skipcq: BAN-B405
from xml.etree.ElementTree import parse as parse_xml  # skipcq: BAN-B405

from kuibit.tree import Tree, merge_trees

# We need this here to avoid circular imports (because SimDir imports
# cactus_timers)
if TYPE_CHECKING:  # pragma: no cover
    from kuibit.simdir import SimDir


class TimersDir:
    """Timer information read from Carpet.

    This is a dictionary-like object that has as keys the process number and as
    values the timer information represented as a :py:class:`~.Tree` object.

    Timers are summed up across different restarts.

    The most useful method in this class is :py:meth:`~.average`, which returns
    the average timing information across all the MPI ranks.

    Files are read only when needed. At the moment, only XML timers are
    supported, but different backends can be added.

    """

    @staticmethod
    def _load_xml(path: str) -> Tree:
        """Load an xml file to a Tree.

        The structure of the xml has to be more or less like:

        ..code-block:xml

          <timer name = "level1"> 1
            <timer name = "level2a"> 2 </timer>
            <timer name = "level2b"> 3.0 </timer>
            <timer name = "level2c"> 4.0
              <timer name = "level3"> 5 </timer>
            </timer>
          </timer>

        There can be more information, but it will be ignored. We only extract
        the timer tags, read the names and the associated values. When a value
        is not set, we assume it is zero.

        """

        def _traverse_recursive(node: Element) -> Tree:
            """Recursively go through the tree and transform XML to Tree.

            Take the current node, read the name, value, then apply this
            function recursively to the sub-nodes.
            """
            name = node.attrib["name"]
            # If the node does not have a time information attached to it, we just
            # add zero
            value = float(node.text.strip()) if node.text else 0.0
            children_nodes = node.findall("timer")
            children = tuple(map(_traverse_recursive, children_nodes))

            return Tree(name, value, children)

        root_node = parse_xml(path).getroot()  # skipcq: BAN-B314
        return _traverse_recursive(root_node)

    def __init__(self, sd: "SimDir"):
        """Constructor."""
        # Find all the timertree.PROC.xml files and organize them by process number

        # self.tree_files is a dictionary that has as keys the process number
        # and as values list containing all the files associated to that process
        self.tree_files: Dict[int, List[str]] = {}

        # self.timertrees maps process number with a Tree for that process
        # number
        self.timertrees: Dict[int, Tree] = {}

        # We match files with name timertree.NUMBER.xml
        # NUMBER is the process number
        timertree_xml_pattern = r"^timertree.([\d]+).xml$"

        timertree_xml_rx = re.compile(timertree_xml_pattern)

        for path in sd.allfiles:
            _folder, filename = os.path.split(path)
            filename_match = timertree_xml_rx.match(filename)
            if filename_match is not None:
                process_number = int(filename_match.group(1))
                self.tree_files.setdefault(process_number, []).append(path)

        # Let's sort the dictionary by key
        self.tree_files = dict(sorted(self.tree_files.items()))

        self.__average: Optional[Tree] = None
        self.__median: Optional[Tree] = None

    def __getitem__(self, process_number: int) -> Tree:
        """Read total timer information for a given process number.

        :param process_number: MPI rank.
        """
        if process_number not in self.timertrees:
            self.timertrees[process_number] = merge_trees(
                [
                    self._load_xml(path)
                    for path in self.tree_files[process_number]
                ]
            )

        return self.timertrees[process_number]

    def keys(self) -> KeysView:
        """Return the process numbers for which timer information is available."""
        # Process numbers available to read
        return self.tree_files.keys()

    @property
    def average(self) -> Tree:
        """Return the average timer information across all the processes."""
        if self.__average is None:
            # We call self[process_number] explicitely to ensure that everything
            # is loaded
            self.__average = merge_trees(
                [self[process_number] for process_number in self.keys()], mean
            )

        return self.__average

    @property
    def median(self) -> Tree:
        """Return the median timer information across all the processes."""
        if self.__median is None:
            # We call self[process_number] explicitely to ensure that everything
            # is loaded
            self.__median = merge_trees(
                [self[process_number] for process_number in self.keys()],
                median,
            )

        return self.__median

    def __str__(self) -> str:
        return f"Timers available for processes {list(self.keys())}"
