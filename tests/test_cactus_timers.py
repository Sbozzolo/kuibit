#!/usr/bin/env python3

# Copyright (C) 2022 Gabriele Bozzola
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

import os
import unittest
from statistics import mean, median

from kuibit import tree as ktree
from kuibit.simdir import SimDir


class TestCactusTimers(unittest.TestCase):
    def setUp(self):
        self.sd = SimDir("tests/tov")
        self.timers = self.sd.timers

    def test_load_xml(self):
        # We create a simple xml file and we parse it
        xml = """\
<timer name = "level1"> 1
  <timer name = "level2a"> 2 </timer>
  <timer name = "level2b"> 3.0 </timer>
  <timer name = "level2c"> 4.0
    <timer name = "level3"> 5 </timer>
  </timer>
</timer>"""

        path = "xml_file"
        with open(path, "w") as file_:
            file_.write(xml)

        tree = self.timers._load_xml(path)

        expected_tree = ktree.Tree(
            "level1",
            1.0,
            (
                ktree.Tree("level2a", 2.0),
                ktree.Tree("level2b", 3.0),
                ktree.Tree("level2c", 4.0, (ktree.Tree("level3", 5.0),)),
            ),
        )

        self.assertEqual(tree, expected_tree)
        os.remove(path)

    def test_init(self):

        base = os.getcwd()

        self.assertDictEqual(
            self.timers.tree_files,
            {
                0: [
                    base + "/tests/tov/output-0000/static_tov/timertree.0.xml",
                    base + "/tests/tov/output-0001/static_tov/timertree.0.xml",
                ],
                1: [
                    base + "/tests/tov/output-0000/static_tov/timertree.1.xml",
                    base + "/tests/tov/output-0001/static_tov/timertree.1.xml",
                ],
            },
        )

    def test_getitem(self):
        # In the test files we have two identical timertree per process, so the
        # sum will be twice a given tree. We just check the first value

        self.assertEqual(self.timers[0].value, 2 * 1.6143)

    def test_median(self):
        # We just check the first value
        self.assertEqual(
            self.timers.median.value,
            median((self.timers[0].value, self.timers[1].value)),
        )

    def test_average(self):
        self.assertEqual(
            self.timers.average.value,
            mean((self.timers[0].value, self.timers[1].value)),
        )

    def test_keys(self):

        self.assertCountEqual(list(self.timers.keys()), [0, 1])

    def test_str(self):

        self.assertIn("[0, 1]", str(self.timers))
