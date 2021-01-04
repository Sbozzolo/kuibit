#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
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

import unittest

from postcactus import attr_dict as ad


class TestAttrDict(unittest.TestCase):
    def test_AttributeDictionary(self):

        dictionary = {"first": "a", "b": 2}
        attr_dict = ad.AttributeDictionary(dictionary)

        with self.assertRaises(RuntimeError):
            attr_dict.b = 3

        with self.assertRaises(RuntimeError):
            attr_dict["b"] = 3

        self.assertEqual(attr_dict.first, "a")
        self.assertEqual(attr_dict["first"], "a")
        self.assertCountEqual(dir(attr_dict), ["first", "b"])
        self.assertCountEqual(attr_dict.keys(), ["first", "b"])
        self.assertEqual("Fields available:\n['first', 'b']", str(attr_dict))

        # Test attribute not available
        with self.assertRaises(AttributeError):
            attr_dict.hey

    def test_TransformDictionary(self):

        dictionary = {
            "first": [2, 1],
            "b": [3, 4],
            "third": {"nested": [6, 5]},
        }

        # Invalid input
        with self.assertRaises(TypeError):
            tran_dict = ad.TransformDictionary(1)

        tran_dict = ad.TransformDictionary(dictionary, transform=sorted)

        self.assertEqual(tran_dict["b"], [3, 4])
        self.assertEqual(tran_dict["third"]["nested"], [5, 6])
        self.assertTrue("first" in tran_dict)
        self.assertCountEqual(tran_dict.keys(), ["first", "b", "third"])

    def test_pythonize_name_dict(self):

        names = ["energy", "rho[0]"]

        pyth_dict = ad.pythonize_name_dict(names)

        self.assertEqual(pyth_dict.energy, "energy")
        self.assertEqual(pyth_dict.rho[0], "rho[0]")
