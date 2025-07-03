#!/usr/bin/env python3

# Copyright (C) 2022-2025 Gabriele Bozzola
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

from kuibit import tree as kt


class TestTree(unittest.TestCase):
    def setUp(self):
        self.single_node = kt.Tree("myname", 2.0)
        self.one_branch = kt.Tree("myname2", 3.0, (self.single_node,))
        self.two_branches = kt.Tree(
            "myname3", 4.0, (self.single_node, self.one_branch)
        )

    def test_init(self):
        # Test with a simple node
        self.assertEqual(self.single_node.name, "myname")
        self.assertEqual(self.single_node.value, 2.0)
        self.assertEqual(self.single_node.children, ())

        # Now we can add this to a more complex tree

        self.assertEqual(self.one_branch.children, (self.single_node,))

    def test_is_left(self):
        self.assertTrue(self.single_node.is_leaf)
        self.assertFalse(self.one_branch.is_leaf)

    def test_tot_value_leaves(self):
        self.assertEqual(self.single_node.tot_value_leaves, 2)
        self.assertEqual(self.one_branch.tot_value_leaves, 2)
        self.assertEqual(self.two_branches.tot_value_leaves, 4)

    def test_eq(self):
        self.assertEqual(self.single_node, self.single_node)
        self.assertNotEqual(self.single_node, self.one_branch)
        self.assertEqual(self.one_branch, self.one_branch)

    def test_hash(self):
        self.assertEqual(
            hash(self.single_node),
            hash(
                (
                    self.single_node.name,
                    self.single_node.value,
                    self.single_node.children,
                )
            ),
        )

    def test_getitem(self):
        self.assertEqual(self.one_branch[0], self.single_node)
        self.assertEqual(self.one_branch["myname"], self.single_node)

        with self.assertRaises(KeyError):
            self.one_branch[2]

        with self.assertRaises(KeyError):
            self.one_branch["bob"]

    def test_str(self):
        self.assertEqual(str(self.one_branch), "myname2: 3.0")

    def test_truediv(self):
        with self.assertRaises(TypeError):
            self.two_branches / "bob"

        divided = self.two_branches / 2

        self.assertEqual(divided.value, 2)
        self.assertEqual(divided[0].value, 1)
        self.assertEqual(divided[1].value, 1.5)
        self.assertEqual(divided[1][0].value, 1)

    def test_to_dict(self):
        self.assertDictEqual(
            self.one_branch.to_dict(),
            {
                "name": "myname2",
                "value": 3.0,
                "children": ({"name": "myname", "value": 2.0},),
            },
        )

    def test_to_json(self):
        self.assertEqual(
            self.one_branch.to_json(),
            '{"name": "myname2", "value": 3.0, "children": [{"name": "myname", "value": 2.0}]}',
        )

    def test_merge_trees(self):
        # Test unmergable tree (different base name)
        with self.assertRaises(RuntimeError):
            kt.merge_trees((self.single_node, self.one_branch))

        other_tree = kt.Tree(
            "myname3",
            5.0,
            (self.single_node, self.one_branch, kt.Tree("myname4", 7.0)),
        )

        expected_tree = kt.Tree(
            "myname3",
            9.0,
            children=(
                kt.Tree("myname", 4.0),
                kt.Tree("myname2", 6.0, children=(kt.Tree("myname", 4.0),)),
                kt.Tree("myname4", 7.0),
            ),
        )

        self.assertEqual(
            kt.merge_trees((self.two_branches, other_tree)), expected_tree
        )
