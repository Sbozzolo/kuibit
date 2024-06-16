#!/usr/bin/env python3

# Copyright (C) 2022-2024 Gabriele Bozzola
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

"""The :py:mod:`~.tree` module provides a data structure to work with trees.

The :py:class:`~.Tree` describes one node of a tree with a name and a value. The
node has any number of children, which are objects of the type
:py:class:`~.Tree` themselves. (There is no real distinction on what is a node
and what is a tree.) Given a :py:class:`~.Tree`, the children can be accessed by
index or by name. :py:class:`~.Tree` can also be exported to dictionaries or
JSON strings.

Different trees can be combined with the function :py:func:`~.merge_trees`,
which acts in the followin way. It takes any number of nodes that share the name
and create a new tree with that name. The children of this new tree are obtained
by combining all the children of the given nodes, and when children have the
same name, the new value is obtained by applying a given function (e.g., summing
up).

Currently, the main use of this structure in ``kuibit`` is for timers.

"""

# from __future__ import annotations allows us to postpone the evaluation of
# type hints. Without it, we would not be able to define type hints for a
# recursive structure.
from __future__ import annotations

import json
from typing import Callable, Iterable, Union


class Tree:
    """Represent one node of a tree (and recursively, the tree itself).

    :ivar name: Name of the node.
    :ivar value: Value of the node.
    :ivar children: Tuple with the nodes that are children of this one.
    """

    def __init__(
        self, name: str, value: float, children: tuple[Tree, ...] = ()
    ):
        """Constructor."""
        self.name = name
        self.value = value
        self.children = children

    @property
    def is_leaf(self) -> bool:
        """Return whether the node is a leaf node or not."""
        return len(self.children) == 0

    @property
    def tot_value_leaves(self) -> float:
        """Return the sum of all the values in all the levas.

        If the node has no children, return the value of the node itself.

        :returns: Cumulative values of the children.
        """
        if not self.is_leaf:
            return sum(child.tot_value_leaves for child in self.children)

        # We have no children
        return self.value

    def __getitem__(self, child_num_or_name: Union[int, str]) -> Tree:
        if isinstance(child_num_or_name, int):
            try:
                return self.children[child_num_or_name]
            except IndexError:
                raise KeyError(
                    f"Child number {child_num_or_name} not available"
                )
        else:
            for child in self.children:
                if child.name == child_num_or_name:
                    return child

            raise KeyError(f"Child named {child_num_or_name} not available")

    def to_dict(self) -> dict:
        """Convert the tree into a dictionary.

        The conversion happens in the following way: each node is converted into
        a dictionary with two or three elements: the name, the value, and, if
        there are children, a list of children. In turn, the children are
        represented as dictionaries in the same way.

        :returns: A dictionary representing the node and all its children.
        """

        ret_dict = {"name": self.name, "value": self.value}
        if not self.is_leaf:
            ret_dict["children"] = tuple(
                child.to_dict() for child in self.children
            )

        return ret_dict

    def to_json(self) -> str:
        """Convert the tree into a string with its JSON representation.

        :returns: String containing the JSON representation of the tree.
        """
        return json.dumps(self.to_dict())

    def __truediv__(self, value: float) -> Tree:
        """Return a new tree with all the values are divided by the given one."""

        # TODO (REFACTORING): Implement other arithmetic operations
        #
        # It would be straightforward to abstract away this method and implement
        # all the arithmetic operations. We just need to call the correct method
        # when setting value=...
        if not isinstance(value, (int, float)):
            raise TypeError("Given value is not a number")

        def walk_rec(node: Tree) -> Tree:
            return type(self)(
                name=node.name,
                value=node.value / value,
                children=tuple(map(walk_rec, node.children)),
            )

        return walk_rec(self)

    def __str__(self) -> str:
        return f"{self.name}: {self.value}"

    def __eq__(self, other: Tree) -> bool:
        return (
            self.name == other.name
            and self.value == other.value
            and self.children == other.children
        )

    def __hash__(self) -> int:
        """Return the hash of this object.

        Since the entire tree has to be traversed, this is an expensive
        operation!

        """
        # We add the hash mostly because we also have __eq__. Realistically,
        # this function should not be used. We combine hashes by taking the hash
        # of the tuple.
        return hash((self.name, self.value, self.children))


def merge_trees(
    trees: Iterable[Tree],
    merge_function: Callable[[Iterable[float]], float] = sum,
) -> Tree:
    """Combine multiple trees that start from the same place to a new one.

    When multiple nodes are found at the same level with the same name, apply
    `merge_function` to the list of values to generate the new value. The
    default is to sum them up, but another good idea would be to take the mean.

    The algorithm that mergers the tree is simple: it combines all the children
    of any given node (as identified by the name) across the tree. Therefore,
    the trees are meaningfully merged only if they are already relatively
    similar one with the other.

    :param trees: List of trees that have to be merged. They have to start from
                  a node with the same name.

    :param merge_function: Function that has to be applied to reduce the various
                           values to a single one.

    :returns: A new tree with all the nodes from the given trees.

    """

    def walk_rec(nodes: Iterable[Tree]) -> Tree:
        # Check if all the trees start from the same node. We put the names into a
        # set, if the size of the set is not 1, then it means we have different
        # initial nodes.
        root_names = {node.name for node in nodes}
        if len(root_names) != 1:
            raise RuntimeError("Trees do not start with the same base node")

        # This is the algorithm we implement to merge different trees. The
        # tricky part here is to extend the children list or aggregate children
        # with the same name using the function `merge_function`.
        #
        # First, we combine all the children in the same list. This is going to
        # include duplicates, i.e. children with the same name. The next step is
        # to apply this function recursively on each subgroup of children that
        # share the same name. In the process, we collect the Trees as new
        # children. Eventually, we reach the case in which we do not have any
        # children. Threfore, the list comprehension that activates the
        # recursion is not invoked (because child_groups is empty), so recursion
        # stops.

        # All the children including duplicates
        children: list[Tree] = []
        for node in nodes:
            children.extend(node.children)

        # Form the subgroups that share the same name. children_groups is a
        # dictionary that maps names with lists of Trees (having the same name)
        children_groups: dict[str, list[Tree]] = {}
        # We go over all the children and sort them
        for child in children:
            children_groups.setdefault(child.name, []).append(child)

        # Now, we apply recursively this function on the various subgroups
        pruned_children = [
            walk_rec(nodes) for nodes in children_groups.values()
        ]

        # root_names is a set, root_names.pop() is the only element
        return Tree(
            name=root_names.pop(),
            value=merge_function(node.value for node in nodes),
            children=tuple(pruned_children),
        )

    return walk_rec(trees)
