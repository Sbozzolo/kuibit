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

"""The :py:mod:`~.tensor` module provides abstractions to work with tensorial
containers and similar.

The main object defined here is :py:class:`~.Tensor`, which is a that acts as
container for any object :py:class:`~.BaseNumerical`, as in
:py:class:`~.TimeSeries`, or :py:class:`~.UniformGridData`. :py:class:`~.Tensor`
behaves as one would expect them to behave: they are high-level interfaces that
support all the mathematical operations. :py:class:`~.Tensor` also inherit the
attributes from their contained object. For example, if the :py:class:`~.Tensor`
contains :py:class:`~.TimeSeries`, the tensor will dynamically acquire all the
methods in such class.

In pratice, the main way to use :py:class:`~.Tensor` objects is through its
derived classes :py:class:`~.Vector` and :py:class:`~.Matrix`, which implement
all those features one might expect from vector calculus.

Consider the following usage example:

.. code-block:: python

   import numpy as np

   from kuibit.timeseries import TimeSeries
   from kuibit.tensor import Vector

   # Fake some data that describe the x, y position of two black holes
   times = np.linspace(0, 2 * np.pi, 100)

   bh1_x = np.sin(times)
   bh1_y = np.cos(times)

   # Not really realistic, but it is fake data
   bh2_x = np.sin(2 * times)
   bh2_y = np.cos(2 * times)

   bh1_centroid = Vector([TimeSeries(times, bh1_x), TimeSeries(times, bh1_y)])
   bh2_centroid = Vector([TimeSeries(times, bh2_x), TimeSeries(times, bh2_y)])

   # If we want to compute vx, vy
   bh1_velocity = bh_centroid.differentiated()  # This is a Vector

   # For the distance with another
   distance = bh1_centroid - bh2_centroid  # This is a Vector

   # The magnitude of the distance
   distance = distance.norm()   # This is a TimeSeries

"""


from __future__ import annotations

from inspect import getsource  # Used in __getattr__
from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np
import numpy.typing as npty

from kuibit.numerical import BaseNumerical

_T = TypeVar("_T", bound=BaseNumerical)


class Tensor(Generic[_T], BaseNumerical):
    """Represents a mathematical hyper-matrix (a Tensor) in N dimensions.

    At the moment, it is only used as a base class for :py:class:`~.Vector` and
    :py:class:`~.Matrix`. So, the class is currently not intended for direct use
    (as it does not have a lot of features). This class is not fully flashed out
    for the general case. It is here mostly as a stub for the two subclasses
    that are below. Implementing a generic tensor class is not trivial and it
    left as an exercise to the reader.

    This class implements the basic infrastrucutre used by :py:class:`~.Vector`
    and :py:class:`~.Matrix`. In particular, it fullfills the requirements set
    by :py:class:`~.BaseNumerical` and it implements a method to broadcast
    attributes from the contained objects to the container itself.

    All the operations are applied component-wise. So, for example, a tensor
    times a tensor is going to be a tensor with the same shape and elements
    multiplied. Reductions return NumPy arrays.

    ..note::

        For efficicency of implmentation, data is sotred without hierarchy
       (i.e., "flattened out"). Therefore, the operations that require
       reconstructing the structure have some small overhead. When, possible
       work with flattened data.


    """

    def __init__(self, data: Collection[_T]):
        """Construct the Tensor by checking and processing the data.

        The data is flattened out and it shape is saved. The shape here is
        defined as the number of elements of the tensor along each dimension.
        For example, for a vector it would only by its length.

        :param data: Representation of the tensor as nested lists.
        :type data: Nested list of data derived from :py:class:`~.BaseNumerical`.

        """
        # We are going to make a design choice here. If we flatten the data, all
        # the operations that are component-wise will become very easy to
        # implement. Also operations that rearrange the indices will become easy
        # The trade-off is that operations that need to know about
        # the actual shape (e.g., maxtrix multiplication) will have to be
        # implemented more carefully.
        if not hasattr(data, "__len__"):
            raise TypeError("data has to be iterable")

        if len(data) == 0:
            raise TypeError("data is empty")

        # Big brain algorithm follows. We want to allow only tensors that have a
        # well defined structure and we want to record this structure. Consider
        # the simple example of a matrix. Matrices look like [[a1, a2], [a3,
        # a4]]. The shape of this matrix would be (2, 2). It does not make sense
        # to have something like [[a1, a2], [a3, [a4]]] (notice the second []
        # around a4). So, below we recursively walk through the data as if it
        # was a tree. We know we reached the lowest level when we find object of
        # type BaseNumerical. The first time we reach the lowest level, we
        # record the depth of that level since all the branches must have the
        # same depth. If any branch has a different depth, it means that the
        # tree is not an hyper-matrix. The second case in which a tree would not
        # be broadcastable as a hyper-matrix is that the number of elements is
        # different for different branches at fixed height, as in this example
        # [[a1, a2], [a3, a4, a5]]. So, at each level we record how many
        # elements we have by putting them in sets. If any set has more than one
        # element at any given time, it means that the data is not like a matrix.

        flattened: List[_T] = []
        shape: Dict[int, Set[int]] = {}

        # Recursively walk through the input, recording the shape
        def _walk(
            data, height: int = 0, depth_of_first_leaf: Optional[int] = None
        ):
            """Walk recursively the tree finding inconsistencies, fattening the
            data, and recording the shape.

            """

            shape.setdefault(height, set()).add(len(data))
            if len(shape[height]) != 1:
                raise RuntimeError("The shape of the data is inconsistent")

            for d in data:
                if isinstance(d, BaseNumerical):
                    # This is a leaf node

                    # Set depth_of_first_leaf to height if it is None
                    depth_of_first_leaf = depth_of_first_leaf or height

                    if height != depth_of_first_leaf:
                        raise RuntimeError("The data has inconsistent depth")

                    flattened.append(d)
                else:
                    depth_of_first_leaf = _walk(
                        d, height + 1, depth_of_first_leaf
                    )

            # The variable `depth_of_first_leaf` has to be carried down all the
            # way, but also "up" from the bottom of the tree. That's why it is
            # the return value, so that we can pass it up when we find it.
            return depth_of_first_leaf

        _walk(data)

        # Change the sets into normal ints
        self.__shape = tuple(v.pop() for v in shape.values())

        self.__flat_data = flattened

        # Check that the type is the same for all the data
        if not all(isinstance(d, self.type) for d in self.flat_data):
            raise TypeError("data has to be of one specific type")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the Tensor.

        The shape is defined as the number of elements in each dimension.

        :returns: Shape of the tensor.
        :rtype: Tuple of ints

        """
        return self.__shape

    @classmethod
    def from_shape_and_flat_data(
        cls, shape: Collection[int], flat_data: Collection[_T]
    ) -> Tensor[_T]:
        """Create a new :py:class:`~.Tensor` from flat data and shape.

        No checks are performed.
        """
        # We set the attributes directly, so we don't use __init__ but __new__.
        ret = cls.__new__(cls)
        ret.__shape = tuple(shape)
        ret.__flat_data = list(flat_data)
        return ret

    @property
    def flat_data(self) -> List[_T]:
        """Return the data in the tensor as a flat list.

        :returns: List with all the data
        :rtype: List of type.
        """
        return self.__flat_data

    def _restructure_data(self, flat_data):
        """Take flatten data and reshape it to the correct tensorial form.

        :param flat_data: Data flattened out
        :type: list

        :returns: Data restructured according to self.shape.
        :rtype: Nested list
        """

        # We make groups of groups, recursively, starting from the individual
        # elements.
        #
        # Consider the example with shape (1, 2, 3) and data = [1, 2, 3, 4, 5,
        # 6]. First, we make groups of three: data = [[1, 2, 3], [4, 5, 6]],
        # then we make groups of two: data = [[[1, 2, 3], [4, 5, 6]]], and then
        # we make another group of one. data = [[[[1, 2, 3], [4, 5, 6]]]]. We
        # end up with an extra set of brackets that is due to the fact that we
        # always return a list. So, the return value of this function peels off
        # that last layer.

        def _reconstruct(data, size):
            # Make groups of length `size`
            ret = [data[i : i + size] for i in range(0, len(data), size)]
            return ret

        data = flat_data

        for size in reversed(self.shape):
            data = _reconstruct(data, size)

        # The return value is always a list, so we end up with an extra level of
        # lists, which we remove
        return data[0]

    # TODO: I do not know how to type-hint a return value that is nested lists
    # with an unknown number of levels.
    @property
    def data(self):
        """Return the data with its tensorial structure.

        For example, if this was a Matrix, the return value would be a list of
        lists.

        Note, this operation is not free! Data is stored flat, so recomputing
        the structure has some overhead.

        :returns: Data structured according to self.shape
        :rtype: Nested lists
        """
        return self._restructure_data(self.flat_data)

    @property
    def type(self) -> type:
        """Return the type of the contained object.

        :returns: Type of the object, a class derived from
                  :py:class:`~.BaseNumerical`
        :rtype: type
        """
        return type(self.__flat_data[0])

    def _apply_unary(self, function, *args, **kwargs) -> Tensor[_T]:
        return type(self).from_shape_and_flat_data(
            self.shape,
            [
                # skipcq PYL-W0212
                x._apply_unary(function, *args, **kwargs)
                for x in self.flat_data
            ],
        )

    def _apply_binary(self, other, function, *args, **kwargs) -> Tensor[_T]:
        if isinstance(other, type(self)):
            if self.type != other.type:
                raise ValueError("Incompatible base types")

            if self.shape != other.shape:
                raise ValueError("Tensors do not have same shape")
            else:
                ret = type(self).from_shape_and_flat_data(
                    self.shape,
                    [
                        # skipcq PYL-W0212
                        x._apply_binary(y, function, *args, **kwargs)
                        for x, y in zip(self.flat_data, other.flat_data)
                    ],
                )
        else:
            ret = type(self).from_shape_and_flat_data(
                self.shape,
                [
                    # skipcq PYL-W0212
                    x._apply_binary(other, function, *args, **kwargs)
                    for x in self.flat_data
                ],
            )
        return ret

    def _apply_reduction(self, reduction, *args, **kwargs) -> npty.NDArray:
        return np.array(
            self._restructure_data(
                [
                    # skipcq PYL-W0212
                    x._apply_reduction(reduction, *args, **kwargs)
                    for x in self.flat_data
                ]
            )
        )

    def _apply_to_self(self, function, *args, **kwargs):
        self.__flat_data = [
            function(x, *args, **kwargs) for x in self.flat_data
        ]

    def __call__(self, val) -> npty.NDArray:
        try:
            ret = np.array(
                self._restructure_data([x(val) for x in self.flat_data])
            )
        except AttributeError as e:
            raise AttributeError(f"{self.type} cannot ba called") from e

        return ret

    def copy(self) -> Tensor[_T]:
        """Return a deep copy of the object.

        :return: Deep copy
        :rtype: Tensor
        """
        try:
            ret = type(self).from_shape_and_flat_data(
                self.shape,
                [x.copy() for x in self.flat_data],
            )
        except AttributeError as e:
            raise AttributeError(
                f"{self.type} does not have a copy method"
            ) from e

        return ret

    def __matmul__(self, other):
        return NotImplemented

    def __rmatmul__(self, other):
        return NotImplemented

    def __getattr__(self, attr: str):
        # This is how we transfer all the methods in _T to the Tensor itself.
        #
        # For example, if T is a TimeSeries, we would like to be able to call
        # .partial_differentiated for all the components and create a vector out
        # of it. With this, if V is a Vector of T, we will be able to call
        # V.partial_differentiated and get the desired output.
        #
        # The return value of __getattr__ is a new function, _apply_attr. This
        # _apply_attr is such that it calls the attribute we want to call with
        # the arguments we pass. In addition to that, we do some basic type
        # checking to return the correct type. The only caveat here is that the
        # attribute has to return T or a number.

        # We broadcast only those attributes that are in the base object
        if not hasattr(self.flat_data[0], attr):
            raise AttributeError(f"No attribute {attr} in {self.type}")

        # There are some nasty things that we have to do to ensure nice
        # compatibility: we need to handle @property methods and methods that
        # edit the object in place. For the first, we first need to see if the
        # attribute is a property or not. Property attributes are not callable.

        # Note, we are applying getattr to the class, not to the instance!
        is_property = isinstance(getattr(self.type, attr), property)

        # For the latter, we are going to inspect the source code of the method
        # as see if it calls self._apply_to_self, in which case we will do the
        # same.
        if is_property:
            source_code = getsource(getattr(self.type, attr).fget)
        else:
            source_code = getsource(getattr(self.type, attr))

        applies_to_self = "self._apply_to_self" in source_code

        # If we have a property, we should not return a callable
        if is_property:
            new_data = [getattr(x, attr) for x in self.flat_data]

            if isinstance(new_data[0], self.type):
                return type(self).from_shape_and_flat_data(
                    self.shape, new_data
                )

            if isinstance(new_data[0], (float, complex, np.ndarray)):
                return np.array(new_data).reshape(self.shape)

            raise NotImplementedError(
                "Attribute broadcasting works only for methods that return numbers or type(self)"
            )

        # If we have don't a property, we should return a callable that does
        # pretty much the same thing

        def _apply_attr(*args, **kwargs):
            new_data = [
                getattr(x, attr)(*args, **kwargs) for x in self.flat_data
            ]
            if isinstance(new_data[0], self.type):
                return type(self).from_shape_and_flat_data(
                    self.shape, new_data
                )
            if isinstance(new_data[0], (float, complex, np.ndarray)):
                return np.array(new_data).reshape(self.shape)

            # If the return type is None, this might mean that we are editing
            # something in place. In that case, there's nothing we have to do,
            # as the objects already changed
            if applies_to_self:
                return None

            raise NotImplementedError(
                "Attribute broadcasting works only for methods that return numbers or type(self)"
            )

        return _apply_attr

    def __eq__(self, other: Any) -> bool:
        # Let's take advantage of some shortcircuiting to reduce comparisons. We
        # check three conditions that would establish inequality, and negate the
        # answer. The reason is that if any of them fails, we wouldn't have to
        # go to the next one. This is particularly relevant for the last
        # condition, which is a little expensive to compute.
        not_equal = (
            not isinstance(other, type(self))
            or self.shape != other.shape
            or any(
                s != i
                for s, i in zip(iter(self.flat_data), iter(other.flat_data))
            )
        )
        return not not_equal

    # From Python's docs: In order to conform to the object model, classes that
    # define their own equality method should also define their own hash method,
    # or be unhashable.

    # Since we consider series unhashable, this object also has to be unhashable.
    __hash__ = None


class Vector(Tensor[_T]):
    """Represents a vector in the mathematical sense.

    It can be used with series, or grid data, or anything that is derived from
    :py:class:`~.BaseNumerical`.

    This abstraction is useful for vector operations: for example, taking the
    cross/dot products between two vectors.

    All the operations are component-wise, and the vector inherits all the
    methods available to the base object.

    """

    def __len__(self) -> int:
        """Length of the vector."""
        return len(self.flat_data)

    def __getitem__(self, i: int) -> _T:
        return self.flat_data[i]

    def dot(self, other: Vector[_T]) -> _T:
        """Return the dot product with another vector.

        :param other: Other vector in the dot product.
        :type other: Vector

        :returns: Dot product
        :rtype: Same as self.type
        """
        if not isinstance(other, Vector):
            raise TypeError("other is not a Vector")

        if self.shape != other.shape:
            raise RuntimeError("Incosistent shape")

        if self.type != other.type:
            raise TypeError("Incosistent base type")

        return sum(s * i for s, i in zip(self, other))

    def norm(self):
        """Return the norm of the vector.

        :returns: Norm of the vector
        :rtype: Same as self.type
        """
        return self.dot(self).sqrt()


class Matrix(Tensor[_T]):
    """Represents a matrix in the mathematical sense.

    It can be used with series, or grid data, or anything that is derived from
    :py:class:`~.BaseNumerical`.

    This abstraction is useful for matrix operations: for example, taking the
    determinant.

    All the operations are component-wise, and the matrix inherits all the
    methods available to the base object.

    """

    def __getitem__(self, ij: Tuple[int]) -> _T:
        # Let's see an example, the matrix [[0, 1, 2], [3, 4, 5]] has shape (2,
        # 3) = (X, Y). If we want element (1, 2) = (i, j), we would have to get
        # index i * Y + j
        i, j = ij
        _, Y = self.shape

        return self.flat_data[i * Y + j]
