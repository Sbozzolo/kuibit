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

"""The :py:mod:`~.hor_utils` module provides functions to perform common
operations using horizon(s).

The functions provided are:

- :py:func:`~.compute_center_of_mass`: compute the Newtonian center of mass for the
  given two horizons.
- :py:func:`~.compute_separation_vector`: compute the Newtonian separation vector
  between the centroids of the given two horizons.
- :py:func:`~.compute_separation`: compute the Newtonian separation between the centroids
  of the given two horizons.
- :py:func:`~.compute_angular_velocity_vector`: compute the Newtonian angular velocity vector
  for the given two horizons.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from kuibit.series import sample_common
from kuibit.tensor import Vector
from kuibit.timeseries import TimeSeries

if TYPE_CHECKING:  # pragma: no cover
    from kuibit.cactus_horizon import OneHorizon


def _two_centroids_as_Vectors(
    horizon1: OneHorizon, horizon2: OneHorizon, resample: bool = True
) -> Tuple[Vector[TimeSeries], Vector[TimeSeries]]:
    """Process horizon centroids.

    The information from the apparent horizons is used (contained in the
    BHDiagnostics files).

    Here we transform the position of the centroids into :py:class:`~.Vector`
    because they are easier to work with for vector operations.

    :param horizon1: First horizon. It has to have AH information.
    :type horizon1: :py:class:`~.OneHorizon`
    :param horizon2: Second horizon. It has to have AH information.
    :type horizon2: :py:class:`~.OneHorizon`

    :returns: Vectors with the centroid positions
    :rtype: Tuple of :py:class:`~.Vector`

    """

    if not horizon1.ah_available:
        raise RuntimeError(
            "Centroid information not available for first horizon"
        )

    if not horizon2.ah_available:
        raise RuntimeError(
            "Centroid information not available for second horizon"
        )

    # We add sample_common to make sure that everything is defined on the same
    # interval.
    (
        cen1_x,
        cen1_y,
        cen1_z,
        cen2_x,
        cen2_y,
        cen2_z,
    ) = sample_common(
        (
            horizon1.ah.centroid_x,
            horizon1.ah.centroid_y,
            horizon1.ah.centroid_z,
            horizon2.ah.centroid_x,
            horizon2.ah.centroid_y,
            horizon2.ah.centroid_z,
        ),
        resample=resample,
    )
    return Vector([cen1_x, cen1_y, cen1_z]), Vector([cen2_x, cen2_y, cen2_z])


def compute_separation_vector(
    horizon1: OneHorizon, horizon2: OneHorizon, resample: bool = True
) -> Vector[TimeSeries]:
    """Compute the vector coordinate separation between the centroids of two horizons.

    The information from the apparent horizons is used (contained in the
    BHDiagnostics files).

    The separation has sign and we compute it as first horizon - second horizon.


    :param horizon1: First horizon.
    :type horizon1: :py:class:`~.OneHorizon`
    :param horizon2: Second horizon.
    :type horizon2: :py:class:`~.OneHorizon`

    :returns: Coordinate distance vector between the two centroids,
              sampled over where both the horizons are available.
    :rtype: :py:class:`~.Vector` of :py:class:`~.TimeSeries`

    """
    ah1, ah2 = _two_centroids_as_Vectors(horizon1, horizon2, resample)
    return ah1 - ah2


def compute_separation(
    horizon1: OneHorizon, horizon2: OneHorizon, resample: bool = True
):
    """Compute the coordinate separation between the centroids of two horizons.

    The information from the apparent horizons is used (contained in the
    BHDiagnostics files).

    :param horizon1: First horizon.
    :type horizon1: :py:class:`~.OneHorizon`
    :param horizon2: Second horizon.
    :type horizon2: :py:class:`~.OneHorizon`

    :returns: Coordinate distance between the two centroids, sampled over both
              the horizons are available.
    :rtype: :py:class:`~.TimeSeries`

    """
    return compute_separation_vector(horizon1, horizon2, resample).norm()


def compute_center_of_mass(
    horizon1: OneHorizon, horizon2: OneHorizon, resample: bool = True
) -> Vector[TimeSeries]:
    """Compute the Newtonian center of mass between the centroids of two horizons.

    The information from the apparent horizons is used (contained in the
    BHDiagnostics files). The center of mass is computed with the irreducible
    mass (which is directly proportional to the area as computed by
    AHFinderDirect).

    :param horizon1: First horizon.
    :type horizon1: :py:class:`~.OneHorizon`
    :param horizon2: Second horizon.
    :type horizon2: :py:class:`~.OneHorizon`

    :returns: Center of mass vector between the two centroids, sampled over
              where both the horizons are available.
    :rtype: :py:class:`~.Vector` of :py:class:`~.TimeSeries`

    """
    cen1, cen2 = _two_centroids_as_Vectors(horizon1, horizon2, resample)

    # Here we use the fact that the irreducible_mass is proportional to the
    # area. We prefer using the area because it is always available when the
    # centroids are available.

    area1, area2 = sample_common(
        (
            horizon1.ah.area,
            horizon2.ah.area,
        ),
        resample=resample,
    )

    # This is morally "total mass" (irreducible_mass = area / 4pi)
    total_area = area1 + area2

    return area1 * cen1 / total_area + area2 * cen2 / total_area


def compute_angular_velocity_vector(
    horizon1: OneHorizon, horizon2: OneHorizon, resample: bool = True
):
    r"""Compute the angular velocity vector.

    The information from the apparent horizons is used (contained in the
    BHDiagnostics files).

    The angular velocity vector is computed as

    .. math::

      \Omega = \frac{\mathbf{r} \times \mathbf{r}}{r^2}

    :param horizon1: First horizon.
    :type horizon1: :py:class:`~.OneHorizon`
    :param horizon2: Second horizon.
    :type horizon2: :py:class:`~.OneHorizon`

    :returns: Vectors with the centroid positions
    :rtype: Tuple of :py:class:`~.Vector`

    """
    sep = compute_separation_vector(horizon1, horizon2, resample)
    dot_sep = sep.differentiated()
    sep_sq = sep.norm() ** 2

    # TODO (Future): Use Vector.cross

    numerator = Vector(
        [
            sep[1] * dot_sep[2] - sep[2] * dot_sep[1],
            sep[2] * dot_sep[0] - sep[0] * dot_sep[2],
            sep[0] * dot_sep[1] - sep[1] * dot_sep[0],
        ]
    )

    return numerator / sep_sq
