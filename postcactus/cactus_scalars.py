#!/usr/bin/env python3

"""The :py:mod:`~.cactus_scalars` module provides functions to load
timeseries in Cactus formats and a class :py:class:`ScalarsDir` for easy
access to all timeseries in a Cactus simulation directory. This module
is normally not used directly, but from the :py:mod:`~.simdir` module.
The data loaded by this module is represented as
:py:class:`~.TimeSeries` objects.
"""

from bz2 import BZ2File as bopen
from gzip import open as gopen


class CactusScalarASCII:
    """

    Compressed files (gz and bz2) are supported.
    """

    _pattern_filename = re.compile("^(\w+)((-(\w+))|(\[\d+\]))?\.(minimum|maximum|norm1|norm2|norm_inf|average)?\.asc(\.(gz|bz2))?$")

    _reduction_types={'minimum':'min',
                      'maximum':'max',
                      'norm1':'norm1',
                      'norm2':'norm2',
                      'norm_inf':'infnorm',
                      'average':'average',
                      None:'scalar'}

    _decompr = {None:open, 'gz':gopen, 'bz2':bopen}
