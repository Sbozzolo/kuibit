#!/usr/bin/env python3

# Copyright (C) 2021 Gabriele Bozzola
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

"""The :py:mod:`~.utils` module contains functions used throughout the codebase.

In particular:
- :py:func:`~.get_logger` can be used to get kuibit's own logger.
- :py:func:`~.set_verbosity` modifies the format and output level for a given logger,
  by default kuibit's own logger.

"""


import logging


def get_logger(name="kuibit"):
    """Return the logger with the given name. By default, this returns kuibit's own
    logger.

    :returns: Logger.
    :rtype: logging.Logger

    """
    return logging.getLogger(name)


def set_verbosity(level="INFO", logger=None):
    """Set format and log level for the given logger.

    :param level: Desired level of the logger (INFO or DEBUG).
    :type level: str

    :param logger: Logger.
    :type logger: logging.Logger

    """

    if logger is None:
        logger = get_logger()

    if level not in ("DEBUG", "INFO"):
        raise TypeError("Verbosity level has to be either DEBUG or INFO")

    if level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
