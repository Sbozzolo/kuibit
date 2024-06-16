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

"""The :py:mod:`~.cactus_twopunctures` module reads the metadata output by TwoPunctures.

The module is rather straightforward and has only one dictionary-like object
that reads the ``TwoPunctures.bbh`` file created by ``TwoPunctures``.

"""
from __future__ import annotations

import configparser
from collections.abc import KeysView
from typing import Union


class TwoPuncturesDir:
    """Read and process metadata from TwoPunctures.

    This object represents the metadata read from `TwoPunctures.bbh` as a
    dictionary.
    """

    def __init__(self, sd) -> None:
        """Read and process the `TwoPunctures.bbh` file, if in the SimDir."""

        # This is where we store the output
        self._metadata = None

        _metadata_files = [
            path for path in sd.allfiles if path.endswith("TwoPunctures.bbh")
        ]

        if _metadata_files:
            if len(_metadata_files) > 1:
                raise RuntimeError(
                    f"Multiple TwoPunctures.bbh found: {_metadata_files}"
                )

            # We have only one TwoPunctures.bbh
            #
            # The structure of the file is essentially a .ini file, so we can
            # read it with configparser
            data = configparser.ConfigParser()
            data.read(*_metadata_files)

            def try_convert(what: Union[str, float]) -> Union[str, float]:
                # Convert what to a float if it makes sense
                try:
                    ret = float(what)
                except ValueError:
                    ret = what
                return ret

            self._metadata = {
                k: try_convert(v) for k, v in data["metadata"].items()
            }

    @property
    def has_metadata(self) -> bool:
        """Was a TwoPunctures.bbh file found and read?"""
        return self._metadata is not None

    def __getitem__(self, key: str) -> Union[str, float]:
        if self.has_metadata:
            # Everything is stored lowercase
            key_l = key.lower()
            if key_l in self._metadata:
                return self._metadata[key_l]
            raise KeyError(f"{key} is not a metadata for TwoPunctures")
        raise RuntimeError("Metadata not available")

    def keys(self) -> KeysView:
        if self.has_metadata:
            return self._metadata.keys()
        return {}.keys()
