#!/usr/bin/env python3

# Copyright (C) 2023 Gabriele Bozzola
#
# Inspired by code originally developed by Wolfgang Kastaun. This file may
# contain algorithms and/or structures first implemented in
# GitHub:wokast/PyCactus/PostCactus/cactus_scalars.py
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

"""Tests for kuibit.cactus_ascii_utils
"""
import unittest

from kuibit import cactus_ascii_utils as cau


class TestASCIIUtils(unittest.TestCase):
    def test_total_filesize(self):
        file_with_dir = ["tests/tov", "tests/tov/log.txt"]

        with self.assertRaises(ValueError):
            cau.total_filesize(file_with_dir)

        files = ["tests/tov/log.txt", "tests/tov/ligo_sens.dat"]

        self.assertEqual(cau.total_filesize(files, unit="B"), 211972)
