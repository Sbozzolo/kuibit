#!/usr/bin/env python3

# Copyright (C) 2023-2025 Gabriele Bozzola
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

"""Tests for kuibit.cactus_ascii_utils"""
import unittest
from io import StringIO

from kuibit import cactus_ascii_utils as cau


class TestASCIIUtils(unittest.TestCase):
    def test_scan_carpetx_header_norms(self):
        path = "tests/tovX/output-0000/static_tovX/norms/hydrobasex-rho.tsv"
        time_column, columns_description, is_comment = cau.scan_carpetx_header(
            path
        )
        self.assertTrue(is_comment)
        self.assertEqual(time_column, 1)
        self.assertEqual(columns_description["hydrobasex::rho.min"], 2)
        self.assertEqual(columns_description["hydrobasex::rho.max"], 3)
        self.assertEqual(columns_description["hydrobasex::rho.avg"], 5)
        self.assertEqual(columns_description["hydrobasex::rho.L1norm"], 8)
        self.assertEqual(columns_description["hydrobasex::rho.L2norm"], 9)
        self.assertEqual(columns_description["hydrobasex::rho.maxabs"], 10)

    def test_scan_carpetx_header_tsv(self):
        sim = "tests/carpetx/boxinbox"
        # Comment header after the IO::out_fileinfo comments, and plain header
        for path in (
            f"{sim}/output-0000/carpetx_tsv/boxinbox-positions.it000000.tsv",
            f"{sim}/output-0001/carpetx_tsv/boxinbox-positions.tsv",
        ):
            with self.subTest(path=path):
                time_column, columns, is_comment = cau.scan_carpetx_header(
                    path
                )
                self.assertEqual(is_comment, "output-0000" in path)
                self.assertEqual(time_column, 1)
                self.assertEqual(len(columns), 11)
                self.assertEqual(columns["iteration"], 0)
                self.assertEqual(columns["position_x[0]"], 2)
                self.assertEqual(columns["position_z[2]"], 10)

    def test_scan_carpetx_header_validation(self):
        tsv_data = {
            "plain.tsv": "iteration\ttime\trho\n0\t0.0\t1.25\n",
            "headerless.tsv": "0\t0.0\t1.25\n",
            "metadata-only.tsv": "# Date: today\n0\t0.0\t1.25\n",
            "missing-time.tsv": "iteration\trho\n0\t1.25\n",
            "malformed.tsv": "# 1:iteration 2:time 3:\n0\t0.0\t1.25\n",
            "duplicate-name.tsv": (
                "# 1:iteration 2:time 3:rho 4:rho\n0\t0.0\t1.25\t2.5\n"
            ),
        }

        def open_tsv(path, mode="r"):
            self.assertEqual(mode, "r")
            return StringIO(tsv_data[path])

        time_column, columns, is_comment = cau.scan_carpetx_header(
            "plain.tsv", opener=open_tsv
        )
        self.assertFalse(is_comment)
        self.assertEqual(time_column, 1)
        self.assertEqual(columns, {"iteration": 0, "time": 1, "rho": 2})

        invalid_headers = {
            "headerless.tsv": "No CarpetX column labels",
            "metadata-only.tsv": "No CarpetX column labels",
            "missing-time.tsv": "Missing time column",
            "malformed.tsv": "Bad CarpetX header",
            "duplicate-name.tsv": "Bad CarpetX header",
        }
        for path, error in invalid_headers.items():
            with self.subTest(path=path), self.assertRaisesRegex(
                RuntimeError, error
            ):
                cau.scan_carpetx_header(path, opener=open_tsv)

    def test_get_carpetx_variable_columns(self):
        for is_comment, separator in ((True, "."), (False, "_")):
            columns = {
                "iteration": 0,
                "time": 1,
                f"psi{separator}real": 2,
                f"psi{separator}imag": 3,
                "rho": 4,
            }
            with self.subTest(is_comment=is_comment):
                self.assertEqual(
                    cau.get_carpetx_variable_columns(
                        columns, {"iteration", "time"}, is_comment=is_comment
                    ),
                    {"psi": (2, 3), "rho": 4},
                )

    def test_get_carpetx_variable_columns_real(self):
        cases = (
            (False, {"psi_real": 2, "rho": 3, "psi_imag": 4}),
            (True, {"psi.real": 2, "rho": 3, "psi.imag": 4}),
            (True, {"psi_real": 2, "psi_imag": 3}),
            (True, {"psi_real": 2, "psi_imag": 3, "psi": 4}),
        )
        for is_comment, columns in cases:
            with self.subTest(is_comment=is_comment, columns=columns):
                self.assertEqual(
                    cau.get_carpetx_variable_columns(
                        columns, set(), is_comment=is_comment
                    ),
                    columns,
                )

    def test_total_filesize(self):
        file_with_dir = ["tests/tov", "tests/tov/log.txt"]

        with self.assertRaises(ValueError):
            cau.total_filesize(file_with_dir)

        files = ["tests/tov/log.txt", "tests/tov/ligo_sens.dat"]

        self.assertEqual(cau.total_filesize(files, unit="B"), 211972)
