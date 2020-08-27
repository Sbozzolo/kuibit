#!/usr/bin/env python3

import unittest
from postcactus import cactus_scalars as cs


class TestCactusScalar(unittest.TestCase):

    def test_init(self):

        # Filename not recogonized
        with self.assertRaises(RuntimeError):
            cs.CactusScalarASCII("123.h5")

        # Reduction not recogonized
        with self.assertRaises(RuntimeError):
            cs.CactusScalarASCII("hydrobase-press.bubu.asc")

        # maximum, vector, one file per variable
        path = "tests/tov/output-0000/static_tov/vel[0].maximum.asc"
        asc = cs.CactusScalarASCII(path)

        self.assertFalse(asc._is_one_file_per_group)
        self.assertFalse(asc._was_header_scanned)
        self.assertEqual(asc.reduction_type, "maximum")
        self.assertDictEqual(asc._vars, {'vel[0]': None})

        # no reduction, scalar, one file per group
        path = "tests/tov/output-0000/static_tov/carpet-timing..asc"
        asc_carp = cs.CactusScalarASCII(path)

        self.assertTrue(asc_carp._is_one_file_per_group)
        self.assertFalse(asc_carp._was_header_scanned)
        self.assertIs(asc_carp.reduction_type, 'scalar')

        # Compressed, scalar, one file per group
        path = "tests/tov/output-0000/static_tov/hydrobase-eps.minimum.asc.gz"
        asc_gz = cs.CactusScalarASCII(path)

        self.assertTrue(asc_gz._is_one_file_per_group)
        self.assertFalse(asc_gz._was_header_scanned)
        self.assertEqual(asc_gz.reduction_type, "minimum")
        self.assertEqual(asc_gz._compression, "gz")

        # Compressed, scalar, one file per group
        path = "tests/tov/output-0000/static_tov/hydrobase-eps.minimum.asc.bz2"
        asc_bz = cs.CactusScalarASCII(path)
        self.assertEqual(asc_bz._compression, "bz2")
