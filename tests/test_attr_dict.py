#!/usr/bin/env python3

import unittest
from postcactus import attr_dict as ad


class TestAttrDict(unittest.TestCase):

    def test_AttributeDictionary(self):

        dictionary = {'first': 'a', 'b': 2}
        attr_dict = ad.AttributeDictionary(dictionary)

        with self.assertRaises(RuntimeError):
            attr_dict.b = 3

        self.assertEqual(attr_dict.first, 'a')
        self.assertCountEqual(dir(attr_dict), ['first', 'b'])

    def test_TransformDictionary(self):

        dictionary = {'first': [2, 1], 'b': [3, 4],
                      'third': {'nested': [6, 5]} }

        # Invalid input
        with self.assertRaises(TypeError):
            tran_dict = ad.TransformDictionary(1)

        tran_dict = ad.TransformDictionary(dictionary,
                                           transform=sorted)

        self.assertEqual(tran_dict['b'], [3, 4])
        self.assertEqual(tran_dict['third']['nested'], [5, 6])
        self.assertTrue('first' in tran_dict)
        self.assertCountEqual(tran_dict.keys(), ['first', 'b', 'third'])

    def test_pythonize_name_dict(self):

        names = ['energy', 'rho[0]']

        pyth_dict = ad.pythonize_name_dict(names)

        self.assertEqual(pyth_dict.energy, 'energy')
        self.assertEqual(pyth_dict.rho[0], 'rho[0]')
