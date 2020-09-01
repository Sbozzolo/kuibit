#!/usr/bin/env python3

import unittest
import numpy as np
import h5py
from postcactus import timeseries as ts
from postcactus import simdir as sd
from postcactus import cactus_multipoles as mp
from postcactus import cactus_waves as cw
import time


class TestCactusWaves(unittest.TestCase):

    def test_WavesDet(self):

        t1 = np.linspace(0, np.pi, 100)
        t2 = np.linspace(2 * np.pi, 3 * np.pi, 100)
        y1 = np.sin(t1)
        y2 = np.sin(t2)
        ts1 = ts.TimeSeries(t1, y1)
        ts2 = ts.TimeSeries(t2, y2)
        dist1 = 100
        dist2 = 200

        data1 = [(2, 2, ts1), (2, 2, ts2)]
        data2 = [(1, 1, ts1), (1, 0, ts2),
                 (1, -1, ts2)]

        gw = cw.GravitationalWavesDet(dist1, data1)
        em = cw.ElectromagneticWavesDet(dist1, data2)


        self.assertEqual(gw.l_min, 2)
        self.assertEqual(em.l_min, 1)


    def test_WavesDir(self):

        # Test the error on wrong input type
        with self.assertRaises(TypeError):
            gwdir = cw.GravitationalWavesDir(0)

        with self.assertRaises(TypeError):
            emdir = cw.ElectromagneticWavesDir(0)

        sim = sd.SimDir("tests/tov")
        gwdir = cw.GravitationalWavesDir(sim)
