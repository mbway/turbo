#!/usr/bin/env python3

import turbo.utils

import unittest


class UtilsTest(unittest.TestCase):
    def test_remap(self):
        # al => range a lower
        al, au = (5, 99)
        bl, bu = (0.5, 1.0)
        self.assertAlmostEqual(turbo.utils.remap(23, (al, au), (bl, bu)),
                               bl + (bu-bl)/(au-al) * (23-al))
        self.assertAlmostEqual(turbo.utils.remap(0.6, (bl, bu), (al, au)),
                               al + (au-al)/(bu-bl) * (0.6-bl))
